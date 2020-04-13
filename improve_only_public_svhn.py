# coding=utf-8
# Copyright 2020 The Adp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Improve dp classifier using ONLY public data w sgd."""

import copy
import itertools
import logging
import pickle
import time
import warnings

from absl import app
from absl import flags
from jax import grad
from jax import jit
from jax import partial
from jax import random
from jax import tree_util
from jax import vmap
from jax.experimental import optimizers
from jax.experimental import stax
from jax.lax import stop_gradient
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import seaborn as sns
import sklearn.cluster
import sklearn.decomposition
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.compat.v1.io import gfile
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from adp import data
from adp import datasets
from adp import utils


FLAGS = flags.FLAGS
flags.DEFINE_string('train_split', 'svhn_cropped-train',
                    'dataset-split used for training dp classifier')
flags.DEFINE_string('test_split', 'svhn_cropped-test',
                    'dataset-split used for evaluating dp classifier')
flags.DEFINE_string('pool_split', 'svhn_cropped-extra',
                    'dataset-split used as extra candidate pool for AL')
flags.DEFINE_boolean('augment_data', False,
                     'augment data with flip, crop, etc.')
flags.DEFINE_boolean('visualize', False,
                     'visualize data')
flags.DEFINE_string('pretrained_dir', None,
                    'Path to pretrained model')
flags.DEFINE_string('root_dir', None,
                    'Root dir from which we load checkpoint')
flags.DEFINE_string('exp_dir', None,
                    'Experiment dir for this launch')
flags.DEFINE_string('work_dir', None,
                    'Specific dir of the current work unit within this launch')
flags.DEFINE_string('ckpt_idx', None,
                    'worker_folder/ckpt_folder')
flags.DEFINE_boolean('dpsgd', False,
                     'True with DP-SGD; False with vanilla SGD.')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0,
                   'Clipping norm')
flags.DEFINE_float('learning_rate', .10,
                   'Learning rate for finetuning.')
flags.DEFINE_integer('batch_size', 256,
                     'Batch size')
flags.DEFINE_integer('epochs', 100,
                     'Number of finetuning epochs')
flags.DEFINE_integer('seed', 0,
                     'Seed for jax PRNG')
flags.DEFINE_integer('uncertain', 0,
                     'entropy, or 1: difference between 1st_prob and 2nd_prob')
flags.DEFINE_integer('k_components', 16,
                     'number of PCs to visualize and optimize')
flags.DEFINE_integer('distance', 0,
                     'euclidean, or 1: weighted_euclidean by PCA singular vals')
flags.DEFINE_string('clustering', 'KMeans',
                    'clustering method for projected embeddings')
flags.DEFINE_integer('ppc', 10,
                     'number of examples picked per cluster')
flags.DEFINE_integer('n_extra', 3000,
                     'number of extra points')
flags.DEFINE_integer('uncertain_extra', 1000,
                     'n_uncertain - n_extra')
flags.DEFINE_bool('show_label', True,
                  'visualize predicted label at top/left, true at bottom/right')


# BEGIN: define the classifier model
init_fn_0, apply_fn_0 = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Tanh,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Tanh,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,  # (-1, 800)
    stax.Dense(32),
    stax.Tanh,  # embeddings
)


init_fn_1, apply_fn_1 = stax.serial(
    stax.Dense(10),  # logits
)


def predict(params, inputs):
  params_0 = params[:-1]
  params_1 = params[-1:]
  embeddings = apply_fn_0(params_0, inputs)
  logits = apply_fn_1(params_1, embeddings)
  return logits
# END: define the classifier model


def loss(params, batch):
  inputs, targets = batch
  logits = predict(params, inputs)
  logits = stax.logsoftmax(logits)  # log normalize
  return -np.mean(np.sum(logits * targets, 1))  # cross entropy loss
grad_loss = jit(grad(loss))


def accuracy(params, batch, return_predicted_class=False):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs), axis=1)
  acc = np.mean(predicted_class == target_class)
  if return_predicted_class:
    return acc, predicted_class
  else:
    return acc


def weighted_euclidean_distances(matrix_x, vector_y, vector_w):
  matrix_z = matrix_x - vector_y
  matrix_d = matrix_z * matrix_z * vector_w
  return onp.sqrt(matrix_d.sum(axis=1, keepdims=False))


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size):
  """Return differentially private gradients for params, evaluated on batch."""

  def _clipped_grad(params, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad_loss(params, single_example_batch)

    nonempty_grads, tree_def = tree_util.tree_flatten(grads)
    total_grad_norm = np.linalg.norm(
        [np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = stop_gradient(np.amax((total_grad_norm / l2_norm_clip, 1.)))
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

  px_clipped_grad_fn = vmap(partial(_clipped_grad, params))
  std_dev = l2_norm_clip * noise_multiplier
  noise_ = lambda n: n + std_dev * random.normal(rng, n.shape)
  normalize_ = lambda n: n / float(batch_size)
  tree_map = tree_util.tree_map
  sum_ = lambda n: np.sum(n, 0)  # aggregate
  aggregated_clipped_grads = tree_map(sum_, px_clipped_grad_fn(batch))
  noised_aggregated_clipped_grads = tree_map(noise_, aggregated_clipped_grads)
  normalized_noised_aggregated_clipped_grads = (
      tree_map(normalize_, noised_aggregated_clipped_grads)
  )
  return normalized_noised_aggregated_clipped_grads


def shape_as_image(images, labels, dummy_dim=False):
  target_shape = (-1, 1, 32, 32, 3) if dummy_dim else (-1, 32, 32, 3)
  return np.reshape(images, target_shape), labels


def compute_epsilon(steps, num_examples=60000, target_delta=1e-5):
  if num_examples * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = FLAGS.batch_size / float(num_examples)
  orders = list(np.linspace(1.1, 10.9, 99)) + range(11, 64)
  rdp_const = compute_rdp(q, FLAGS.noise_multiplier, steps, orders)
  eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
  return eps


def get_cluster_method(clustering):
  """Clustering methods."""
  if 'KMeans' in clustering:
    nc = int(clustering.split('-')[-1])
    return sklearn.cluster.KMeans(n_clusters=nc)

  elif 'SpectC' in clustering:
    nc = int(clustering.split('-')[-1])
    return sklearn.cluster.SpectralClustering(n_clusters=nc)

  elif 'AggloC' in clustering:
    nc = int(clustering.split('-')[-1])
    return sklearn.cluster.AgglomerativeClustering(n_clusters=nc)

  elif 'AfProp' in clustering:
    dg = float(clustering.split('-')[-1]) / 10
    return sklearn.cluster.AffinityPropagation(damping=dg)

  elif 'MShift' in clustering:
    return sklearn.cluster.MeanShift()


def main(_):
  sns.set()
  sns.set_palette(sns.color_palette('hls', 10))
  npr.seed(FLAGS.seed)

  logging.info('Starting experiment.')

  # Create model folder for outputs
  try:
    gfile.MakeDirs(FLAGS.work_dir)
  except gfile.GOSError:
    pass
  stdout_log = gfile.Open('{}/stdout.log'.format(FLAGS.work_dir), 'w+')

  # use mean/std of svhn train
  train_images, _, _ = datasets.get_dataset_split(
      name=FLAGS.train_split.split('-')[0],
      split=FLAGS.train_split.split('-')[1],
      shuffle=False)
  train_mu, train_std = onp.mean(train_images), onp.std(train_images)
  del train_images

  # BEGIN: fetch test data and candidate pool
  test_images, test_labels, _ = datasets.get_dataset_split(
      name=FLAGS.test_split.split('-')[0],
      split=FLAGS.test_split.split('-')[1],
      shuffle=False)
  pool_images, pool_labels, _ = datasets.get_dataset_split(
      name=FLAGS.pool_split.split('-')[0],
      split=FLAGS.pool_split.split('-')[1],
      shuffle=False)

  n_pool = len(pool_images)
  test_images = (test_images - train_mu) / train_std  # normalize w train mu/std
  pool_images = (pool_images - train_mu) / train_std  # normalize w train mu/std

  # augmentation for train/pool data
  if FLAGS.augment_data:
    augmentation = data.chain_transforms(
        data.RandomHorizontalFlip(0.5), data.RandomCrop(4), data.ToDevice)
  else:
    augmentation = None
  # END: fetch test data and candidate pool

  # BEGIN: load ckpt
  opt_init, opt_update, get_params = optimizers.sgd(FLAGS.learning_rate)

  if FLAGS.pretrained_dir is not None:
    with gfile.Open(FLAGS.pretrained_dir, 'rb') as fpre:
      pretrained_opt_state = optimizers.pack_optimizer_state(
          pickle.load(fpre))
    fixed_params = get_params(pretrained_opt_state)[:7]

    ckpt_dir = '{}/{}'.format(FLAGS.root_dir, FLAGS.ckpt_idx)
    with gfile.Open(ckpt_dir, 'wr') as fckpt:
      opt_state = optimizers.pack_optimizer_state(
          pickle.load(fckpt))
    params = get_params(opt_state)
    # combine fixed pretrained params and dpsgd trained last layers
    params = fixed_params + params
    opt_state = opt_init(params)
  else:
    ckpt_dir = '{}/{}'.format(FLAGS.root_dir, FLAGS.ckpt_idx)
    with gfile.Open(ckpt_dir, 'wr') as fckpt:
      opt_state = optimizers.pack_optimizer_state(
          pickle.load(fckpt))
    params = get_params(opt_state)

  stdout_log.write('finetune from: {}\n'.format(ckpt_dir))
  logging.info('finetune from: %s', ckpt_dir)
  test_acc, test_pred = accuracy(
      params, shape_as_image(test_images, test_labels),
      return_predicted_class=True)
  logging.info('test accuracy: %.2f', test_acc)
  stdout_log.write('test accuracy: {}\n'.format(test_acc))
  stdout_log.flush()
  # END: load ckpt

  # BEGIN: setup for dp model
  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad_loss(params, batch), opt_state)

  @jit
  def private_update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)  # get new key for new random numbers
    return opt_update(i,
                      private_grad(params, batch, rng, FLAGS.l2_norm_clip,
                                   FLAGS.noise_multiplier, FLAGS.batch_size),
                      opt_state)
  # END: setup for dp model

  n_uncertain = FLAGS.n_extra + FLAGS.uncertain_extra

  ### BEGIN: prepare extra points picked from pool data
  # BEGIN: on pool data
  pool_embeddings = [apply_fn_0(params[:-1],
                                pool_images[b_i:b_i + FLAGS.batch_size]) \
                     for b_i in range(0, n_pool, FLAGS.batch_size)]
  pool_embeddings = np.concatenate(pool_embeddings, axis=0)

  pool_logits = apply_fn_1(params[-1:], pool_embeddings)

  pool_true_labels = np.argmax(pool_labels, axis=1)
  pool_predicted_labels = np.argmax(pool_logits, axis=1)
  pool_correct_indices = \
      onp.where(pool_true_labels == pool_predicted_labels)[0]
  pool_incorrect_indices = \
      onp.where(pool_true_labels != pool_predicted_labels)[0]
  assert len(pool_correct_indices) + \
      len(pool_incorrect_indices) == len(pool_labels)

  pool_probs = stax.softmax(pool_logits)

  if FLAGS.uncertain == 0 or FLAGS.uncertain == 'entropy':
    pool_entropy = -onp.sum(pool_probs * onp.log(pool_probs), axis=1)
    stdout_log.write('all {} entropy: min {}, max {}\n'.format(
        len(pool_entropy), onp.min(pool_entropy), onp.max(pool_entropy)))

    pool_entropy_sorted_indices = onp.argsort(pool_entropy)
    # take the n_uncertain most uncertain points
    pool_uncertain_indices = \
        pool_entropy_sorted_indices[::-1][:n_uncertain]
    stdout_log.write('uncertain {} entropy: min {}, max {}\n'.format(
        len(pool_entropy[pool_uncertain_indices]),
        onp.min(pool_entropy[pool_uncertain_indices]),
        onp.max(pool_entropy[pool_uncertain_indices])))

  elif FLAGS.uncertain == 1 or FLAGS.uncertain == 'difference':
    # 1st_prob - 2nd_prob
    assert len(pool_probs.shape) == 2
    sorted_pool_probs = onp.sort(pool_probs, axis=1)
    pool_probs_diff = sorted_pool_probs[:, -1] - sorted_pool_probs[:, -2]
    assert min(pool_probs_diff) > 0.
    pool_uncertain_indices = onp.argsort(pool_probs_diff)[:n_uncertain]

  # END: on pool data

  # BEGIN: cluster uncertain pool points
  big_pca = sklearn.decomposition.PCA(
      n_components=pool_embeddings.shape[1])
  big_pca.random_state = FLAGS.seed
  # fit PCA onto embeddings of all the pool points
  big_pca.fit(pool_embeddings)

  # For uncertain points, project embeddings onto the first K components
  pool_uncertain_projected_embeddings, _ = utils.project_embeddings(
      pool_embeddings[pool_uncertain_indices],
      big_pca, FLAGS.k_components)

  n_cluster = int(FLAGS.n_extra / FLAGS.ppc)
  cluster_method = get_cluster_method(
      '{}_nc-{}'.format(FLAGS.clustering, n_cluster))
  cluster_method.random_state = FLAGS.seed
  pool_uncertain_cluster_labels = cluster_method.fit_predict(
      pool_uncertain_projected_embeddings)
  pool_uncertain_cluster_label_indices = {x: [] for x in set(
      pool_uncertain_cluster_labels)}  # local i within n_uncertain
  for i, c_label in enumerate(pool_uncertain_cluster_labels):
    pool_uncertain_cluster_label_indices[c_label].append(i)

  # find center of each cluster
  # aka, the most representative point of each 'tough' cluster
  pool_picked_indices = []
  pool_uncertain_cluster_label_pick = {}
  for c_label, indices in pool_uncertain_cluster_label_indices.items():
    cluster_projected_embeddings = \
        pool_uncertain_projected_embeddings[indices]
    cluster_center = onp.mean(cluster_projected_embeddings,
                              axis=0, keepdims=True)
    if FLAGS.distance == 0 or FLAGS.distance == 'euclidean':
      cluster_distances = euclidean_distances(
          cluster_projected_embeddings, cluster_center).reshape(-1)
    elif FLAGS.distance == 1 or FLAGS.distance == 'weighted_euclidean':
      cluster_distances = weighted_euclidean_distances(
          cluster_projected_embeddings, cluster_center,
          big_pca.singular_values_[:FLAGS.k_components])

    sorted_is = onp.argsort(cluster_distances)
    sorted_indices = onp.array(indices)[sorted_is]
    pool_uncertain_cluster_label_indices[c_label] = sorted_indices
    center_i = sorted_indices[0]  # center_i in 3000
    pool_uncertain_cluster_label_pick[c_label] = center_i
    pool_picked_indices.extend(
        pool_uncertain_indices[sorted_indices[:FLAGS.ppc]])

    # BEGIN: visualize cluster of picked pool
    if FLAGS.visualize:
      this_cluster = []
      for i in sorted_indices:
        idx = pool_uncertain_indices[i]
        img = utils.denormalize(pool_images[idx], train_mu, train_std)
        if idx in pool_correct_indices:
          border_color = 'green'
        else:
          border_color = 'red'
          img = utils.mark_labels(img, pool_predicted_labels[idx],
                                  pool_true_labels[idx])
        img = utils.add_border(img, width=2, color=border_color)
        this_cluster.append(img)
      utils.tile_image_list(
          this_cluster,
          '{}/picked_uncertain_pool_cid-{}'.format(FLAGS.work_dir, c_label))
    # END: visualize cluster of picked pool

  # END: cluster uncertain pool points

  pool_picked_indices = list(set(pool_picked_indices))

  n_gap = FLAGS.n_extra - len(pool_picked_indices)
  gap_indices = list(set(pool_uncertain_indices) - set(pool_picked_indices))
  pool_picked_indices.extend(npr.choice(gap_indices, n_gap, replace=False))
  stdout_log.write('n_gap: {}\n'.format(n_gap))
  ### END: prepare extra points picked from pool data

  finetune_images = copy.deepcopy(pool_images[pool_picked_indices])
  finetune_labels = copy.deepcopy(pool_labels[pool_picked_indices])

  stdout_log.write('{} points picked via {}\n'.format(
      len(finetune_images), FLAGS.uncertain))
  logging.info('%d points picked via %s', len(finetune_images), FLAGS.uncertain)
  assert FLAGS.n_extra == len(finetune_images)
  # END: gather points to be used for finetuning

  stdout_log.write('Starting fine-tuning...\n')
  logging.info('Starting fine-tuning...')
  stdout_log.flush()

  for epoch in range(1, FLAGS.epochs + 1):

    # BEGIN: finetune model with extra data, evaluate and save
    num_extra = len(finetune_images)
    num_complete_batches, leftover = divmod(num_extra, FLAGS.batch_size)
    num_batches = num_complete_batches + bool(leftover)

    finetune = data.DataChunk(
        X=finetune_images, Y=finetune_labels,
        image_size=32, image_channels=3, label_dim=1, label_format='numeric')

    batches = data.minibatcher(
        finetune, FLAGS.batch_size, transform=augmentation)

    itercount = itertools.count()
    key = random.PRNGKey(FLAGS.seed)

    start_time = time.time()

    for _ in range(num_batches):
      # tmp_time = time.time()
      b = next(batches)
      if FLAGS.dpsgd:
        opt_state = private_update(
            key, next(itercount), opt_state,
            shape_as_image(b.X, b.Y, dummy_dim=True))
      else:
        opt_state = update(
            key, next(itercount), opt_state, shape_as_image(b.X, b.Y))
      # stdout_log.write('single update in {:.2f} sec\n'.format(
      #     time.time() - tmp_time))

    epoch_time = time.time() - start_time
    stdout_log.write('Epoch {} in {:.2f} sec\n'.format(epoch, epoch_time))
    logging.info('Epoch %d in %.2f sec', epoch, epoch_time)

    # accuracy on test data
    params = get_params(opt_state)

    test_pred_0 = test_pred
    test_acc, test_pred = accuracy(
        params, shape_as_image(test_images, test_labels),
        return_predicted_class=True)
    test_loss = loss(params, shape_as_image(test_images, test_labels))
    stdout_log.write('Eval set loss, accuracy (%): ({:.2f}, {:.2f})\n'.format(
        test_loss, 100 * test_acc))
    logging.info('Eval set loss, accuracy: (%.2f, %.2f)',
                 test_loss, 100 * test_acc)
    stdout_log.flush()

    # visualize prediction difference between 2 checkpoints.
    if FLAGS.visualize:
      utils.visualize_ckpt_difference(
          test_images, np.argmax(test_labels, axis=1),
          test_pred_0, test_pred,
          epoch - 1, epoch, FLAGS.work_dir, mu=train_mu, sigma=train_std)

  # END: finetune model with extra data, evaluate and save

  stdout_log.close()


if __name__ == '__main__':
  app.run(main)
