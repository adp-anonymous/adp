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

"""Train CNN on MNIST/Fashion-MNIST w SGD/DPSGD."""

import logging
import operator
import pickle
import time

from absl import app
from absl import flags
from jax import grad
from jax import jit
from jax import partial
from jax import random
from jax import tree_util
from jax import vmap
import jax.experimental.optimizers as optimizers
import jax.experimental.stax as stax
from jax.lax import stop_gradient
import jax.numpy as np
from tensorflow.compat.v1.io import gfile
# https://github.com/tensorflow/privacy
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from adp import data
from adp import datasets


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'config', '', 'Training configuration.')
flags.DEFINE_boolean('dpsgd', False,
                     'True, train with DP-SGD. False, train with vanilla SGD.')
flags.DEFINE_string('dataset', 'mnist', 'Dataset, or fashion_mnist')
flags.DEFINE_string('exp_dir', None, 'Experiment directory')


def cnn():
  return stax.serial(
      stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
      stax.Tanh,
      stax.MaxPool((2, 2), (1, 1)),
      stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
      stax.Tanh,
      stax.MaxPool((2, 2), (1, 1)),
      stax.Flatten,
      stax.Dense(32),
      stax.Tanh,  # embeddings
      stax.Dense(10),  # logits
      )


def compute_epsilon(steps, batch_size, num_examples=50000, target_delta=1e-5,
                    noise_multiplier=1.1):
  if num_examples * target_delta > 1.:
    logging.warning('Your target_delta might be too high.')
  q = batch_size / float(num_examples)
  orders = list(np.linspace(1.1, 10.9, 99)) + range(11, 64)
  rdp_const = compute_rdp(q, noise_multiplier, steps, orders)
  eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
  return eps


def main(_):
  logging.info('Starting experiment.')
  configs = FLAGS.config

  # Create model folder for outputs
  try:
    gfile.MakeDirs(FLAGS.exp_dir)
  except gfile.GOSError:
    pass
  stdout_log = gfile.Open('{}/stdout.log'.format(FLAGS.exp_dir), 'w+')

  logging.info('Loading data.')
  tic = time.time()

  train_images, train_labels, _ = datasets.get_dataset_split(
      FLAGS.dataset, 'train')
  # train_mu, train_std = 128., 128.
  # normalize to range [-1.0, 127./128]
  train_images = train_images / np.float32(128.0) - np.float32(1.0)
  n_train = len(train_images)
  train = data.DataChunk(
      X=train_images, Y=train_labels,
      image_size=28, image_channels=1, label_dim=1, label_format='numeric')

  test_images, test_labels, _ = datasets.get_dataset_split(
      FLAGS.dataset, 'test')
  test_images = test_images / np.float32(128.0) - np.float32(1.0)
  test = data.DataChunk(
      X=test_images, Y=test_labels,
      image_size=28, image_channels=1, label_dim=1, label_format='numeric')

  # Data augmentation
  if configs.augment_data:
    augmentation = data.chain_transforms(
        data.RandomHorizontalFlip(0.5), data.RandomCrop(4), data.ToDevice)
  else:
    augmentation = None
  batch = data.minibatcher(train, configs.batch_size, transform=augmentation)

  # Model architecture
  if configs.architect == 'cnn':
    init_random_params, predict = cnn()
  else:
    raise ValueError('Model architecture not implemented.')

  if configs.seed is not None:
    key = random.PRNGKey(configs.seed)
  else:
    key = random.PRNGKey(int(time.time()))
  _, params = init_random_params(key, (-1, 28, 28, 1))

  # count params of JAX model
  def count_parameters(params):
    return tree_util.tree_reduce(
        operator.add, tree_util.tree_map(lambda x: np.prod(x.shape), params))
  logging.info('Number of parameters: %d', count_parameters(params))
  stdout_log.write('Number of params: {}\n'.format(count_parameters(params)))

  # loss functions
  def cross_entropy_loss(params, x_img, y_lbl):
    return -np.mean(stax.logsoftmax(predict(params, x_img)) * y_lbl)

  def mse_loss(params, x_img, y_lbl):
    return 0.5 * np.mean((y_lbl - predict(params, x_img)) ** 2)

  def accuracy(y_lbl_hat, y_lbl):
    target_class = np.argmax(y_lbl, axis=1)
    predicted_class = np.argmax(y_lbl_hat, axis=1)
    return np.mean(predicted_class == target_class)

  # Loss and gradient
  if configs.loss == 'xent':
    loss = cross_entropy_loss
  elif configs.loss == 'mse':
    loss = mse_loss
  else:
    raise ValueError('Loss function not implemented.')
  grad_loss = jit(grad(loss))

  # learning rate schedule and optimizer
  def cosine(initial_step_size, train_steps):
    k = np.pi / (2.0 * train_steps)
    def schedule(i):
      return initial_step_size * np.cos(k * i)
    return schedule

  if configs.optimization == 'sgd':
    lr_schedule = optimizers.make_schedule(configs.learning_rate)
    opt_init, opt_update, get_params = optimizers.sgd(lr_schedule)
  elif configs.optimization == 'momentum':
    lr_schedule = cosine(configs.learning_rate, configs.train_steps)
    opt_init, opt_update, get_params = optimizers.momentum(lr_schedule, 0.9)
  else:
    raise ValueError('Optimizer not implemented.')

  opt_state = opt_init(params)

  def private_grad(params, batch,
                   rng, l2_norm_clip, noise_multiplier, batch_size):
    """Return differentially private gradients of params, evaluated on batch."""

    def _clipped_grad(params, single_example_batch):
      """Evaluate gradient for a single-example batch and clip its grad norm."""
      grads = grad_loss(params,
                        single_example_batch[0].reshape((-1, 28, 28, 1)),
                        single_example_batch[1])

      nonempty_grads, tree_def = tree_util.tree_flatten(grads)
      total_grad_norm = np.linalg.norm(
          [np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
      divisor = stop_gradient(np.amax((total_grad_norm / l2_norm_clip, 1.)))
      normalized_nonempty_grads = [neg / divisor for neg in nonempty_grads]
      return tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

    px_clipped_grad_fn = vmap(partial(_clipped_grad, params))
    std_dev = l2_norm_clip * noise_multiplier
    noise_ = lambda n: n + std_dev * random.normal(rng, n.shape)
    normalize_ = lambda n: n / float(batch_size)
    sum_ = lambda n: np.sum(n, 0)  # aggregate
    aggregated_clipped_grads = tree_util.tree_map(
        sum_, px_clipped_grad_fn(batch))
    noised_aggregated_clipped_grads = tree_util.tree_map(
        noise_, aggregated_clipped_grads)
    normalized_noised_aggregated_clipped_grads = (
        tree_util.tree_map(normalize_, noised_aggregated_clipped_grads)
    )
    return normalized_noised_aggregated_clipped_grads

  # summarize measurements
  steps_per_epoch = n_train // configs.batch_size

  def summarize(step, params):
    """Compute measurements in a zipped way."""
    set_entries = [train, test]
    set_bsizes = [configs.train_eval_bsize, configs.test_eval_bsize]
    set_names, loss_dict, acc_dict = ['train', 'test'], {}, {}

    for set_entry, set_bsize, set_name in zip(
        set_entries, set_bsizes, set_names):
      temp_loss, temp_acc, points = 0.0, 0.0, 0
      for b in data.batch(set_entry, set_bsize):
        temp_loss += loss(params, b.X, b.Y) * b.X.shape[0]
        temp_acc += accuracy(predict(params, b.X), b.Y) * b.X.shape[0]
        points += b.X.shape[0]
      loss_dict[set_name] = temp_loss / float(points)
      acc_dict[set_name] = temp_acc / float(points)

    logging.info('Step: %s', str(step))
    logging.info('Train acc : %.4f', acc_dict['train'])
    logging.info('Train loss: %.4f', loss_dict['train'])
    logging.info('Test acc  : %.4f', acc_dict['test'])
    logging.info('Test loss : %.4f', loss_dict['test'])

    stdout_log.write('Step: {}\n'.format(step))
    stdout_log.write('Train acc : {}\n'.format(acc_dict['train']))
    stdout_log.write('Train loss: {}\n'.format(loss_dict['train']))
    stdout_log.write('Test acc  : {}\n'.format(acc_dict['test']))
    stdout_log.write('Test loss : {}\n'.format(loss_dict['test']))

    return acc_dict['test']

  toc = time.time()
  logging.info('Elapsed SETUP time: %s', str(toc - tic))
  stdout_log.write('Elapsed SETUP time: {}\n'.format(toc - tic))

  # BEGIN: training steps
  logging.info('Training network.')
  tic = time.time()
  t = time.time()

  for s in range(configs.train_steps):
    b = next(batch)
    params = get_params(opt_state)

    # t0 = time.time()
    if FLAGS.dpsgd:
      key = random.fold_in(key, s)  # get new key for new random numbers
      opt_state = opt_update(
          s,
          private_grad(params, (b.X.reshape((-1, 1, 28, 28, 1)), b.Y),
                       key, configs.l2_norm_clip, configs.noise_multiplier,
                       configs.batch_size),
          opt_state)
    else:
      opt_state = opt_update(s, grad_loss(params, b.X, b.Y), opt_state)
    # t1 = time.time()
    # logging.info('batch update time: %s', str(t1 - t0))

    if s % steps_per_epoch == 0:
      with gfile.Open('{}/ckpt_{}'.format(
          FLAGS.exp_dir, int(s/steps_per_epoch)), 'wb') as fckpt:
        pickle.dump(optimizers.unpack_optimizer_state(opt_state), fckpt)

      if FLAGS.dpsgd:
        eps = compute_epsilon(s, configs.batch_size, n_train,
                              configs.target_delta, configs.noise_multiplier)
        stdout_log.write(
            'For delta={:.0e}, current epsilon is: {:.2f}\n'.format(
                configs.target_delta, eps))

      logging.info('Elapsed EPOCH time: %s', str(time.time() - t))
      stdout_log.write('Elapsed EPOCH time: {}'.format(time.time() - t))
      stdout_log.flush()
      t = time.time()

  toc = time.time()
  summarize(configs.train_steps, params)
  logging.info('Elapsed TRAIN time: %s', str(toc - tic))
  stdout_log.write('Elapsed TRAIN time: {}'.format(toc - tic))
  stdout_log.close()
  # END: training steps


if __name__ == '__main__':
  app.run(main)
