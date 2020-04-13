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

"""Shared functions."""

import copy
import numpy as np
import scipy.misc
import six
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.compat.v1.io import gfile




def to_rgb(img):
  """Converts an image to RGB from greyscale (no-op if already rgb)."""
  bsz, height, width = img.shape[:3]
  new_img = np.zeros((bsz, height, width, 3), dtype='uint8')
  new_img[:, :, :, 0] = img[:, :, :, 0]
  new_img[:, :, :, 1] = img[:, :, :, 0]
  new_img[:, :, :, 2] = img[:, :, :, 0]
  return new_img


def add_border(image, width=1, color='red'):
  """Adds border of given width and color to image.

  Args:
    image: a uint8 RGB image with values in [0,255]
    width: the integer width of the border
    color: the color of the border, one of ['red', 'green', 'blue']

  Returns:
      A new numpy array representing an image with the added border.
  """
  w = width
  image = np.pad(image, [(w, w), (w, w), (0, 0)], mode='constant')
  color_to_channel = {'red': 0, 'green': 1, 'blue': 2}
  channel = color_to_channel[color]
  image[:w, :, channel] = 255
  image[-w:, :, channel] = 255
  image[:, :w, channel] = 255
  image[:, -w:, channel] = 255
  return image


def make_images_viewable(images):
  dataset_shape = [images.shape[0], 28, 28, 1]
  images = np.reshape(images, dataset_shape)
  images = ((images*127.999)+128)
  images = np.clip(images, a_min=0, a_max=255)
  images = images.astype('uint8')
  return images


def denormalize(images, mu=0., sigma=1.):
  original_images = images * sigma + mu
  original_images = np.clip(original_images, a_min=0., a_max=255.)
  original_images = np.rint(original_images).astype('uint8')
  return original_images


def tile_image_list(image_list, filename, n_rows=None):
  """Tiles images into a grid and saves them."""
  n_samples = len(image_list)
  if not n_rows:  # Allow for forcing of particular n_rows value.
    n_rows = int(np.sqrt(n_samples))
  # while n_samples % n_rows != 0:
  #   n_rows -= 1
  n_cols = n_samples//n_rows
  if n_cols * n_rows < n_samples:
    n_cols += 1

  # Copy each image into its spot in the grid
  height, width = image_list[0].shape[:2]
  grid_image = np.zeros((height*n_rows, width*n_cols, 3), dtype='uint8')
  for n, image in enumerate(image_list):
    j = n // n_cols
    i = n % n_cols
    grid_image[j*height:j*height+height, i*width:i*width+width] = image
  imsave(grid_image, filename + '.png')
  return grid_image


def mark_labels(image_, predicted_label, true_label):
  """Mark predicted at top and true at bottom."""
  image = copy.deepcopy(image_)
  if image.shape == (28, 28, 1):  # mnist or fashion-mnist
    # black region at top+bottom vs left+right
    if np.sum(image[0, :, 0] + image[27, :, 0]) \
      <= np.sum(image[:, 0, 0] + image[:, 27, 0]):
      # more black region at top+bottom
      for i in range(10):
        if i != predicted_label:
          image[0, 3 * i, 0] = 1.
        if i != true_label:
          image[27, 3 * i, 0] = 1.
    else:
      # more black region at left+right
      for i in range(10):
        if i != predicted_label:
          image[3 * i, 0, 0] = 1.
        if i != true_label:
          image[3 * i, 27, 0] = 1.

  elif image.shape == (32, 32, 3):  # svhn or cifar10
    # black region at top+bottom vs left+right
    if np.sum(image[0, :, :] + image[31, :, :]) \
      <= np.sum(image[:, 0, :] + image[:, 31, :]):
      # more black region at top+bottom
      for i in range(10):
        if i != predicted_label:
          image[0, 3 * i + 2, :] = 255
        if i != true_label:
          image[31, 3 * i + 2, :] = 255
    else:
      # more black region at left+right
      for i in range(10):
        if i != predicted_label:
          image[3 * i + 2, 0, :] = 255
        if i != true_label:
          image[3 * i + 2, 31, :] = 255

  return image


def project_embeddings(embeddings, pca_object, n_components, pc_cols=None):
  """Project embs.

  Gives the dimensionality reduction to n_components and
  also the rank-n_components projection of those features, given
  embeddings in shape (n_samples, n_features) and pca_object
  that has already had fit called on it.
  if the svd_solver is exact, this should give the same results as doing
  pca.transform(x) and pca.inverse_transform(pca.transform(x)).
  """
  if pc_cols is None:
    pc_rows = pca_object.components_[:n_components, :]
  else:
    pc_rows = pc_cols.T[:n_components, :]

  embeddings_mean = np.mean(embeddings, axis=0)
  centered_embeddings = embeddings - embeddings_mean
  projected_embeddings = np.dot(centered_embeddings, pc_rows.T)
  inverted_projections = (np.dot(projected_embeddings, pc_rows)
                          + embeddings_mean)
  return projected_embeddings, inverted_projections


def cluster_pcs(big_pca, embeddings, cluster_method, eval_images_numpy,
                fpath, checkpoint_idx, k_components=16,
                m_demos=8, incorrect_indices=None, visualize_clusters=True):
  """Cluster projected embeddings onto pcs.

  Visualize examples in each cluster, sorted by distance from cluster center.
  If border in red and green, red means predicted incorrectly.
  """
  if incorrect_indices:
    def color_from_idx(index):
      return 'red' if index in incorrect_indices else 'green'
  else:
    def color_from_idx(_):
      return 'red'

  # Create folder for outputs
  try:
    gfile.MakeDirs(fpath)
  except gfile.GOSError:
    pass

  # Project embeddings onto the first K components
  projected_embeddings, _ = project_embeddings(
      embeddings, big_pca, k_components)

  # Cluster projected embeddings
  cluster_labels = cluster_method.fit_predict(projected_embeddings)

  cluster_label_indices = {x: [] for x in set(cluster_labels)}
  for idx, c_label in enumerate(cluster_labels):
    cluster_label_indices[c_label].append(idx)
  n_clusters = len(cluster_label_indices.keys())

  # adjust m_demos wrt the smallest cluster
  for k in cluster_label_indices.keys():
    m_demos = min(m_demos, int(len(cluster_label_indices[k]) / 2))
    print('cluster {}, count {}, incorrect {}'.format(
        k, len(cluster_label_indices[k]),
        len(incorrect_indices & set(cluster_label_indices[k]))))

  fname = '{}/ckpt-{}_npc-{}_nc-{}_nd-{}'.format(
      fpath, checkpoint_idx, k_components, n_clusters, m_demos)

  # Prepare each row/cluster gradually
  big_list = []

  for cid, c_label in enumerate(cluster_label_indices.keys()):

    original_indices = cluster_label_indices[c_label]

    cluster_projected_embeddings = projected_embeddings[original_indices]

    cluster_center = np.mean(cluster_projected_embeddings,
                             axis=0, keepdims=True)

    cluster_distances = list(euclidean_distances(cluster_projected_embeddings,
                                                 cluster_center).reshape(-1))

    cluster_indices_distances = zip(original_indices, cluster_distances)
    # sort as from the outside to the center of cluster
    cluster_indices_distances = sorted(cluster_indices_distances,
                                       key=lambda x: x[1], reverse=True)

    sorted_indices = list(zip(*cluster_indices_distances)[0])

    min_indices = sorted_indices[:m_demos]      # outer of cluster
    max_indices = sorted_indices[-m_demos:]     # inner of cluster

    min_imgs = eval_images_numpy[min_indices]
    max_imgs = eval_images_numpy[max_indices]
    this_row = ([min_imgs[m, :] for m in range(m_demos)] +
                [max_imgs[m, :] for m in range(m_demos)])
    this_idx_row = min_indices + max_indices

    this_row = [make_images_viewable(np.expand_dims(x, 0))
                for x in this_row]
    this_row = [to_rgb(x) for x in this_row]
    this_row = [add_border(np.squeeze(x), width=2, color=color_from_idx(index))
                for index, x in zip(this_idx_row, this_row)]
    big_list += this_row

    if visualize_clusters:

      this_cluster = [make_images_viewable(np.expand_dims(x, 0))
                      for x in eval_images_numpy[sorted_indices]]
      this_cluster = [to_rgb(x) for x in this_cluster]
      this_cluster = [add_border(np.squeeze(x), width=2,
                                 color=color_from_idx(index))
                      for index, x in zip(sorted_indices, this_cluster)]

      tile_image_list(this_cluster,
                      fname + '_cid-{}'.format(cid))

  grid_image = tile_image_list(big_list, fname, n_rows=n_clusters)
  return grid_image


def visualize_ckpt_difference(images, labels, predicted_0, predicted_1,
                              idx_0, idx_1, save_dir, mu=0., sigma=1.):
  """Visualize prediction difference between 2 checkpoints.

  red border: changed from correct to incorrect
  green border: changed from inccorrect to correct
  blue border: both incorrect

  Args:
    images: normalized images [-1, 1]
    labels: ground truth labels, shape (n,)
    predicted_0: prediction of ckpt_0, shape (n,)
    predicted_1: prediction of ckpt_1, shape (n,)
    idx_0: index of ckpt_0
    idx_1: index of ckpt_1
    save_dir: folder path to save the output image
    mu: mean of train data
    sigma: std of train data
  """
  correct_indices_0 = set(np.where(labels == predicted_0)[0])
  incorrect_indices_0 = set(np.where(labels != predicted_0)[0])
  assert len(correct_indices_0) + len(incorrect_indices_0) == labels.shape[0]

  correct_indices_1 = set(np.where(labels == predicted_1)[0])
  incorrect_indices_1 = set(np.where(labels != predicted_1)[0])
  assert len(correct_indices_1) + len(incorrect_indices_1) == labels.shape[0]

  red_indices = list(correct_indices_0 & incorrect_indices_1)
  green_indices = list(incorrect_indices_0 & correct_indices_1)
  blue_indices = list(incorrect_indices_0 & incorrect_indices_1)

  def prepare_image_tile(indices, color):
    """prepare images of the same border color."""
    tiles = []
    for idx in indices:

      if images.shape[-1] == 1:
        tile = mark_labels(
            images[idx],
            predicted_0[idx] if color == 'green' else predicted_1[idx],
            labels[idx])  # mark out labels
        tile = np.squeeze(to_rgb(
            denormalize(np.expand_dims(tile, 0), mu, sigma)))
      elif images.shape[-1] == 3:
        tile = denormalize(images[idx], mu, sigma)
        tile = mark_labels(
            tile,
            predicted_0[idx] if color == 'green' else predicted_1[idx],
            labels[idx])  # mark out labels

      tile = add_border(tile, width=2, color=color)
      tiles.append(tile)
    return tiles

  this_cluster = []
  for indices, color in [(red_indices, 'red'), (green_indices, 'green'),
                         (blue_indices, 'blue')]:
    this_cluster.extend(prepare_image_tile(indices, color))

  tile_image_list(this_cluster,
                  '{}/prediction_ckpt-{}_vs_ckpt-{}'.format(
                      save_dir, idx_0, idx_1))


def imsave(image, filename):
  image_to_file = six.StringIO()
  scipy.misc.imsave(image_to_file, image, format='png')
  file_inst = gfile.GFile(filename, mode='w+')
  file_inst.write(image_to_file.getvalue())
  file_inst.close()
  # scipy.misc.imsave(filename + '.png', image, format='png')
