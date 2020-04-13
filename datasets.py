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

"""Datasets used in examples."""

import array
import codecs
import gzip
import operator
import os
from os import path
import struct
from jax import lax
from jax import pmap
from jax import random
from jax import vmap
import jax.numpy as np
from jax.tree_util import tree_map
from jax.tree_util import tree_reduce
import numpy as onp

from six.moves.urllib.request import urlretrieve
import tensorflow as tf
from tensorflow.compat.v1.io import gfile
import tensorflow_datasets as tfds


_DATA = '/tmp/jax_example_data/'


def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urlretrieve(url, out_file)


def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return onp.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=onp.float32):
  """Create a one-hot encoding of x of size k."""
  return onp.array(x[:, None] == onp.arange(k), dtype)


def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = 'http://yann.lecun.com/exdb/mnist/'

  def parse_labels(filename):
    with gzip.open(filename, 'rb') as fh:
      _ = struct.unpack('>II', fh.read(8))
      return onp.array(array.array('B', fh.read()), dtype=onp.uint8)

  def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
      _, num_data, rows, cols = struct.unpack('>IIII', fh.read(16))
      return onp.array(array.array('B', fh.read()),
                       dtype=onp.uint8).reshape(num_data, rows, cols)

  for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, 'train-images-idx3-ubyte.gz'))
  train_labels = parse_labels(path.join(_DATA, 'train-labels-idx1-ubyte.gz'))
  test_images = parse_images(path.join(_DATA, 't10k-images-idx3-ubyte.gz'))
  test_labels = parse_labels(path.join(_DATA, 't10k-labels-idx1-ubyte.gz'))

  return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / onp.float32(255.)
  train_images = train_images * onp.float32(2.0) - onp.float32(1.0)  # [-1., 1.]
  test_images = _partial_flatten(test_images) / onp.float32(255.)
  test_images = test_images * onp.float32(2.0) - onp.float32(1.0)  # [-1.0, 1.0]
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = onp.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def fashion_mnist_load_data():
  """Loads the Fashion-MNIST dataset.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  file_path = '/file/oi-d/home/person/datasets/fashion_mnist'
  with gfile.Open('/gzip{}/train-labels-idx1-ubyte.gz'.format(file_path),
                  'rb') as lbpath:
    y_train = onp.frombuffer(lbpath.read(), onp.uint8, offset=8)

  with gfile.Open('/gzip{}/train-images-idx3-ubyte.gz'.format(file_path),
                  'rb') as imgpath:
    x_train = onp.frombuffer(
        imgpath.read(), onp.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gfile.Open('/gzip{}/t10k-labels-idx1-ubyte.gz'.format(file_path),
                  'rb') as lbpath:
    y_test = onp.frombuffer(lbpath.read(), onp.uint8, offset=8)

  with gfile.Open('/gzip{}/t10k-images-idx3-ubyte.gz'.format(file_path),
                  'rb') as imgpath:
    x_test = onp.frombuffer(
        imgpath.read(), onp.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)


def fashionmnist(permute_train=False):
  """Fashion MNIST."""
  (train_images, train_labels), (test_images, test_labels) = \
      fashion_mnist_load_data()

  train_images = _partial_flatten(train_images) / onp.float32(255.)
  train_images = train_images * onp.float32(2.0) - onp.float32(1.0)  # [-1., 1.]
  test_images = _partial_flatten(test_images) / onp.float32(255.)
  test_images = test_images * onp.float32(2.0) - onp.float32(1.0)  # [-1.0, 1.0]
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = onp.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def get_qmnist_split(split, shuffle=False):
  """https://github.com/facebookresearch/qmnist ."""
  file_path = '/file/oi-d/home/person/datasets/qmnist'
  if split == 'train':
    with gfile.Open('/gzip{}/qmnist-train-images-idx3-ubyte.gz'.format(
        file_path), 'rb') as f:
      images = read_idx3_ubyte(f)

    with gfile.Open('/gzip{}/qmnist-train-labels-idx2-int.gz'.format(
        file_path), 'rb') as f:
      labels = read_idx2_int(f)
      labels = labels[:, 0]

  elif split == 'test':
    with gfile.Open('/gzip{}/qmnist-test-images-idx3-ubyte.gz'.format(
        file_path), 'rb') as f:
      images = read_idx3_ubyte(f)

    with gfile.Open('/gzip{}/qmnist-test-labels-idx2-int.gz'.format(
        file_path), 'rb') as f:
      labels = read_idx2_int(f)
      labels = labels[:, 0]

  elif split == 'test50k':
    with gfile.Open('/gzip{}/qmnist-test-images-idx3-ubyte.gz'.format(
        file_path), 'rb') as f:
      images = read_idx3_ubyte(f)

    with gfile.Open('/gzip{}/qmnist-test-labels-idx2-int.gz'.format(
        file_path), 'rb') as f:
      labels = read_idx2_int(f)
      labels = labels[:, 0]

    images, labels = images[-50000:], labels[-50000:]

  if shuffle:
    perm = onp.random.RandomState(0).permutation(images.shape[0])
    images = images[perm]
    labels = labels[perm]

  label_names = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9']
  return images, labels, label_names


def get_dataset_split(name, split, shuffle=False):
  """Get specific dataset split as numpy array.

  Args:
    name: dataset string {qmnist, mnist, fashion_mnist, cifar10, svhn_cropped}
    split: string {train, validation, test}
    shuffle: bool

  Returns:
    images, labels, label_names
  """
  if name == 'qmnist':
    images, labels, label_names = get_qmnist_split(split, shuffle)
  else:
    if split == 'train_train' or split == 'train_pool':
      ds, info = tfds.load(name=name, split='train',
                           download=True, with_info=True)
      num_points = int(info.splits['train'].num_examples)
    else:
      ds, info = tfds.load(name=name, split=split,
                           download=True, with_info=True)
      num_points = int(info.splits[split].num_examples)
    ds = ds.cache().batch(num_points).repeat()
    if shuffle:
      ds = ds.shuffle(buffer_size=num_points)
    iterator = ds.make_one_shot_iterator()
    features = iterator.get_next()
    with tf.Session().as_default():
      images = features['image'].eval().astype('float32')
      labels = features['label'].eval().astype('uint8')
    label_names = info.features['label'].names

  # images = images / onp.float32(127.5) - onp.float32(1.0)  # [-1.0, 1.0]
  labels = _one_hot(labels, 10)

  if split == 'train_train' or split == 'train_pool':
    train_pool_indices = []
    with gfile.Open(
        '/file/oi-d/home/person/datasets/{}.train_pool'.format(name),
        'r') as f:
      for line in f:
        train_pool_indices.append(int(line.strip()))
    assert len(train_pool_indices) == 10000
    train_train_indices = sorted(list(
        set(range(len(images))) - set(train_pool_indices)))

  if split == 'train_train':
    images, labels = images[train_train_indices], labels[train_train_indices]
  elif split == 'train_pool':
    images, labels = images[train_pool_indices], labels[train_pool_indices]

  return images, labels, label_names


# BEGIN code in https://github.com/facebookresearch/qmnist/blob/master/qmnist.py
def get_int(b):
  return int(codecs.encode(b, 'hex'), 16)


def read_idx2_int(f):
  data = f.read()
  assert get_int(data[:4]) == 12*256 + 2
  length = get_int(data[4:8])
  width = get_int(data[8:12])
  parsed = onp.frombuffer(data, dtype=onp.dtype('>i4'), offset=12)
  return parsed.astype('uint8').reshape(length, width)


def read_idx3_ubyte(f):
  data = f.read()
  assert get_int(data[:4]) == 8 * 256 + 3
  length = get_int(data[4:8])
  num_rows = get_int(data[8:12])
  num_cols = get_int(data[12:16])
  parsed = onp.frombuffer(data, dtype=onp.uint8, offset=16)
  return parsed.reshape(length, num_rows, num_cols, 1)
# END qmnist code


# BEGIN: data augmentation.
def crop(key, image_and_label):
  """Random flips and crops."""
  image, label = image_and_label
  pixels = 4
  pixpad = (pixels, pixels)
  zero = (0, 0)
  padded_image = np.pad(image, (pixpad, pixpad, zero), 'constant', 0.0)
  corner = random.randint(key, (2,), 0, 2 * pixels)
  corner = np.concatenate((corner, np.zeros((1,), np.int32)))
  img_size = (32, 32, 3)
  cropped_image = lax.dynamic_slice(padded_image, corner, img_size)
  return cropped_image, label
crop = vmap(crop, 0, 0)


def mixup(key, alpha, image_and_label):
  """https://arxiv.org/abs/1710.09412 mixup."""
  image, label = image_and_label
  batch_size = image.shape[0]

  weight = random.beta(key, alpha, alpha, (batch_size, 1))
  mixed_label = weight * label + (1.0 - weight) * label[::-1]

  weight = np.reshape(weight, (batch_size, 1, 1, 1))
  mixed_image = weight * image + (1.0 - weight) * image[::-1]

  return mixed_image, mixed_label


def augment_color32(key, image_and_label):
  """Augment SVHN or CIFAR10."""
  image, label = image_and_label
  key, split = random.split(key)

  batch_size = image.shape[0]
  image = np.reshape(image, (batch_size, 32, 32, 3))

  image = np.where(
      random.uniform(split, (batch_size, 1, 1, 1)) < 0.5,
      image[:, :, ::-1],  # invert color channels
      image)

  key, split = random.split(key)
  batch_split = random.split(split, batch_size)
  image, label = crop(batch_split, (image, label))

  # return mixup(key, 1.0, (image, label))
  return image, label
# END: data augmentation.


# BEGIN: shard data pipeline.
def shard_data(n_devices, data):
  """Shard data."""
  _, ragged = divmod(data[0].shape[0], n_devices)

  if ragged:
    assert NotImplementedError('Cannot split data evenly across devies.')

  data = [np.reshape(x, (n_devices, -1) + x.shape[1:]) for x in data]
  data = map(pmap(lambda x: x), data)

  return data


def sharded_minibatcher(batch_size, n_devices, transform=None):
  """Shard mini batches."""
  batch_size_per_device, ragged = divmod(batch_size, n_devices)

  if ragged:
    raise NotImplementedError('Cannot divide batch evenly across devices.')

  def shuffle(key_and_data):
    key, data = key_and_data
    key, subkey = random.split(key)
    datapoints_per_device = data[0].shape[0]
    indices = np.arange(datapoints_per_device)
    perm = random.shuffle(subkey, indices)
    return key, [x[perm] for x in data], 0

  def init_fn(key, data):
    datapoints_per_device = data[0].shape[0]

    key, data, i = shuffle((key, data))

    num_batches = datapoints_per_device // batch_size_per_device

    return (key, data, i, num_batches)

  def batch_fn(state):
    """Function for batching."""
    key, data, i, num_batches = state

    slice_start = (
        [i * batch_size_per_device, 0, 0],
        [i * batch_size_per_device, 0]
    )

    slice_size = (
        [batch_size_per_device, 32, 32 * 3],
        [batch_size_per_device, 10]
    )

    batch = [
        lax.dynamic_slice(x, start, size) for x, start, size in
        zip(data, slice_start, slice_size)
    ]

    if transform is not None:
      key, subkey = random.split(key)
      batch = transform(subkey, batch)

    i = i + 1
    key, data, i = lax.cond(
        i >= num_batches,
        (key, data), shuffle,
        (key, data, i), lambda x: x)

    return batch, (key, data, i, num_batches)

  return init_fn, batch_fn
# END: shard data pipeline.


# Loss Definition.
_l2_norm = lambda params: tree_map(lambda x: np.sum(x ** 2), params)
l2_regularization = lambda params: tree_reduce(operator.add, _l2_norm(params))

cross_entropy = lambda y, y_hat: -np.mean(np.sum(y * y_hat, axis=1))


# Learning rate schedule of Cosine.
def cosine_schedule(initial_lr, training_steps):
  return lambda t: initial_lr * 0.5 * (1 + np.cos(t / training_steps * np.pi))
