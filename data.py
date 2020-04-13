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

"""Data utilities."""

import collections

from jax.api import device_get
from jax.api import device_put
import jax.numpy as np
import jax.random as random
import numpy as onp
import numpy.random as npr
import tensorflow_datasets as tfds


DataChunk = collections.namedtuple(
    'DataChunk',
    ['X', 'Y', 'image_size', 'image_channels', 'label_dim', 'label_format'])


def Mnist():
  train_data = tfds.load('mnist', split=tfds.Split.TRAIN, batch_size=-1)
  train = tfds.as_numpy(train_data)
  train = (train['image'], train['label'])
  test_data = tfds.load('mnist', split=tfds.Split.TEST, batch_size=-1)
  test = tfds.as_numpy(test_data)
  test = (test['image'], test['label'])

  def get(data):
    return DataChunk(
        X=data[0], Y=data[1],
        image_size=28, image_channels=1, label_dim=1, label_format='numeric')
  return (get(train), get(test))


def FashionMnist():
  train_data = tfds.load('fashion_mnist', split=tfds.Split.TRAIN, batch_size=-1)
  train = tfds.as_numpy(train_data)
  train = (train['image'], train['label'])
  test_data = tfds.load('fashion_mnist', split=tfds.Split.TEST, batch_size=-1)
  test = tfds.as_numpy(test_data)
  test = (test['image'], test['label'])

  def get(data):
    return DataChunk(
        X=data[0], Y=data[1],
        image_size=28, image_channels=1, label_dim=1, label_format='numeric')
  return (get(train), get(test))


def Cifar10():
  train_data = tfds.load('cifar10', split=tfds.Split.TRAIN, batch_size=-1)
  train = tfds.as_numpy(train_data)
  train = (train['image'], train['label'])
  test_data = tfds.load('cifar10', split=tfds.Split.TEST, batch_size=-1)
  test = tfds.as_numpy(test_data)
  test = (test['image'], test['label'])

  def get(data):
    return DataChunk(
        X=data[0], Y=data[1],
        image_size=32, image_channels=3, label_dim=1, label_format='numeric')
  return (get(train), get(test))


def Testing(datapoints=10, size=5, channels=1):
  """Generate testing data regressing to f(x) = |x|^2."""
  data_xs = npr.randn(datapoints, size, size, channels)
  data_xs = np.reshape(data_xs, (datapoints, -1))
  data_ys = np.sum(data_xs ** 2, axis=1, keepdims=True)
  data = (data_xs, data_ys)
  def get(data):
    return DataChunk(
        X=data[0], Y=data[1],
        image_size=size, image_channels=channels, label_dim=1,
        label_format='numeric')
  return (get(data),)


def OneHotLabels():
  def transform(c, rng=None):
    assert c.label_format == 'numeric', (
        'No conversion from label format {} to One-Hot.')

    return DataChunk(
        c.X, np.eye(10)[c.Y],
        c.image_size, c.image_channels, c.label_dim, 'one_hot')
  return transform
OneHotLabels = OneHotLabels()


def SelectClasses(classes):
  def transform(c, rng=None):
    if c.label_format != 'numeric':
      raise NotImplementedError

    return DataChunk(
        c.X[onp.in1d(c.Y, classes)], c.Y[onp.in1d(c.Y, classes)],
        c.image_size, c.image_channels, c.label_dim, 'numeric')
  return transform


def BinarizeLabels(low=-1.0, high=1.0, threshold=5):
  def transform(c, rng=None):
    if c.label_format != 'numeric':
      raise NotImplementedError

    Y = onp.array(c.Y)
    Y[Y < threshold] = low
    Y[Y >= threshold] = high
    Y = np.reshape(Y, (-1, 1))
    return DataChunk(
        c.X, Y,
        c.image_size, c.image_channels, c.label_dim, 'binary')
  return transform


def Standardize(mu=0.0, sigma=1.0):
  def transform(c, rng=None):
    X = c.X
    axes = tuple(range(1, len(X.shape)))
    mean = np.mean(X, axis=axes, keepdims=True)
    std_dev = np.std(X, axis=axes, keepdims=True)
    X = sigma * (X - mean) / std_dev + mu
    return DataChunk(
        X, c.Y,
        c.image_size, c.image_channels, c.label_dim, c.label_format)
  return transform


def Flatten():
  def transform(c, rng=None):
    size = image_flat_size(c)
    return DataChunk(
        np.reshape(c.X, [-1, size]), c.Y,
        size, 0, c.label_dim, c.label_format)
  return transform
Flatten = Flatten()


def ToFloat64():
  def transform(c, rng=None):
    return DataChunk(
        np.array(c.X, np.float64), np.array(c.Y, np.float64),
        c.image_size, c.image_channels, c.label_dim, c.label_format)
  return transform
ToFloat64 = ToFloat64()


def ToFloat32():
  def transform(c, rng=None):
    return DataChunk(
        np.array(c.X, np.float32), np.array(c.Y, np.float32),
        c.image_size, c.image_channels, c.label_dim, c.label_format)
  return transform
ToFloat32 = ToFloat32()


def ToDevice():
  def transform(c, rng=None):
    return DataChunk(
        device_put(c.X), device_put(c.Y),
        c.image_size, c.image_channels, c.label_dim, c.label_format)
  return transform
ToDevice = ToDevice()


def FromDevice():
  def transform(c, rng=None):
    return DataChunk(
        device_get(c.X, np.float64), device_get(c.Y, np.float64),
        c.image_size, c.image_channels, c.label_dim, c.label_format)
  return transform
FromDevice = FromDevice()


def Subset(n):
  def transform(c, rng=None):
    assert n < c.X.shape[0] and n < c.Y.shape[0]
    return DataChunk(
        c.X[:n], c.Y[:n],
        c.image_size, c.image_channels, c.label_dim, c.label_format)
  return transform


def RandomHorizontalFlip(p):
  def transform(c, rng=None):
    if rng is None:
      msg = 'Random data transformations require a PRNG key.'
      raise ValueError(msg)
    flip = random.uniform(rng, shape=(len(c.X), 1, 1, 1))
    flippedX = c.X[:, :, ::-1, :]
    X = np.where(flip < p, flippedX, c.X)
    return DataChunk(
        X, c.Y, c.image_size, c.image_channels, c.label_dim, c.label_format)
  return transform


def RandomCrop(pixels):
  def transform(c, rng=None):
    if rng is None:
      msg = 'Random data transformations require a PRNG key.'
      raise ValueError(msg)
    zero = (0, 0)
    pixpad = (pixels, pixels)
    paddedX = onp.pad(c.X, (zero, pixpad, pixpad, zero), 'reflect')
    corner = random.randint(rng, (c.X.shape[0], 2), 0, 2 * pixels)
    img_size = c.image_size
    slices = [
        (slice(int(o[0]), int(o[0]) + img_size),
         slice(int(o[1]), int(o[1]) + img_size),
         slice(None)) for x, o in zip(paddedX, corner)]
    paddedX = np.concatenate(
        [x[np.newaxis, s[0], s[1], s[2]] for x, s in zip(paddedX, slices)])
    return DataChunk(
        paddedX, c.Y, c.image_size, c.image_channels,
        c.label_dim, c.label_format)
  return transform


def chain_transforms(*transforms):
  ntransforms = len(transforms)
  def transform(c, rng=None):
    rngs = random.split(rng, ntransforms) if rng is not None \
        else (None,) * ntransforms
    for T, rng in zip(transforms, rngs):
      c = T(c, rng)
    return c
  return transform


def image_flat_size(c):
  return c.image_size ** 2 * c.image_channels


def batch(c, batch_size):
  for i in range(0, len(c.X), batch_size):
    yield DataChunk(
        c.X[i:i + batch_size], c.Y[i:i + batch_size],
        c.image_size, c.image_channels, c.label_dim, c.label_format)


def get_train_mask(dataset):
  train_mask = np.arange(dataset.total_size) < dataset.train_size
  train_mask = np.array(train_mask, np.float32)
  train_mask = np.reshape(train_mask, (1, -1))

  return train_mask


def minibatcher(data, batch_size, transform=None, seed=0):
  key = random.PRNGKey(seed)
  size = data.X.shape[0]
  indices = np.arange(size, dtype=np.int32)
  num_batches = size // batch_size

  while True:
    key, subkey = random.split(key)
    perm = random.shuffle(key, indices)
    for i in range(num_batches):
      batch_ids = perm[i * batch_size : (i + 1) * batch_size]
      b = data._replace(X=data.X[batch_ids], Y=data.Y[batch_ids])
      if transform:
        key, subkey = random.split(key)
        b = transform(b, subkey)
      yield b
