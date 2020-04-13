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

"""Differentially Private PCA."""

import numpy as onp
import sklearn.preprocessing


def sanitize(x, epsilon, delta, sigma=None,
             l2norm_bound=1., num_examples=None):
  """Sanitize the given x by adding Gaussian noise."""
  if sigma is None:
    assert epsilon > 0.
    assert delta > 0.
    sigma = onp.sqrt(2.0 * onp.log(1.25 / delta)) / epsilon

  if num_examples is None:
    num_examples = len(x)

  noise = onp.random.normal(scale=sigma * l2norm_bound, size=x.shape)
  saned_x = x + noise

  return saned_x


def dp_pca(data, projection_dims, epsilon, delta, sigma):
  """Differentially Private PCA."""
  # Normalize each row.
  normalized_data = sklearn.preprocessing.normalize(data, norm='l2', axis=1)
  covar = onp.matmul(onp.transpose(normalized_data), normalized_data)
  saved_shape, num_examples = covar.shape, covar.shape[0]

  if epsilon > 0:
    # No need to clip covar since data already normalized.
    assert delta > 0.
    saned_covar = sanitize(covar.reshape([1, -1]),
                           epsilon, delta, sigma,
                           num_examples=num_examples)
    saned_covar = onp.reshape(saned_covar, saved_shape)
    # Symmetrize saned_covar. This also reduces the noise variance.
    saned_covar = 0.5 * (saned_covar + onp.transpose(saned_covar))
  else:
    saned_covar = covar

  # Compute the eigen decomposition of the covariance matrix.
  eigvals, eigvecs = onp.linalg.eig(saned_covar)

  # Return top projection_dims eigenvectors, as cols of projection matrix.
  topk_indices = onp.argsort(eigvals)[::-1][:projection_dims]

  # Gather and return the corresponding eigenvectors.
  pc_cols = onp.transpose(onp.transpose(eigvecs)[topk_indices])
  return pc_cols, eigvals[topk_indices]
