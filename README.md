# Description

This directory contains reference code for anonymous submission.

Improving Differentially Private Models with Active Learning


## MNIST

```
# train baseline classifier via sgd/dpsgd from scratch
python -m train_mnist.py --dpsgd=True

# improve dpsgd baseline via sgd on extra public qmnist data
# baseline methods: random and uncertain
python -m improve_baseline_mnist.py \
--dpsgd=False --pool_split=qmnist-test50k --exp_id=? --ckpt_idx=?

# improve dpsgd baseline via sgd on extra public qmnist data
# PCA and clustering on public extra data ONLY
python -m improve_only_public_mnist.py \
--dpsgd=False --pool_split=qmnist-test50k --exp_id=? --ckpt_idx=?

# improve dpsgd baseline via sgd on extra public qmnist data
# dpPCA on private data and assign/link each to ONLY 1 public point/cluster
python -m improve_near_private_mnist.py \
--dpsgd=False --pool_split=qmnist-test50k --exp_id=? --ckpt_idx=?

```

## SVHN

### train classifier via dpsgd from scratch

```
# train baseline classifier via sgd/dpsgd from scratch on 1 GPU
python -m train_svhn.py
```

### pretrain classifier via sgd on rotations, then continue training via dpsgd

```
# pretrain model via sgd on unlabeled public extra data
# self-supervised by 4 random rotations
python -m pretrain_rotation_svhn.py

# continue training last dense layers w conv layers fixed
# via dpsgd on labeled private train data
python -m train_dpsgd_svhn.py \
--dpsgd=True --pretrained_dir=?
```

### improve dpsgd baseline via sgd on extra public data

```
# baseline methods including random and uncertain sampling
python -m improve_baseline_svhn.py \
--dpsgd=False --exp_id=? --pretrained_dir=? --ckpt_idx=?

# PCA and clustering on public extra data ONLY
python -m improve_only_public_svhn.py \
--dpsgd=False --exp_id=? --pretrained_dir=? --ckpt_idx=?

# dpPCA on private data and assign/link each to ONLY 1 public point/cluster
python -m improve_near_private_svhn.py \
--dpsgd=False --exp_id=? --pretrained_dir=? --ckpt_idx=?
```
