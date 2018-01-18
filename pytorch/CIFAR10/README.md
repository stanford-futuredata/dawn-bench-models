# Install

1. Install PyTorch v0.1.12. If you don't already have it set up, [please follow the official install instructions](http://pytorch.org/).
2. Clone this repo and go to this directory

```bash
git clone git@github.com:stanford-futuredata/dawn-bench-models.git
cd dawn-bench-models/pytorch/CIFAR10
```

3. Install this package

```bash
pip install -e .
```

# Quick start

This package adds <code>cifar10</code> and <code>imagenet</code> command line interfaces.
Both include the <code>train</code> subcommands to learn a model from scratch.
As an example, here is how to train ResNet164 with preactivation on CIFAR10:

```bash
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 5 -l 0.01
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 90 -l 0.1 --restore latest
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 45 -l 0.01 --restore latest
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 45 -l 0.001 --restore latest
```

The first command creates a new run of ResNet164 with preactivation (`--arch preact164`) in the `./run/preact164/[TIMESTAMP]` directory and starts a warm up of 5 epochs (`-e 5`) with SGD (`--optimizer sgd`) and a learning rate of 0.01 (`-l 0.01`).
`-c last` indicates that we only want to save a checkpoint after the last epoch of the warm up.
`-b 128` sets the batch size to 128.
`--augmentation` turns on standard data augmentation, i.e. random crop and flip.
`--tracking` saves training and validation results to csv files at `./run/preact164/[TIMESTAMP]/[train|valid]_results.csv`

The second command resumes the run from the first command (`--restore latest`) for another 90 epochs (`-e 90`) but with a new learning rate (`-l 0.1`). The third and fourth commands function similarly to the second command, changing the learning rate and running for more epochs.
