# ResNet on CIFAR10 and CIFAR100

(Borrowed from the tensorflow/models repository)

## Dataset

https://www.cs.toronto.edu/~kriz/cifar.html

## Related papers

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027v2.pdf)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)
- [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v1.pdf)

## Setting

* Pad to 36x36 and random crop. Horizontal flip. Per-image whitening.
* Momentum optimizer (momentum = 0.9).
* Learning rate schedule: 0.01 (1 epoch), 0.1 (90 epochs), 0.01 (45 epochs), 0.001 (45 epochs).
* L2 weight decay: 0.005.
* Batch size: 128. (28-10 wide and 1001 layer bottleneck use 64)

## Results

CIFAR-10 Model|Best Precision|Steps
--------------|--------------|------
32 layer|92.5%|~80k
110 layer|93.6%|~80k
164 layer bottleneck|94.5%|~80k
1001 layer bottleneck|94.9%|~80k
28-10 wide|95%|~90k

CIFAR-100 Model|Best Precision|Steps
---------------|--------------|-----
32 layer|68.1%|~45k
110 layer|71.3%|~60k
164 layer bottleneck|75.7%|~50k
1001 layer bottleneck|78.2%|~70k
28-10 wide|78.3%|~70k

## Prerequisites

1. Install TensorFlow 1.2 (preferably from source for higher performance) and Python 3.6.2.

2. Download CIFAR-10/CIFAR-100 dataset.

```shell
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

## How to run

```shell
# cd to the models repository and run with bash. Expected command output shown.
# The directory should contain an empty WORKSPACE file, the resnet code, and the cifar10 dataset.
# Note: The user can split 5k from train set for eval set.
$ ls -R
.:
cifar10  resnet  WORKSPACE

./cifar10:
data_batch_1.bin  data_batch_2.bin  data_batch_3.bin  data_batch_4.bin
data_batch_5.bin  test_batch.bin

./resnet:
cifar_input.py  README.md  resnet_main.py  resnet_model.py

# Train the model.
$ python3 resnet/resnet_main.py --train_data_path=cifar10/data_batch* \
                                --log_root=/tmp/resnet_model \
                                --train_dir=/tmp/resnet_model/train \
                                --dataset='cifar10' \
                                --num_gpus=1

# While the model is training, you can also check on its progress using tensorboard:
$ tensorboard --logdir=/tmp/resnet_model

# Evaluate the model.
# Avoid running on the same GPU as the training job at the same time,
# otherwise, you might run out of memory.
$ python3 resnet/resnet_main.py --eval_data_path=cifar10/test_batch.bin \
                                --log_root=/tmp/resnet_model \
                                --eval_dir=/tmp/resnet_model/test \
                                --mode=eval \
                                --dataset='cifar10' \
                                --num_gpus=0
```
