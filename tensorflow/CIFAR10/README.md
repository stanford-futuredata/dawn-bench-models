# ResNets on TensorFlow

To train a ResNet, run,

```bash
python3 resnet/resnet_main.py --train_data_path=cifar10/data_batch* --log_root=data/resnet20/log_root \
                              --train_dir=data/resnet20/log_root/train --dataset='cifar10' --model=resnet20 \
                              --num_gpus=1 --checkpoint_dir=data/resnet20/checkpoints --data_format=NCHW
```

To evaluate resulting checkpoints, run,

```bash
python3 eval_checkpoints.py -i data/resnet20/checkpoints \
                            -c "python3 resnet/resnet_main.py --mode=eval --eval_data_path=cifar10/test_batch.bin --eval_dir=data/resnet20/log_root/eval --dataset='cifar10' --model=resnet20 --num_gpus=1 --eval_batch_count=100 --eval_once=True --data_format=NCHW"
```

Make sure to first follow the instructions in `resnet/README.md` to get necessary data, etc.
