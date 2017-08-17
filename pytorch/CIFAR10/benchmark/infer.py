import os
import timeit
from glob import glob
from collections import OrderedDict

import click
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets

from benchmark.train import load, MEAN, STD, save_result, MODELS


class PyTorchEngine:
    def __init__(self, filename, use_cuda=False, name=None):
        self.filename = filename
        self.use_cuda = use_cuda
        self.name = name
        model, epoch, accuracy = load(self.filename)

        if self.use_cuda:
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.epoch = epoch
        self.accuracy = accuracy

    def pred(self, inputs):
        inputs = Variable(inputs, requires_grad=False, volatile=True)

        if self.use_cuda:
            inputs = inputs.cuda()
            return self.model(inputs).data.cpu().numpy()
        else:
            return self.model(inputs).data.numpy()


def time_batch_size(dataset, batch_size, pred, use_cuda, repeat=100, bestof=3):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, pin_memory=use_cuda)
    inputs, targets = loader.__iter__().next()
    assert inputs.size(0) == batch_size

    times = timeit.repeat('pred(inputs)', globals=locals(),
                          repeat=repeat, number=1)

    return times


def infer_cifar10(dataset, engine, start=1, end=128, repeat=100, log2=True,
                  output=None):
    if log2:
        start = int(np.floor(np.log2(start)))
        end = int(np.ceil(np.log2(end)))
        assert start >= 0
        assert end >= start
        batch_sizes = map(lambda x: 2**x, range(start, end + 1))
    else:
        batch_sizes = range(start, end + 1)
    results = []
    for batch_size in batch_sizes:
        times = time_batch_size(dataset, batch_size, engine.pred,
                                engine.use_cuda, repeat=repeat)

        result = OrderedDict()
        result['nodename'] = os.uname().nodename
        result['model'] = engine.name
        result['use_cuda'] = engine.use_cuda
        result['batch_size'] = batch_size
        result['mean'] = np.mean(times)
        result['std'] = np.std(times)
        result['throughput'] = batch_size / np.mean(times)
        result['filename'] = engine.filename
        if output is not None:
            save_result(result, output)

        print('batch_size: {batch_size:4d}'
              ' - mean: {mean:.4f}'
              ' - std: {std:.4f}'
              ' - throughput: {throughput:.4f}'.format(**result))
        results.append(result)

    return results


@click.command()
@click.option('--dataset-dir', default='./data/cifar10')
@click.option('--run-dir', default='./run/')
@click.option('--output-file', default='inference.csv')
@click.option('--start', '-s', default=1)
@click.option('--end', '-e', default=128)
@click.option('--repeat', '-r', default=100)
@click.option('--log2/--no-log2', default=True)
@click.option('--cpu/--no-cpu', default=True)
@click.option('--gpu/--no-gpu', default=True)
@click.option('--append', is_flag=True)
@click.option('--models', '-m', type=click.Choice(MODELS.keys()),
              multiple=True)
def infer(dataset_dir, run_dir, output_file, start, end, repeat, log2,
          cpu, gpu, append, models):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    testset = datasets.CIFAR10(root=dataset_dir, train=False, download=True,
                               transform=transform_test)
    models = models or os.listdir(run_dir)
    output_path = os.path.join(run_dir, output_file)
    assert not os.path.exists(output_path) or append
    for model in models:
        model_dir = os.path.join(run_dir, model)
        paths = glob(f"{model_dir}/*/checkpoint_best_model.t7")
        assert len(paths) > 0
        path = os.path.abspath(paths[0])

        print(f'Model: {model}')
        print(f'Path: {path}')

        if cpu:
            print('With CPU:')
            engine = PyTorchEngine(path, use_cuda=False, name=model)
            infer_cifar10(testset, engine, start=start, end=end, log2=log2,
                          repeat=repeat, output=output_path)

        if gpu and torch.cuda.is_available():
            print('With GPU:')
            engine = PyTorchEngine(path, use_cuda=True, name=model)
            # Warmup
            time_batch_size(testset, 1, engine.pred, engine.use_cuda, repeat=1)

            infer_cifar10(testset, engine, start=start, end=end, log2=log2,
                          repeat=repeat, output=output_path)


if __name__ == '__main__':
    infer()
