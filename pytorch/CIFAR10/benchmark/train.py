import os
import re
import json
from functools import reduce
from datetime import datetime
from collections import OrderedDict

import click
import torch
import progressbar
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets as dset

from benchmark.models import resnet, densenet

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

MODELS = {
        # "Deep Residual Learning for Image Recognition"
        'resnet20': resnet.ResNet20,
        'resnet32': resnet.ResNet32,
        'resnet44': resnet.ResNet44,
        'resnet56': resnet.ResNet56,
        'resnet110': resnet.ResNet110,
        'resnet1202': resnet.ResNet1202,

        # "Wide Residual Networks"
        'wrn-40-4': resnet.WRN_40_4,
        'wrn-16-8': resnet.WRN_16_8,
        'wrn-28-10': resnet.WRN_28_10,

        # Based on "Identity Mappings in Deep Residual Networks"
        'preact20': resnet.PreActResNet20,
        'preact56': resnet.PreActResNet56,
        'preact164-basic': resnet.PreActResNet164Basic,

        # "Identity Mappings in Deep Residual Networks"
        'preact110': resnet.PreActResNet110,
        'preact164': resnet.PreActResNet164,
        'preact1001': resnet.PreActResNet1001,

        # "Aggregated Residual Transformations for Deep Neural Networks"
        'resnext29-8-64': lambda _=None: resnet.ResNeXt29(8, 64),
        'resnext29-16-64': lambda _=None: resnet.ResNeXt29(16, 64),

        # "Densely Connected Convolutional Networks"
        'densenetbc100': densenet.DenseNetBC100,
        'densenetbc250': densenet.DenseNetBC250,
        'densenetbc190': densenet.DenseNetBC190,

        # Kuangliu/pytorch-cifar
        'resnet18': resnet.ResNet18,
        'resnet50': resnet.ResNet50,
        'resnet101': resnet.ResNet101,
        'resnet152': resnet.ResNet152,
}


def count_parameters(model):
    c = map(lambda p: reduce(lambda x, y: x * y, p.size()), model.parameters())
    return sum(c)


def correct(outputs, targets, top=(1, )):
    _, predictions = outputs.topk(max(top), dim=1, largest=True, sorted=True)
    targets = targets.view(-1, 1).expand_as(predictions)
    corrects = predictions.eq(targets).cpu().cumsum(1).sum(0)
    tops = list(map(lambda k: corrects.data[0][k - 1], top))
    return tops


def save_result(result, path):
    write_heading = not os.path.exists(path)
    with open(path, mode='a') as out:
        if write_heading:
            out.write(",".join([str(k) for k, v in result.items()]) + '\n')
        out.write(",".join([str(v) for k, v in result.items()]) + '\n')


def run(epoch, model, loader, criterion=None, optimizer=None, top=(1, 5),
        use_cuda=False, tracking=None, max_value=None, train=True):

    assert criterion is not None or not train, 'Need criterion to train model'
    assert optimizer is not None or not train, 'Need optimizer to train model'
    max_value = max_value or progressbar.UnknownLength
    bar = progressbar.ProgressBar(max_value=max_value)
    total = 0
    correct_counts = {}
    if train:
        model.train()
    else:
        model.eval()

    start = datetime.now()
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = Variable(inputs, requires_grad=False, volatile=not train)
        targets = Variable(targets, requires_grad=False, volatile=not train)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        if train:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, predictions = torch.max(outputs.data, 1)
        batch_size = targets.size(0)
        top_correct = correct(outputs, targets, top=top)
        total += batch_size
        for k, count in zip(top, top_correct):
            correct_counts[k] = correct_counts.get(k, 0) + count

        end = datetime.now()
        if tracking is not None:
            result = OrderedDict()
            result['timestamp'] = datetime.now()
            result['batch_duration'] = end - start
            result['epoch'] = epoch
            result['batch'] = batch_index
            result['batch_size'] = batch_size
            for i, k in enumerate(top):
                result['top{}_correct'.format(k)] = top_correct[i]
            if train:
                result['loss'] = loss.data[0]
            save_result(result, tracking)

        bar.update(batch_index + 1)
        start = datetime.now()

    print()
    if train:
        message = 'Training accuracy of'
    else:
        message = 'Test accuracy of'
    for k in top:
        accuracy = correct_counts[k] / total
        message += ' top-{}: {}'.format(k, accuracy)
    print(message)
    return (1. * correct_counts[top[0]]) / total, batch_index + 1


def save(model, directory, epoch, accuracy, use_cuda=False, filename=None):
    state = {
        'model': model.module if use_cuda else model,
        'epoch': epoch,
        'accuracy': accuracy
    }

    filename = filename or 'checkpoint_{}.t7'.format(epoch)
    torch.save(state, os.path.join(directory, filename))


def save_config(config, run_dir):
    path = os.path.join(run_dir, "config_{}.json".format(config['timestamp']))
    with open(path, 'w') as config_file:
        json.dump(config, config_file)
        config_file.write('\n')


def load(path):
    assert os.path.exists(path)
    state = torch.load(path)
    model = state['model']
    epoch = state['epoch']
    accuracy = state['accuracy']
    return model, epoch, accuracy


def latest_file(model):
    restore = f'./run/{model}'
    timestamps = sorted(os.listdir(restore))
    assert len(timestamps) > 0
    run_dir = os.path.join(restore, timestamps[-1])
    files = os.listdir(run_dir)
    max_checkpoint = -1
    for filename in files:
        if re.search('checkpoint_\d+.t7', filename):
            num = int(re.search('\d+', filename).group())

            if num > max_checkpoint:
                max_checkpoint = num
                max_checkpoint_file = filename

    assert max_checkpoint != -1
    return os.path.join(run_dir, max_checkpoint_file)


@click.command()
@click.option('--dataset-dir', default='./data/cifar10')
@click.option('--checkpoint', '-c', type=click.Choice(['best', 'all', 'last']),
              default='last')
@click.option('--restore', '-r')
@click.option('--tracking/--no-tracking', default=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--epochs', '-e', default=200)
@click.option('--batch-size', '-b', default=32)
@click.option('--learning-rate', '-l', default=1e-3)
@click.option('--sgd', 'optimizer', flag_value='sgd')
@click.option('--adam', 'optimizer', flag_value='adam', default=True)
@click.option('--augmentation/--no-augmentation', default=True)
@click.option('--num-workers', type=int)
@click.option('--weight-decay', default=5e-4)
@click.option('--model', '-m', type=click.Choice(MODELS.keys()),
              default='resnet20')
def main(dataset_dir, checkpoint, restore, tracking, cuda, epochs,
         batch_size, learning_rate, optimizer, augmentation, num_workers,
         weight_decay, model):
    timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
    config = {k: v for k, v in locals().items()}

    use_cuda = cuda and torch.cuda.is_available()
    if use_cuda:
        num_workers = num_workers or torch.cuda.device_count()
    else:
        num_workers = num_workers or 1

    print(f"using {num_workers} workers for data loading")

    print("Preparing data:")

    if augmentation:
        transform_train = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
        ]
    else:
        transform_train = []

    transform_train = transforms.Compose(transform_train + [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    trainset = dset.CIFAR10(root=dataset_dir, train=True, download=True,
                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=use_cuda)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    testset = dset.CIFAR10(root=dataset_dir, train=False, download=True,
                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_cuda)

    if restore is not None:
        if restore == 'latest':
            restore = latest_file(model)
        print(f'Restoring model from {restore}')
        model, start_epoch, best_accuracy = load(restore)
        start_epoch += 1
        print('Starting accuracy is {}'.format(best_accuracy))
        run_dir = os.path.split(restore)[0]
    else:
        print(f'Building {model} model')
        best_accuracy = -1
        start_epoch = 1
        run_dir = f"./run/{model}/{timestamp}"
        model = MODELS[model]()

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    save_config(config, run_dir)

    print(model)
    print("{} parameters".format(count_parameters(model)))
    print(f"Run directory set to {run_dir}")

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    if tracking:
        train_results_file = os.path.join(run_dir, 'train_results.csv')
        test_results_file = os.path.join(run_dir, 'test_results.csv')
    else:
        train_results_file = None
        test_results_file = None

    if use_cuda:
        print('Copying model to GPU')
        model.cuda()
        model = torch.nn.DataParallel(
            model, device_ids=range(torch.cuda.device_count()))
    criterion = nn.CrossEntropyLoss()

    # Other parameters?
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9,
                              weight_decay=weight_decay)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(optimizer))

    train_max_value = None
    test_max_value = None
    end_epoch = start_epoch + epochs
    for epoch in range(start_epoch, end_epoch):
        print('Epoch {} of {}'.format(epoch, end_epoch - 1))
        train_acc, train_max_value = run(epoch, model, train_loader, criterion,
                                         optimizer, use_cuda=use_cuda,
                                         tracking=train_results_file,
                                         max_value=train_max_value, train=True)

        test_acc, test_max_value = run(epoch, model, test_loader,
                                       use_cuda=use_cuda,
                                       tracking=test_results_file, train=False)

        if test_acc > best_accuracy:
            print('New best model!')
            save(model, run_dir, epoch, test_acc, use_cuda=use_cuda,
                 filename='checkpoint_best_model.t7')
            best_accuracy = test_acc

        last_epoch = epoch == (end_epoch - 1)
        if checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
            save(model, run_dir, epoch, test_acc, use_cuda=use_cuda)


if __name__ == '__main__':
    main()
