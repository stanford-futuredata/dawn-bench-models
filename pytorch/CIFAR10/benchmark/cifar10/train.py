import os
from datetime import datetime
from collections import OrderedDict

import click
import torch
import tqdm
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets

from benchmark import utils
from benchmark.yellowfin import YFOptimizer
from benchmark.cifar10.models import resnet, densenet

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
        'preact8': resnet.PreActResNet8,
        'preact14': resnet.PreActResNet14,
        'preact20': resnet.PreActResNet20,
        'preact56': resnet.PreActResNet56,
        'preact164-basic': resnet.PreActResNet164Basic,

        # "Identity Mappings in Deep Residual Networks"
        'preact110': resnet.PreActResNet110,
        'preact164': resnet.PreActResNet164,
        'preact1001': resnet.PreActResNet1001,

        # Based on "Deep Networks with Stochastic Depth"
        'stochastic56': resnet.StochasticResNet56,
        'stochastic56-08': resnet.StochasticResNet56_08,
        'stochastic110': resnet.StochasticResNet110,
        'stochastic1202': resnet.StochasticResNet1202,

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


def correct(outputs, targets, top=(1, )):
    _, predictions = outputs.topk(max(top), dim=1, largest=True, sorted=True)
    targets = targets.view(-1, 1).expand_as(predictions)

    corrects = predictions.eq(targets).cpu().int().cumsum(1).sum(0)
    tops = list(map(lambda k: corrects.data[0][k - 1], top))
    return tops


def run(epoch, model, loader, criterion=None, optimizer=None, top=(1, 5),
        use_cuda=False, tracking=None, train=True, half=False):
    accuracies = [utils.AverageMeter() for _ in top]

    assert criterion is not None or not train, 'Need criterion to train model'
    assert optimizer is not None or not train, 'Need optimizer to train model'
    loader = tqdm.tqdm(loader)
    if train:
        model.train()
        losses = utils.AverageMeter()
    else:
        model.eval()

    start = datetime.now()
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = Variable(inputs, requires_grad=False, volatile=not train)
        targets = Variable(targets, requires_grad=False, volatile=not train)
        batch_size = targets.size(0)
        assert batch_size < 2**32, 'Size is too large! correct will overflow'

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            if half:
                inputs = inputs.half()

        outputs = model(inputs)

        if train:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.data[0], batch_size)

        _, predictions = torch.max(outputs.data, 1)
        top_correct = correct(outputs, targets, top=top)
        for i, count in enumerate(top_correct):
            accuracies[i].update(count * (100. / batch_size), batch_size)

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
                result['top{}_accuracy'.format(k)] = accuracies[i].val
            if train:
                result['loss'] = loss.data[0]
            utils.save_result(result, tracking)

        desc = 'Epoch {} {}'.format(epoch, '(Train):' if train else '(Val):  ')
        if train:
            desc += ' Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)
        for k, acc in zip(top, accuracies):
            desc += ' Prec@{} {acc.val:.3f} ({acc.avg:.3f})'.format(k, acc=acc)
        loader.set_description(desc)
        start = datetime.now()

    if train:
        message = 'Training accuracy of'
    else:
        message = 'Validation accuracy of'
    for i, k in enumerate(top):
        message += ' top-{}: {}'.format(k, accuracies[i].avg)
    print(message)
    return accuracies[0].avg


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
@click.option('--lr-factor', default=1.0, help='only for yellowfin')
@click.option('--momentum', default=0.9)
@click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam', 'yellowfin']),
              default='sgd')
@click.option('--augmentation/--no-augmentation', default=True)
@click.option('device_ids', '--device', '-d', multiple=True, type=int)
@click.option('--num-workers', type=int)
@click.option('--weight-decay', default=5e-4)
@click.option('--validation', '-v', default=0.0)
@click.option('--evaluate', is_flag=True)
@click.option('--shuffle/--no-shuffle', default=True)
@click.option('--half', is_flag=True)
@click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
              default='resnet20')
def train(dataset_dir, checkpoint, restore, tracking, cuda, epochs,
          batch_size, learning_rate, lr_factor, momentum, optimizer, augmentation,
          device_ids, num_workers, weight_decay, validation, evaluate, shuffle,
          half, arch):
    timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
    local_timestamp = str(datetime.now())
    config = {k: v for k, v in locals().items()}

    use_cuda = cuda and torch.cuda.is_available()

    # create model
    model = MODELS[arch]()

    # create optimizer
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer == 'yellowfin':
        optimizer = YFOptimizer(model.parameters(), lr=learning_rate,
                                mu=momentum, weight_decay=weight_decay)

    else:
        raise NotImplementedError("Unknown optimizer: {}".format(optimizer))

    if restore is not None:
        if restore == 'latest':
            restore = utils.latest_file(arch)
        print(f'Restoring model from {restore}')
        assert os.path.exists(restore)
        restored_state = torch.load(restore)
        assert restored_state['arch'] == arch

        model.load_state_dict(restored_state['model'])
        optimizer.load_state_dict(restored_state['optimizer'])
        if not isinstance(optimizer, YFOptimizer):
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

        best_accuracy = restored_state['accuracy']
        start_epoch = restored_state['epoch'] + 1
        run_dir = os.path.split(restore)[0]
    else:
        best_accuracy = 0.0
        start_epoch = 1
        run_dir = f"./run/{arch}/{timestamp}"

    print('Starting accuracy is {}'.format(best_accuracy))

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    utils.save_config(config, run_dir)

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))
    print(f"Run directory set to {run_dir}")

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    if tracking:
        train_results_file = os.path.join(run_dir, 'train_results.csv')
        valid_results_file = os.path.join(run_dir, 'valid_results.csv')
        test_results_file = os.path.join(run_dir, 'test_results.csv')
    else:
        train_results_file = None
        valid_results_file = None
        test_results_file = None

    # create loss
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        print('Copying model to GPU')
        model = model.cuda()
        criterion = criterion.cuda()

        if half:
            model = model.half()
            criterion = criterion.half()
        device_ids = device_ids or list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(
            model, device_ids=device_ids)
        num_workers = num_workers or len(device_ids)
    else:
        num_workers = num_workers or 1
        if half:
            print('Half precision (16-bit floating point) only works on GPU')
    print(f"using {num_workers} workers for data loading")

    # load data
    print("Preparing data:")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=dataset_dir, train=False, download=True,
                         transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_cuda)

    if evaluate:
        print("Only running evaluation of model on test dataset")
        run(start_epoch - 1, model, test_loader, use_cuda=use_cuda,
            tracking=test_results_file, train=False)
        return

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

    train_dataset = datasets.CIFAR10(root=dataset_dir, train=True,
                                     download=True, transform=transform_train)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    assert 1 > validation and validation >= 0, "Validation must be in [0, 1)"
    split = num_train - int(validation * num_train)

    if shuffle:
        np.random.shuffle(indices)

    train_indices = indices[:split]
    valid_indices = indices[split:]

    print('Using {} examples for training'.format(len(train_indices)))
    print('Using {} examples for validation'.format(len(valid_indices)))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_cuda)
    if validation != 0:
        valid_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=valid_sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=use_cuda)
    else:
        print('Using test dataset for validation')
        valid_loader = test_loader

    end_epoch = start_epoch + epochs
    # YellowFin doesn't have param_groups causing AttributeError
    if not isinstance(optimizer, YFOptimizer):
        for group in optimizer.param_groups:
            if 'lr' in group:
                print('Learning rate set to {}'.format(group['lr']))
                assert group['lr'] == learning_rate
    else:
        print(f"set lr_factor to {lr_factor}")
        optimizer.set_lr_factor(lr_factor)
    for epoch in range(start_epoch, end_epoch):
        run(epoch, model, train_loader, criterion, optimizer,
            use_cuda=use_cuda, tracking=train_results_file, train=True,
            half=half)

        valid_acc = run(epoch, model, valid_loader, use_cuda=use_cuda,
                        tracking=valid_results_file, train=False, half=half)

        is_best = valid_acc > best_accuracy
        last_epoch = epoch == (end_epoch - 1)
        if is_best or checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
            state = {
                'epoch': epoch,
                'arch': arch,
                'model': (model.module if use_cuda else model).state_dict(),
                'accuracy': valid_acc,
                'optimizer': optimizer.state_dict()
            }
        if is_best:
            print('New best model!')
            filename = os.path.join(run_dir, 'checkpoint_best_model.t7')
            print(f'Saving checkpoint to {filename}')
            best_accuracy = valid_acc
            torch.save(state, filename)
        if checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
            filename = os.path.join(run_dir, f'checkpoint_{epoch}.t7')
            print(f'Saving checkpoint to {filename}')
            torch.save(state, filename)


if __name__ == '__main__':
    train()
