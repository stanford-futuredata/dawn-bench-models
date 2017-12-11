import os
import time
from datetime import datetime
from collections import OrderedDict

import click
import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from benchmark import utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


@click.command()
@click.option('--dataset-dir', default='./data/imagenet')
@click.option('--checkpoint', '-c', type=click.Choice(['best', 'all', 'last']),
              default='last')
@click.option('--restore', '-r')
@click.option('--tracking/--no-tracking', default=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--epochs', '-e', default=90)
@click.option('--batch-size', '-b', default=256)
@click.option('--learning-rate', '-l', default=0.1)
@click.option('--learning-rate-decay', default=0.1)
@click.option('--learning-rate-freq', default=30)
@click.option('--momentum', default=0.9)
@click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']),
              default='sgd')
@click.option('--augmentation/--no-augmentation', default=True)
@click.option('--pretrained', is_flag=True)
@click.option('--evaluate', is_flag=True)
@click.option('--num-workers', type=int)
@click.option('--weight-decay', default=1e-4)
@click.option('--arch', '-a', type=click.Choice(model_names),
              default='resnet18')
def train(dataset_dir, checkpoint, restore, tracking, cuda, epochs,
          batch_size, learning_rate, learning_rate_decay,
          learning_rate_freq, momentum, optimizer, augmentation,
          pretrained, evaluate, num_workers, weight_decay, arch):
    timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
    config = {k: v for k, v in locals().items()}

    use_cuda = cuda and torch.cuda.is_available()

    # create model
    if pretrained:
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(optimizer))

    # optionally resume from a checkpoint
    if restore is not None:
        if restore == 'latest':
            restore = utils.latest_file(arch)
        print(f'=> restoring model from {restore}')
        restored_state = torch.load(restore)
        start_epoch = restored_state['epoch'] + 1
        best_prec1 = restored_state['prec1']
        model.load_state_dict(restored_state['state_dict'])
        optimizer.load_state_dict(restored_state['optimizer'])
        print('=> starting accuracy is {} (epoch {})'
              .format(best_prec1, start_epoch))
        run_dir = os.path.split(restore)[0]
    else:
        best_prec1 = 0.0
        start_epoch = 1
        run_dir = f"./run/{arch}/{timestamp}"

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    utils.save_config(config, run_dir)

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))
    print(f"Run directory set to {run_dir}")

    # save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    if tracking:
        train_results_file = os.path.join(run_dir, 'train_results.csv')
        test_results_file = os.path.join(run_dir, 'test_results.csv')
    else:
        train_results_file = None
        test_results_file = None

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # move model and criterion to GPU
    if use_cuda:
        model.cuda()
        criterion = criterion.cuda()
        model = torch.nn.parallel.DataParallel(model)
        num_workers = num_workers or torch.cuda.device_count()
    else:
        num_workers = num_workers or 1
    print(f"=> using {num_workers} workers for data loading")

    cudnn.benchmark = True

    # Data loading code
    print("=> preparing data:")
    traindir = os.path.join(dataset_dir, 'train')
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    if evaluate:
        validate(val_loader, model, criterion)
        return

    end_epoch = start_epoch + epochs
    for epoch in range(start_epoch, end_epoch):
        print('Epoch {} of {}'.format(epoch, end_epoch - 1))
        adjust_learning_rate(optimizer, epoch, learning_rate,
                             decay=learning_rate_decay,
                             freq=learning_rate_freq)

        # train for one epoch
        _ = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch,
            tracking=train_results_file)

        # evaluate on validation set
        prec1, _ = validate(
            val_loader, model, criterion, epoch, tracking=test_results_file)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        last_epoch = epoch == (end_epoch - 1)
        if is_best or checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
            state = {
                'epoch': epoch,
                'arch': arch,
                'state_dict': (model.module if use_cuda else model).state_dict(),
                'prec1': prec1,
                'optimizer': optimizer.state_dict(),
            }
            if is_best:
                print('New best model!')
                filename = os.path.join(run_dir, 'checkpoint_best_model.t7')
                print(f'=> saving checkpoint to {filename}')
                torch.save(state, filename)
                best_prec1 = prec1
            if checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
                filename = os.path.join(run_dir, f'checkpoint_{epoch}.t7')
                print(f'=> saving checkpoint to {filename}')
                torch.save(state, filename)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch,
                    tracking=None):
    train_loader = tqdm.tqdm(train_loader)
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        if tracking is not None:
            result = OrderedDict()
            result['timestamp'] = datetime.now()
            result['batch_duration'] = batch_time.val
            result['epoch'] = epoch
            result['batch'] = i
            result['batch_size'] = input.size(0)
            result['top1_accuracy'] = prec1[0]
            result['top5_accuracy'] = prec5[0]
            result['loss'] = loss.data[0]
            result['data_duration'] = data_time.val
            utils.save_result(result, tracking)

        desc = ('Epoch {0} (Train):'
                ' Loss {loss.val:.4f} ({loss.avg:.4f})'
                ' Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                ' Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
        train_loader.set_description(desc)

        end = time.time()

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch, tracking=None):
    val_loader = tqdm.tqdm(val_loader)
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        if tracking is not None:
            result = OrderedDict()
            result['timestamp'] = datetime.now()
            result['batch_duration'] = batch_time.val
            result['epoch'] = epoch
            result['batch'] = i
            result['batch_size'] = input.size(0)
            result['top1_accuracy'] = prec1[0]
            result['top5_accuracy'] = prec5[0]
            result['loss'] = loss.data[0]
            utils.save_result(result, tracking)

        desc = ('Epoch {0} (Val):  '
                ' Loss {loss.val:.4f} ({loss.avg:.4f})'
                ' Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                ' Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))
        val_loader.set_description(desc)
        end = time.time()

    print("Evaluation: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}"
          .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


def adjust_learning_rate(optimizer, epoch, initial_learning_rate, decay, freq):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_learning_rate * (decay ** ((epoch - 1) // freq))
    print(f'=> learning rate is set to {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    train()
