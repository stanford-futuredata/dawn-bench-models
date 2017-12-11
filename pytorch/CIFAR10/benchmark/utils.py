import os
import json
import re
from functools import reduce


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    c = map(lambda p: reduce(lambda x, y: x * y, p.size()), model.parameters())
    return sum(c)


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


def save_result(result, path):
    write_heading = not os.path.exists(path)
    with open(path, mode='a') as out:
        if write_heading:
            out.write(",".join([str(k) for k, v in result.items()]) + '\n')
        out.write(",".join([str(v) for k, v in result.items()]) + '\n')


def save_config(config, run_dir):
    path = os.path.join(run_dir, "config_{}.json".format(config['timestamp']))
    with open(path, 'w') as config_file:
        json.dump(config, config_file)
        config_file.write('\n')
