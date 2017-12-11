import os
import json

import pandas as pd

from benchmark.cifar10.train import MODELS
from benchmark.utils import count_parameters


MODEL_SIZES = {key: count_parameters(MODELS[key]()) for key in MODELS.keys()}


def single_run_acc(df):
    df = df.copy()
    df['duration'] = (df['timestamp'] - df['prev_timestamp']).apply(lambda x: x.total_seconds())
    df['batch_duration'] = df['batch_duration'].apply(lambda x: x.total_seconds())

    tmp = df.loc[:, ['epoch', 'batch_size', 'ncorrect', 'duration', 'batch_duration']].groupby('epoch').sum()
    tmp['accuracy'] = tmp['ncorrect'] / tmp['batch_size']
    tmp['throughput'] = tmp['batch_size'] / tmp['duration']
    tmp['_throughput'] = tmp['batch_size'] / tmp['batch_duration']
    tmp['elapsed'] = df.groupby('epoch')['elapsed'].agg('max')
    tmp.reset_index(inplace=True)

    return tmp


def load_file(file, start_timestamp=None):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['batch_duration'] = pd.to_timedelta(df['batch_duration'])
    df['ncorrect'] = df['top1_correct']
    start_timestamp = start_timestamp or df['timestamp'].iloc[0]
    df['elapsed'] = df['timestamp'] - start_timestamp
    df['batch_accuracy'] = df['ncorrect'] / df['batch_size']
    return df


def load_data(directory, verbose=True):
    train_file = os.path.join(directory, 'train_results.csv')
    train = load_file(train_file)
    start_timestamp = train['timestamp'].iloc[0]

    if verbose:
        print(train_file)
        print("Training results shape: {}".format(train.shape))

    try:
        test_file = os.path.join(directory, 'test_results.csv')
        test = load_file(test_file, start_timestamp=start_timestamp)
    except FileNotFoundError:
        test_file = os.path.join(directory, 'valid_results.csv')
        test = load_file(test_file, start_timestamp=start_timestamp)

    if verbose:
        print(test_file)
        print('Test results shape: {}'.format(test.shape))

    train['mode'] = 'train'
    test['mode'] = 'test'

    combined = pd.concat([train, test], ignore_index=True).sort_values(by=['timestamp'])
    combined['prev_timestamp'] = combined['timestamp'].shift(1)
    combined.loc[0, 'prev_timestamp'] = combined.loc[0, 'timestamp'] - combined.loc[0, 'batch_duration']
    train = combined[combined['mode'] == 'train'].copy()
    test = combined[combined['mode'] == 'test'].copy()

    return single_run_acc(train), single_run_acc(test)


def load_multiple(directory, timestamps=None, verbose=False):
    timestamps = timestamps or os.listdir(directory)
    train_sets = []
    test_sets = []
    for timestamp in sorted(timestamps):
        _dir = os.path.join(directory, timestamp)
        train, test = load_data(_dir, verbose=verbose)
        if verbose:
            print()
        train['run'] = _dir
        test['run'] = _dir
        train['job_start'] = timestamp
        test['job_start'] = timestamp
        train_sets.append(train)
        test_sets.append(test)

    return pd.concat(train_sets), pd.concat(test_sets)


def load_multiple_models(directory, verbose=False):
    paths = os.listdir(directory)
    models = [path for path in paths if path in MODELS]

    train_sets = []
    test_sets = []
    for model in sorted(models):
        if verbose:
            print(f"Loading {model}")
        _dir = os.path.join(directory, model)
        train, test = load_multiple(_dir, verbose=verbose)
        train['model'] = model
        train['nparameters'] = MODEL_SIZES[model]
        test['model'] = model
        test['nparameters'] = MODEL_SIZES[model]

        train_sets.append(train)
        test_sets.append(test)

    return pd.concat(train_sets), pd.concat(test_sets)


def concat_update(existing, other, repeat=False):
    for key in other.keys():
        if key in existing:
            if existing[key] != other[key] or repeat:
                current = existing[key]
                if isinstance(current, list):
                    current.append(other[key])
                else:
                    existing[key] = [current, other[key]]
        else:
            existing[key] = other[key]


def run_config(run, repeat=False):
    full = {}
    configs = (os.path.join(run, entry.name) for entry in os.scandir(run) if 'config' in entry.name)

    for config in sorted(configs):
        with open(config) as file:
            tmp = json.load(file)

        tmp['path'] = config
        concat_update(full, tmp, repeat=repeat)
    return full


def search_configs(criteria, configs):
    matches = []
    for run, config in configs.items():
        is_match = True
        for key, value in criteria.items():
            try:
                config_value = config[key]
                if config_value != value:
                    is_match = False
            except KeyError:
                is_match = False

        if is_match:
            matches.append(run)

    return matches
