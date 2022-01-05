
import csv
import json
import os
import shutil
from collections import defaultdict

import numpy as np

import torch
import torchvision
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter


COMMON_TRAIN_FORMAT = [('step', 'S', 'int'),
                       ('master_loss', 'ML', 'float'), ('sub_loss', 'SL', 'float')
                       ]




class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self,
                 log_dir,
                 save_tb=False,
                 log_frequency=10000,
                 action_repeat=1,
                 agent='drq'):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        self._action_repeat = action_repeat
        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        train_format = COMMON_TRAIN_FORMAT

        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT)

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0