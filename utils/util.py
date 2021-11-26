import os
import sys
import logging
import torch
import glob
from collections import OrderedDict
import shutil


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def output_process(output_path, opt):
    if os.path.exists(output_path):
        print("{} file exist!".format(output_path))
        action = opt.lower().strip()
        act = action
        if act == 'd':
            shutil.rmtree(output_path)
        else:
            raise OSError("Directory {} exits!".format(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)


def get_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger


def print_network(net, logger):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info('model={}'.format(net))
    logger.info('Total number of parameters: {}'.format(num_params))


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
