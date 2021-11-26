#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 15:40
# @Author  : zheng
# @Site    : 
# @File    : checkpoint.py
# @Software: PyCharm

import os
import glob
from collections import OrderedDict

import torch
from torch import nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_state(model, state_dict):
    """
    load state_dict to model
    :params model:
    :params state_dict:
    :return: model
    """

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model


def save_model(save_path, epoch, model, optimizer, psnr_best, logger):
    check_point_params = {}
    if isinstance(model, nn.DataParallel):
        check_point_params["model"] = model.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()

    check_point_params["optimizer"] = optimizer.state_dict()
    check_point_params['epoch'] = epoch
    check_point_params["psnr_best"] = psnr_best

    mkdir(save_path)
    torch.save(check_point_params, save_path)
    logger.info('epoch={}, save model and optimizer, psnr_best={}'.format(epoch, psnr_best))


def load_model(checkpoint_path, model, optimizer, logger):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    optimizer.load_state_dict(check_point_params["optimizer"])
    start_epoch = check_point_params['epoch']
    psnr_best = check_point_params["psnr_best"]

    model = load_state(model, model_state)

    logger.info("load pretrained model and optimizer: epoch={}, psnr_best={}, model={}, optimizer={}"
                .format(start_epoch, psnr_best, model, optimizer))
    return model, start_epoch, optimizer, psnr_best


def load_single_model(checkpoint_path, model, logger):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    psnr_best = check_point_params["psnr_best"]
    model = load_state(model, model_state)

    logger.info("load pretrained model and optimizer: psnr_best={}, model={}".format(psnr_best, model))
    return model, psnr_best
