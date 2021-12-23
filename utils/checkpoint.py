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
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.util import *


def load_state(model, state_dict):
    """
    load state_dict to model
    :params model:
    :params state_dict:
    :return: model
    """

    if isinstance(model, nn.DataParallel) or isinstance(model, DDP):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model


def load_single_model(checkpoint_path, model, logger):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    psnr_best = check_point_params["psnr_best"]
    model = load_state(model, model_state)
    logger.info("load pretrained model and optimizer: psnr_best={}, model={}".format(psnr_best, model))
    return model, psnr_best


def load_ad_net_model(checkpoint_path, ad_net, logger):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["ad_net"]
    psnr_best = check_point_params["acc_best"]
    ad_net = load_state(ad_net, model_state)
    logger.info("load pretrained model and optimizer: acc_best={}, model={}".format(psnr_best, ad_net))
    return ad_net, psnr_best


def save_model(save_path, epoch, model, optimizer, psnr_best, logger):
    check_point_params = {}
    if isinstance(model, nn.DataParallel) or isinstance(model, DDP):
        check_point_params["model"] = model.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()

    check_point_params["optimizer"] = optimizer.state_dict()
    check_point_params['epoch'] = epoch
    check_point_params["psnr_best"] = psnr_best

    torch.save(check_point_params, save_path)
    logger.info('||==> Epoch={}, save model and optimizer, psnr_best={}'.format(epoch, psnr_best))


def save_ad_net(save_path, epoch, ad_net, optimizer, acc_best, logger):
    check_point_params = {}
    if isinstance(ad_net, nn.DataParallel) or isinstance(ad_net, DDP):
        check_point_params["ad_net"] = ad_net.module.state_dict()
    else:
        check_point_params["ad_net"] = ad_net.state_dict()

    check_point_params["optimizer"] = optimizer.state_dict()
    check_point_params['epoch'] = epoch
    check_point_params["acc_best"] = acc_best

    torch.save(check_point_params, save_path)
    logger.info('||==> Epoch={}, save model and optimizer, acc_best={}'.format(epoch, acc_best))


def save_model_ad_net(save_path, epoch, model, ad_net, optimizer, optimizer_ad, psnr_best, logger):
    check_point_params = {}
    if isinstance(model, nn.DataParallel) or isinstance(model, DDP):
        check_point_params["model"] = model.module.state_dict()
        check_point_params["ad_net"] = ad_net.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()
        check_point_params["ad_net"] = ad_net.state_dict()

    check_point_params["optimizer"] = optimizer.state_dict()
    check_point_params["optimizer_ad"] = optimizer_ad.state_dict()
    check_point_params['epoch'] = epoch
    check_point_params["psnr_best"] = psnr_best

    torch.save(check_point_params, save_path)
    logger.info('||==> Epoch={}, save model and optimizer, psnr_best={}'.format(epoch, psnr_best))


def load_model_dp(checkpoint_path, model, optimizer, logger):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    optimizer.load_state_dict(check_point_params["optimizer"])
    start_epoch = check_point_params['epoch']
    psnr_best = check_point_params["psnr_best"]

    model = load_state(model, model_state)
    logger.info("load pretrained model and optimizer: epoch={}, psnr_best={}, model={}, optimizer={}"
                .format(start_epoch, psnr_best, model, optimizer))
    return model, start_epoch, optimizer, psnr_best


def load_ad_net_dp(checkpoint_path, model, optimizer, logger):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    optimizer.load_state_dict(check_point_params["optimizer"])
    start_epoch = check_point_params['epoch']
    acc_best = check_point_params["acc_best"]

    model = load_state(model, model_state)
    logger.info("load pretrained model and optimizer: epoch={}, acc_best={}, model={}, optimizer={}"
                .format(start_epoch, acc_best, model, optimizer))
    return model, start_epoch, optimizer, acc_best


def load_model(checkpoint_path, model, optimizer, logger, local_rank):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    optimizer.load_state_dict(check_point_params["optimizer"])
    start_epoch = check_point_params['epoch']
    psnr_best = check_point_params["psnr_best"]

    model = load_state(model, model_state)
    ddp_logger_info("load pretrained model and optimizer: epoch={}, psnr_best={}, model={}, optimizer={}"
                    .format(start_epoch, psnr_best, model, optimizer), logger, local_rank)
    return model, start_epoch, optimizer, psnr_best


def load_ad_net(checkpoint_path, ad_net, logger, local_rank):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["ad_net"]
    start_epoch = check_point_params['epoch']
    acc_best = check_point_params["acc_best"]

    ad_net = load_state(ad_net, model_state)
    ddp_logger_info("load pretrained model and optimizer: epoch={}, acc_best={}, model={}"
                    .format(start_epoch, acc_best, ad_net), logger, local_rank)
    return ad_net, start_epoch, acc_best



def load_model_ad_net(checkpoint_path, model, ad_net, optimizer, optimizer_ad, logger, local_rank):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    ad_net_state = check_point_params["ad_net"]
    optimizer.load_state_dict(check_point_params["optimizer"])
    optimizer_ad.load_state_dict(check_point_params["optimizer_ad"])
    start_epoch = check_point_params['epoch']
    psnr_best = check_point_params["psnr_best"]

    model = load_state(model, model_state)
    ad_net = load_state(ad_net, ad_net_state)

    ddp_logger_info("load pretrained model and optimizer: epoch={}, psnr_best={}, model={}, ad_net={}, optimizer={}, optimizer_ad={}"
                    .format(start_epoch, psnr_best, model, ad_net, optimizer, optimizer_ad), logger, local_rank)
    return model, ad_net, start_epoch, optimizer, optimizer_ad, psnr_best
