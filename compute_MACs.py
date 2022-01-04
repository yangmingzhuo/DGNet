#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/4 16:59
# @Author  : zheng
# @Site    : 
# @File    : compute_MACs.py
# @Software: PyCharm

import torch
from thop import profile
from model.ELD_UNet import ELD_UNet


model = ELD_UNet()
input = torch.randn(1, 3, 128, 128)
mac, params = profile(model, inputs=(input, ))

mac = mac / 1e9
params = params / 1e9
print('--'*20)
print('MAC = {}G, FLOPS={}G'.format(mac, mac*2))
print("params = {}G".format(params))

