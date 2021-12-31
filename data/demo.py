import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx
import numpy as np
import cv2
from PIL import Image
from scipy.signal import convolve2d
import torchvision.transforms.functional as tf

# -*- coding: utf-8 -*-

from petrel_client.client import Client
from multiprocessing import Process
import logging
import random

LOG = logging.getLogger('petrel_client.test')

conf_path = '~/petreloss.conf'
client = Client(conf_path)
files = client.get_file_iterator('s3://denoising_data_bucket/nind/test')

for p, k in files:
    key = 's3://{}'.format(p)
    print(key)
