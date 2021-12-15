import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx
import numpy as np
import cv2
from PIL import Image
from scipy.signal import convolve2d

img1 = cv2.imread('/home/SENSETIME/yangmingzhuo/Documents/ECCV/processed/renoir/test/Mi3_Aligned_Batch_005_img_002_patch_002.png')
img2 = cv2.imread('/home/SENSETIME/yangmingzhuo/Documents/ECCV/dst/9.png')
print(img1.shape)
print(img2.shape)
img3 = img2.transpose((1, 0, 2))
print(compare_psnr(img1[:, 256:, :], img3))
