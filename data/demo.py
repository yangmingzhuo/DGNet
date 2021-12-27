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
import torchvision.transforms.functional as tf

# img1 = np.array(Image.open('/home/SENSETIME/yangmingzhuo/Documents/ECCV/processed/renoir/test/12_Mi3_Aligned_Batch_005_img_001_patch_075.png'))
# noisy_img = tf.to_tensor(img1[:, :256, :])
# p = 0.5
# list_random = [0, 0, 0]
# if list_random[0] > p:
#     noisy_img = torch.rot90(noisy_img, dims=(1, 2))
# if list_random[0] > p:
#     noisy_img = noisy_img.flip(1)
# if list_random[0] > p:
#     noisy_img = noisy_img.flip(2)
# cv2.imwrite('/home/SENSETIME/yangmingzhuo/dst/{}{}{}.png'.format(list_random[0], list_random[1], list_random[2]), cv2.cvtColor(img_as_ubyte(noisy_img.data.cpu().permute(1, 2, 0).numpy().astype(np.float32)), cv2.COLOR_RGB2BGR))
#
# print(np.exp(1), np.exp(2))
# print(torch.exp(torch.Tensor(1).fill_(1)), torch.exp(torch.Tensor(1).fill_(2)))

from torchvision.models import resnet18

model = resnet18()
print(model)