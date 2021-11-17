import glob
from PIL import Image
import numpy as np
import torch
import random
import h5py
import torch.utils.data as data


class Dataset_h5_real(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):

        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        if train:
            random.shuffle(self.keys)
        h5f.close()

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key]).reshape(h5f[key].shape)
        h5f.close()

        if self.train:
            (H, W, C) = data.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch = data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            p = 0.5
            if random.random() > p: #RandomRot90
                patch = patch.transpose(1, 0, 2)
            if random.random() > p: #RandomHorizontalFlip
                patch = patch[:, ::-1, :]
            if random.random() > p: #RandomVerticalFlip
                patch = patch[::-1, :, :]
        else:
            patch = data

        patch = np.clip(patch.astype(np.float32)/255.0, 0.0, 1.0)
        noisy = patch[:, :, 0:3]
        clean = patch[:, :, 3:6]

        noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(noisy, (2, 0, 1)))).float()
        clean = torch.from_numpy(np.ascontiguousarray(np.transpose(clean, (2, 0, 1)))).float()

        return noisy, clean

    def __len__(self):
        return len(self.keys)
