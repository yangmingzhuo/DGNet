import glob
from PIL import Image, ImageFile
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as tf


ImageFile.LOAD_TRUNCATED_IMAGES = True
torchvision.set_image_backend('accimage')


class LoadDataset(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):

        self.path = src_path
        files = glob.glob(src_path + '/*.png')
        files.sort()
        self.img_paths = []
        for file_name in files:
            self.img_paths.append(file_name)
        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_array = np.array(Image.open(self.img_paths[index]))
        noisy, clean = np.split(img_array, 2, axis=1)
        patch_size = self.patch_size

        noisy_img = tf.to_tensor(noisy)
        clean_img = tf.to_tensor(clean)

        if self.train:
            # crop Patch
            height, width = noisy_img.shape[1], clean_img.shape[2]
            rr = random.randint(0, height - patch_size)
            cc = random.randint(0, width - patch_size)
            noisy_img = noisy_img[:, rr:rr + patch_size, cc:cc + patch_size]
            clean_img = clean_img[:, rr:rr + patch_size, cc:cc + patch_size]
            # data Augmentation
            p = 0.5
            if random.random() > p:
                noisy_img = torch.rot90(noisy_img, dims=(1, 2))
                clean_img = torch.rot90(clean_img, dims=(1, 2))
            if random.random() > p:
                noisy_img = noisy_img.flip(1)
                clean_img = clean_img.flip(1)
            if random.random() > p:
                noisy_img = noisy_img.flip(2)
                clean_img = clean_img.flip(2)

        return noisy_img, clean_img

    def __len__(self):
        return len(self.img_paths)


class LoadMultiDataset(data.Dataset):

    def __init__(self, src_path1, src_path2, src_path3, patch_size=128, train=True):

        files1 = glob.glob(src_path1 + '/*.png')
        files1.sort()
        self.img_paths = []
        for file_name in files1:
            self.img_paths.append(file_name)
        files2 = glob.glob(src_path2 + '/*.png')
        files2.sort()
        for file_name in files2:
            self.img_paths.append(file_name)
        files3 = glob.glob(src_path3 + '/*.png')
        files3.sort()
        for file_name in files3:
            self.img_paths.append(file_name)
        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_array = np.array(Image.open(self.img_paths[index]))
        noisy, clean = np.split(img_array, 2, axis=1)
        patch_size = self.patch_size

        noisy_img = tf.to_tensor(noisy)
        clean_img = tf.to_tensor(clean)

        if self.train:
            # crop Patch
            height, width = noisy_img.shape[1], clean_img.shape[2]
            rr = random.randint(0, height - patch_size)
            cc = random.randint(0, width - patch_size)
            noisy_img = noisy_img[:, rr:rr + patch_size, cc:cc + patch_size]
            clean_img = clean_img[:, rr:rr + patch_size, cc:cc + patch_size]
            # data Augmentation
            p = 0.5
            if random.random() > p:
                noisy_img = torch.rot90(noisy_img, dims=(1, 2))
                clean_img = torch.rot90(clean_img, dims=(1, 2))
            if random.random() > p:
                noisy_img = noisy_img.flip(1)
                clean_img = clean_img.flip(1)
            if random.random() > p:
                noisy_img = noisy_img.flip(2)
                clean_img = clean_img.flip(2)

        return noisy_img, clean_img

    def __len__(self):
        return len(self.img_paths)
