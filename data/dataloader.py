import glob
import os.path

from PIL import Image, ImageFile
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms.functional as tf
import h5py
import cv2
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from petrel_client.client import Client
from data.data_utils import *


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class LoadDataset(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):

        self.path = src_path
        files = glob.glob(os.path.join(src_path, '*.png'))
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
        self.img_paths = []

        files1 = glob.glob(os.path.join(src_path1, '*.png'))
        # files1.sort()
        self.len1 = len(files1)
        for file_name in files1:
            self.img_paths.append((file_name, 0))

        files2 = glob.glob(os.path.join(src_path2, '*.png'))
        # files2.sort()
        self.len2 = len(files2)
        for file_name in files2:
            self.img_paths.append((file_name, 1))

        files3 = glob.glob(os.path.join(src_path3, '*.png'))
        # files3.sort()
        self.len3 = len(files3)
        for file_name in files3:
            self.img_paths.append((file_name, 2))

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        img_array = np.array(Image.open(img_path))
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

        return noisy_img, clean_img, label

    def __len__(self):
        return len(self.img_paths)


class LoadMultiDataset_clean(data.Dataset):

    def __init__(self, src_path1, src_path2, src_path3, patch_size=128, train=True):
        self.img_paths = []

        files1 = glob.glob(os.path.join(src_path1, '*.png'))
        # files1.sort()
        self.len1 = len(files1)
        for file_name in files1:
            self.img_paths.append((file_name, 0))

        files2 = glob.glob(os.path.join(src_path2, '*.png'))
        # files2.sort()
        self.len2 = len(files2)
        for file_name in files2:
            self.img_paths.append((file_name, 1))

        files3 = glob.glob(os.path.join(src_path3, '*.png'))
        # files3.sort()
        self.len3 = len(files3)
        for file_name in files3:
            self.img_paths.append((file_name, 2))

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        img_array = np.array(Image.open(img_path))
        noisy, clean = np.split(img_array, 2, axis=1)
        patch_size = self.patch_size

        # noisy_img = tf.to_tensor(noisy)
        clean_img = tf.to_tensor(clean)

        if self.train:
            # crop Patch
            height, width = clean_img.shape[1], clean_img.shape[2]
            rr = random.randint(0, height - patch_size)
            cc = random.randint(0, width - patch_size)
            # noisy_img = noisy_img[:, rr:rr + patch_size, cc:cc + patch_size]
            clean_img = clean_img[:, rr:rr + patch_size, cc:cc + patch_size]
            # data Augmentation
            p = 0.5
            if random.random() > p:
                # noisy_img = torch.rot90(noisy_img, dims=(1, 2))
                clean_img = torch.rot90(clean_img, dims=(1, 2))
            if random.random() > p:
                # noisy_img = noisy_img.flip(1)
                clean_img = clean_img.flip(1)
            if random.random() > p:
                # noisy_img = noisy_img.flip(2)
                clean_img = clean_img.flip(2)

        return clean_img, label

    def __len__(self):
        return len(self.img_paths)


class LoadH5Dataset(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):
        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        h5f = h5py.File(self.src_path, 'r')
        key = self.keys[index]
        img_data = np.array(h5f[key])
        h5f.close()

        noisy = img_data[:, :, 0:3]
        clean = img_data[:, :, 3:6]
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
        return len(self.keys)


class LoadMultiH5Dataset(data.Dataset):

    def __init__(self, src_path1, src_path2, src_path3, patch_size=128, train=True):
        self.src_path1 = src_path1
        h5f1 = h5py.File(src_path1, 'r')
        self.keys1 = list(h5f1.keys())
        h5f1.close()

        self.src_path2 = src_path2
        h5f2 = h5py.File(src_path2, 'r')
        self.keys2 = list(h5f2.keys())
        h5f2.close()

        self.src_path3 = src_path3
        h5f3 = h5py.File(src_path3, 'r')
        self.keys3 = list(h5f3.keys())
        h5f3.close()

        self.len1 = len(self.keys1)
        self.len2 = len(self.keys2)
        self.len3 = len(self.keys3)

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        if index < self.len1:
            h5f1 = h5py.File(self.src_path1, 'r')
            key = self.keys1[index]
            img_data = np.array(h5f1[key])
            label = 1
            h5f1.close()
        elif index < self.len2 + self.len1:
            h5f2 = h5py.File(self.src_path2, 'r')
            key = self.keys2[index - self.len1]
            img_data = np.array(h5f2[key])
            label = 2
            h5f2.close()
        else:
            h5f3 = h5py.File(self.src_path3, 'r')
            key = self.keys3[index - self.len2 - self.len1]
            img_data = np.array(h5f3[key])
            label = 3
            h5f3.close()

        noisy = img_data[:, :, 0:3]
        clean = img_data[:, :, 3:6]
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

        return noisy_img, clean_img, label

    def __len__(self):
        return self.len1 + self.len2 + self.len3


class LoadDataset_ceph(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):

        self.path = src_path
        self.img_paths = []

        conf_path = '~/petreloss.conf'
        self.client = Client(conf_path)
        files = self.client.get_file_iterator(src_path)
        for p, k in files:
            self.img_paths.append('s3://{}'.format(p))
        self.img_paths.sort()

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_bytes = self.client.get(self.img_paths[index])
        assert (img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
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


class LoadMultiDataset_ceph(data.Dataset):

    def __init__(self, src_path1, src_path2, src_path3, patch_size=128, train=True):
        self.img_paths = []
        conf_path = '~/petreloss.conf'
        self.client = Client(conf_path)
        files1 = self.client.get_file_iterator(src_path1)
        files2 = self.client.get_file_iterator(src_path2)
        files3 = self.client.get_file_iterator(src_path3)

        self.len1 = 0
        for p, k in files1:
            self.img_paths.append(('s3://{}'.format(p), 0))
            self.len1 += 1

        self.len2 = 0
        for p, k in files2:
            self.img_paths.append(('s3://{}'.format(p), 1))
            self.len2 += 1

        self.len3 = 0
        for p, k in files3:
            self.img_paths.append(('s3://{}'.format(p), 2))
            self.len3 += 1

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_name, label = self.img_paths[index]
        img_bytes = self.client.get(img_name)
        assert (img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
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

        return noisy_img, clean_img, label

    def __len__(self):
        return len(self.img_paths)


class LoadDataset_ceph_pd(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):

        self.path = src_path
        self.img_paths = []

        conf_path = '~/petreloss.conf'
        self.client = Client(conf_path)
        files = self.client.get_file_iterator(src_path)
        for p, k in files:
            self.img_paths.append('s3://{}'.format(p))
        self.img_paths.sort()

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_bytes = self.client.get(self.img_paths[index])
        assert (img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        noisy, clean = np.split(img_array, 2, axis=1)
        patch_size = self.patch_size
        noisy = pixelshuffle(noisy, 2)
        clean = pixelshuffle(clean, 2)
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


class LoadMultiDataset_ceph_pd(data.Dataset):

    def __init__(self, src_path1, src_path2, src_path3, patch_size=128, train=True):
        self.img_paths = []
        conf_path = '~/petreloss.conf'
        self.client = Client(conf_path)
        files1 = self.client.get_file_iterator(src_path1)
        files2 = self.client.get_file_iterator(src_path2)
        files3 = self.client.get_file_iterator(src_path3)

        self.len1 = 0
        for p, k in files1:
            self.img_paths.append(('s3://{}'.format(p), 0))
            self.len1 += 1

        self.len2 = 0
        for p, k in files2:
            self.img_paths.append(('s3://{}'.format(p), 1))
            self.len2 += 1

        self.len3 = 0
        for p, k in files3:
            self.img_paths.append(('s3://{}'.format(p), 2))
            self.len3 += 1

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_name, label = self.img_paths[index]
        img_bytes = self.client.get(img_name)
        assert (img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        noisy, clean = np.split(img_array, 2, axis=1)
        noisy = pixelshuffle(noisy, 2)
        clean = pixelshuffle(clean, 2)
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

        return noisy_img, clean_img, label

    def __len__(self):
        return len(self.img_paths)


class LoadDataset_ceph_pd_2(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):

        self.path = src_path
        self.img_paths = []

        conf_path = '~/petreloss.conf'
        self.client = Client(conf_path)
        files = self.client.get_file_iterator(src_path)
        for p, k in files:
            self.img_paths.append('s3://{}'.format(p))
        self.img_paths.sort()

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_bytes = self.client.get(self.img_paths[index])
        assert (img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        noisy, clean = np.split(img_array, 2, axis=1)
        patch_size = self.patch_size
        noisy = pixelshuffle_2(noisy, 2)
        clean = pixelshuffle_2(clean, 2)
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


class LoadMultiDataset_ceph_pd_2(data.Dataset):

    def __init__(self, src_path1, src_path2, src_path3, patch_size=128, train=True):
        self.img_paths = []
        conf_path = '~/petreloss.conf'
        self.client = Client(conf_path)
        files1 = self.client.get_file_iterator(src_path1)
        files2 = self.client.get_file_iterator(src_path2)
        files3 = self.client.get_file_iterator(src_path3)

        self.len1 = 0
        for p, k in files1:
            self.img_paths.append(('s3://{}'.format(p), 0))
            self.len1 += 1

        self.len2 = 0
        for p, k in files2:
            self.img_paths.append(('s3://{}'.format(p), 1))
            self.len2 += 1

        self.len3 = 0
        for p, k in files3:
            self.img_paths.append(('s3://{}'.format(p), 2))
            self.len3 += 1

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        img_name, label = self.img_paths[index]
        img_bytes = self.client.get(img_name)
        assert (img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        noisy, clean = np.split(img_array, 2, axis=1)
        noisy = pixelshuffle_2(noisy, 2)
        clean = pixelshuffle_2(clean, 2)
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

        return noisy_img, clean_img, label

    def __len__(self):
        return len(self.img_paths)
