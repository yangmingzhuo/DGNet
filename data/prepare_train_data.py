import argparse
import os
import numpy as np
import glob
import random
import cv2
from scipy.io import loadmat
from tqdm import tqdm
import shutil
from data_utils import *


def prepare_sidd_data(src_path, dst_path, patch_size=256, rand_num_train=300):
    dst_path = make_dir(os.path.join(dst_path, 'sidd'))
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    # prepare training data
    print('SIDD train data processing...')
    scene_path = glob.glob(src_path)
    scene_path.sort()
    for scene_num, scene_name in enumerate(tqdm(scene_path), 0):
        gt_imgs = glob.glob(scene_name + '/*GT*.PNG')
        gt_imgs.sort()
        noisy_imgs = glob.glob(scene_name + '/*NOISY*.PNG')
        noisy_imgs.sort()

        for img_num in range(len(noisy_imgs)):
            gt = np.array(cv2.imread(gt_imgs[img_num]))
            noisy = np.array(cv2.imread(noisy_imgs[img_num]))
            img = np.concatenate([noisy, gt], 2)
            [h, w, c] = img.shape

            patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=True,
                                    rand_num_train=rand_num_train)
            for patch_num in range(len(patch_list)):
                noisy_patch = patch_list[patch_num][:, :, 0:3]
                clean_patch = patch_list[patch_num][:, :, 3:6]
                img = np.concatenate([noisy_patch, clean_patch], 1)
                cv2.imwrite(os.path.join(dst_path_train,
                                         'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1,
                                                                                           img_num + 1,
                                                                                           patch_num + 1)), img)


def prepare_renoir_data(src_path, dst_path, patch_size, rand_num_train=300):
    dst_path = make_dir(os.path.join(dst_path, 'renoir'))
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    file_path = glob.glob(src_path)
    file_path.sort()
    # prepare training data
    print('RENOIR train data processing...')
    for scene_num, file_name in enumerate(tqdm(file_path), 0):
        if 'RENOIR' in file_name:
            ref_path = glob.glob(file_name + '/*Reference.bmp')
            full_path = glob.glob(file_name + '/*full.bmp')
            noisy_paths = glob.glob(file_name + '/*Noisy.bmp')
            noisy_paths.sort()
            ref = np.array(cv2.imread(ref_path[0])).astype(np.float32)
            full = np.array(cv2.imread(full_path[0])).astype(np.float32)
            gt = (ref + full) / 2
            gt = np.clip(gt, 0, 255).astype(np.uint8)
            for img_num in range(len(noisy_paths)):
                noisy = np.array(cv2.imread(noisy_paths[img_num]))
                img = np.concatenate([noisy, gt], 2)
                [h, w, c] = img.shape
                patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=True,
                                        rand_num_train=rand_num_train)
                for patch_num in range(len(patch_list)):
                    noisy_patch = patch_list[patch_num][:, :, 0:3]
                    clean_patch = patch_list[patch_num][:, :, 3:6]
                    img = np.concatenate([noisy_patch, clean_patch], 1)
                    cv2.imwrite(os.path.join(dst_path_train,
                                             'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1,
                                                                                               img_num + 1,
                                                                                               patch_num + 1)), img)


def prepare_polyu_data(src_path, dst_path, patch_size, rand_num_train=300):
    dst_path = make_dir(os.path.join(dst_path, 'polyu'))
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    file_path = glob.glob(src_path)
    file_path.sort()
    # prepare training data
    print('PolyU train data processing...')
    for scene_num, file_name in enumerate(tqdm(file_path), 0):
        if 'PolyU' in file_name:
            noisy_paths = glob.glob(file_name + '/*.JPG')
            gt_path = glob.glob(file_name + '/mean.png')

            for img_num in range(len(noisy_paths)):
                noisy = np.array(cv2.imread(noisy_paths[img_num]))
                gt = np.array(cv2.imread(gt_path[img_num]))
                img = np.concatenate([noisy, gt], 2)
                [h, w, c] = img.shape
                patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=True,
                                        rand_num_train=rand_num_train)
                for patch_num in range(len(patch_list)):
                    noisy_patch = patch_list[patch_num][:, :, 0:3]
                    clean_patch = patch_list[patch_num][:, :, 3:6]
                    img = np.concatenate([noisy_patch, clean_patch], 1)
                    cv2.imwrite(os.path.join(dst_path_train,
                                             'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1,
                                                                                               img_num + 1,
                                                                                               patch_num + 1)), img)


def main():
    parser = argparse.ArgumentParser(description='PyTorch prepare train data')
    parser.add_argument('--patch_size', type=int, default=256, help='size of cropped image')
    parser.add_argument('--data_set', type=str, default='sidd', help='the dataset to crop')
    parser.add_argument('--src_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/split_dataset/',
                        help='the path of dataset dir')
    parser.add_argument('--dst_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/processed/',
                        help='the path of destination dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use default=0')
    parser.add_argument('--rand_num_train', type=int, default=300, help='training patch number to randomly crop images')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    patch_size = opt.patch_size

    root_dir = opt.src_dir
    sidd_src_path_train = os.path.join(root_dir, 'sidd', 'train')
    renoir_src_path_train = os.path.join(root_dir, 'renoir', 'train')
    polyu_src_path_train = os.path.join(root_dir, 'polyu', 'train')
    if opt.data_set == 'sidd':
        print("start...")
        print("start...SIDD...")
        prepare_sidd_data(sidd_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...SIDD")
    elif opt.data_set == 'renoir':
        print("start...RENOIR...")
        prepare_renoir_data(renoir_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...RENOIR")
    elif opt.data_set == 'polyu':
        print("start...PolyU...")
        prepare_polyu_data(polyu_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...PolyU")
    print('end')


if __name__ == "__main__":
    main()
