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
    scene_paths = glob.glob(os.path.join(src_path, '*'))
    scene_paths.sort()
    for scene_num, scene_path in enumerate(tqdm(scene_paths), 0):
        scene_name = os.path.basename(scene_path)
        gt_imgs = glob.glob(scene_path + '/*GT*.PNG')
        gt_imgs.sort()
        noisy_imgs = glob.glob(scene_path + '/*NOISY*.PNG')
        noisy_imgs.sort()

        for img_num in range(len(noisy_imgs)):
            gt = np.array(cv2.imread(gt_imgs[img_num]))
            noisy = np.array(cv2.imread(noisy_imgs[img_num]))
            img = np.concatenate([noisy, gt], 2)
            [h, w, c] = img.shape

            patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=True,
                                    crop_num=rand_num_train)
            for patch_num in range(len(patch_list)):
                noisy_patch = patch_list[patch_num][:, :, 0:3]
                clean_patch = patch_list[patch_num][:, :, 3:6]
                img = np.concatenate([noisy_patch, clean_patch], 1)
                cv2.imwrite(os.path.join(dst_path_train,
                                         '{}_img_{:03d}_patch_{:03d}.png'.format(scene_name,
                                                                                 img_num + 1,
                                                                                 patch_num + 1)), img)


def prepare_renoir_data(src_path, dst_path, patch_size, rand_num_train=300):
    dst_path = make_dir(os.path.join(dst_path, 'renoir'))
    dst_path_train = os.path.join(dst_path, 'train_old')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    scene_paths = glob.glob(os.path.join(src_path, '*'))
    scene_paths.sort()
    # prepare training data
    print('RENOIR train data processing...')
    for scene_num, scene_path in enumerate(tqdm(scene_paths), 0):
        scene_name = os.path.basename(scene_path)
        ref_path = glob.glob(os.path.join(scene_path, '*Reference.bmp'))
        full_path = glob.glob(os.path.join(scene_path, '*full.bmp'))
        noisy_paths = glob.glob(os.path.join(scene_path, '*Noisy.bmp'))
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
                                    crop_num=rand_num_train)
            for patch_num in range(len(patch_list)):
                noisy_patch = patch_list[patch_num][:, :, 0:3]
                clean_patch = patch_list[patch_num][:, :, 3:6]
                img = np.concatenate([noisy_patch, clean_patch], 1)
                cv2.imwrite(os.path.join(dst_path_train,
                                         '{}_img_{:03d}_patch_{:03d}.png'.format(scene_name,
                                                                                 img_num + 1,
                                                                                 patch_num + 1)), img)


def prepare_polyu_data(src_path, dst_path, patch_size, rand_num_train=300):
    dst_path = make_dir(os.path.join(dst_path, 'polyu'))
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    # prepare training data
    print('PolyU train data processing...')
    noisy_paths = glob.glob(os.path.join(src_path, '*Real.JPG'))
    noisy_paths.sort()
    gt_paths = glob.glob(os.path.join(src_path, '*mean.JPG'))
    gt_paths.sort()

    for img_num, noisy_path in enumerate(tqdm(noisy_paths)):
        noisy = np.array(cv2.imread(noisy_path))
        gt = np.array(cv2.imread(gt_paths[img_num]))
        img = np.concatenate([noisy, gt], 2)
        [h, w, c] = img.shape
        patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=True,
                                crop_num=rand_num_train)
        for patch_num in range(len(patch_list)):
            noisy_patch = patch_list[patch_num][:, :, 0:3]
            clean_patch = patch_list[patch_num][:, :, 3:6]
            img = np.concatenate([noisy_patch, clean_patch], 1)
            cv2.imwrite(os.path.join(dst_path_train,
                                     '{}_img_{:03d}_patch_{:03d}.png'.format(
                                         os.path.join(os.path.basename(noisy_path).replace('_Real.JPG', '')),
                                         img_num + 1, patch_num + 1)), img)


def prepare_nind_data(src_path, dst_path, patch_size, rand_num_train=300):
    dst_path = make_dir(os.path.join(dst_path, 'nind'))
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    scene_paths = glob.glob(os.path.join(src_path, '*'))
    scene_paths.sort()
    # prepare training data
    print('NIND train data processing...')
    for scene_num, scene_path in enumerate(tqdm(scene_paths), 0):
        scene_name = os.path.basename(scene_path)
        gt_path = glob.glob(os.path.join(scene_path, '*gt.png'))
        noisy_paths = glob.glob(os.path.join(scene_path, '*ISO*'))
        noisy_paths.sort()
        gt = cv2.imread(gt_path[0])
        for img_num in range(len(noisy_paths)):
            noisy = np.array(cv2.imread(noisy_paths[img_num]))
            img = np.concatenate([noisy, gt], 2)
            [h, w, c] = img.shape
            patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=True,
                                    crop_num=rand_num_train)
            for patch_num in range(len(patch_list)):
                noisy_patch = patch_list[patch_num][:, :, 0:3]
                clean_patch = patch_list[patch_num][:, :, 3:6]
                img = np.concatenate([noisy_patch, clean_patch], 1)
                cv2.imwrite(os.path.join(dst_path_train,
                                         '{}_img_{:03d}_patch_{:03d}.png'.format(scene_name,
                                                                                 img_num + 1,
                                                                                 patch_num + 1)), img)


def prepare_rid2021_data(src_path, dst_path, patch_size, rand_num_train=300):
    dst_path = make_dir(os.path.join(dst_path, 'rid2021'))
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    # prepare training data
    print('RID2021 train data processing...')
    noisy_paths = glob.glob(os.path.join(src_path, '*_noisy.jpeg'))
    noisy_paths.sort()
    gt_paths = glob.glob(os.path.join(src_path, '*_gt.jpeg'))
    gt_paths.sort()

    for img_num, noisy_path in enumerate(tqdm(noisy_paths)):
        noisy = np.array(cv2.imread(noisy_path))
        gt = np.array(cv2.imread(gt_paths[img_num]))
        img = np.concatenate([noisy, gt], 2)
        [h, w, c] = img.shape
        patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=True,
                                crop_num=rand_num_train)
        for patch_num in range(len(patch_list)):
            noisy_patch = patch_list[patch_num][:, :, 0:3]
            clean_patch = patch_list[patch_num][:, :, 3:6]
            img = np.concatenate([noisy_patch, clean_patch], 1)
            cv2.imwrite(os.path.join(dst_path_train,
                                     '{}_img_{:03d}_patch_{:03d}.png'.format(
                                         os.path.join(os.path.basename(noisy_path).replace('_noisy.jpeg', '')),
                                         img_num + 1, patch_num + 1)), img)


def main():
    parser = argparse.ArgumentParser(description='PyTorch prepare train data')
    parser.add_argument('--patch_size', type=int, default=256, help='size of cropped image')
    parser.add_argument('--data_set', type=str, default='sidd', help='the dataset to crop')
    parser.add_argument('--src_dir', type=str, default='/home/SENSETIME/yangmingzhuo/Documents/ECCV/split_dataset/',
                        help='the path of dataset dir')
    parser.add_argument('--dst_dir', type=str, default='/home/SENSETIME/yangmingzhuo/Documents/ECCV/processed/',
                        help='the path of destination dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use default=0')
    parser.add_argument('--rand_num_train', type=int, default=300, help='training patch number to randomly crop images')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    patch_size = opt.patch_size

    root_dir = opt.src_dir
    print("start...")
    if opt.data_set == 'sidd':
        print("start...SIDD...")
        sidd_src_path_train = os.path.join(root_dir, 'sidd', 'train')
        prepare_sidd_data(sidd_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...SIDD")
    elif opt.data_set == 'renoir':
        print("start...RENOIR...")
        renoir_src_path_train = os.path.join(root_dir, 'renoir', 'train')
        prepare_renoir_data(renoir_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...RENOIR")
    elif opt.data_set == 'polyu':
        print("start...PolyU...")
        polyu_src_path_train = os.path.join(root_dir, 'polyu', 'train')
        prepare_polyu_data(polyu_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...PolyU")
    elif opt.data_set == 'nind':
        print("start...NIND...")
        nind_src_path_train = os.path.join(root_dir, 'nind', 'train')
        prepare_nind_data(nind_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...NIND")
    elif opt.data_set == 'rid2021':
        print("start...RID2021...")
        nind_src_path_train = os.path.join(root_dir, 'rid2021', 'train')
        prepare_rid2021_data(nind_src_path_train, opt.dst_dir, patch_size, opt.rand_num_train)
        print("end...RID2021")
    print('end')


if __name__ == "__main__":
    main()
