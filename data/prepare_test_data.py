import argparse
import os
import random

import numpy as np
import glob
from skimage.measure import compare_psnr, compare_ssim
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from scipy.io import loadmat
from tqdm import tqdm
import shutil
from data_utils import *


def prepare_sidd_data(src_path, dst_path, patch_size):
    dst_path = make_dir(os.path.join(dst_path, 'sidd'))
    dst_path_test = os.path.join(dst_path, 'test')
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)

    noisy_data_mat_file = os.path.join(src_path, 'ValidationNoisyBlocksSrgb.mat')
    clean_data_mat_file = os.path.join(src_path, 'ValidationGtBlocksSrgb.mat')
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
    clean_data_mat_name = os.path.basename(clean_data_mat_file).replace('.mat', '')
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]
    clean_data_mat = loadmat(clean_data_mat_file)[clean_data_mat_name]

    # prepare testing data
    print('SIDD test data processing...')
    for image_index in tqdm(range(noisy_data_mat.shape[0])):
        for block_index in range(noisy_data_mat.shape[1]):
            noisy_image = noisy_data_mat[image_index, block_index, :, :, :]
            noisy_image = np.float32(noisy_image)
            clean_image = clean_data_mat[image_index, block_index, :, :, :]
            clean_image = np.float32(clean_image)
            img = np.concatenate([noisy_image, clean_image], 1)
            cv2.imwrite(
                os.path.join(dst_path_test, 'scene_{:03d}_patch_{:03d}.png'.format(image_index + 1, block_index + 1)),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def prepare_renoir_data(src_path, dst_path, patch_size):
    dst_path = make_dir(os.path.join(dst_path, 'renoir'))
    dst_path_test = os.path.join(dst_path, 'test_3')
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)
    path_real = '/home/SENSETIME/yangmingzhuo/Documents/ECCV/dst/'
    scene_paths = glob.glob(os.path.join(src_path, '*'))
    scene_paths.sort()
    print(scene_paths)
    # prepare testing data
    print('RENOIR test data processing...')
    total_num = 0
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
        # pos_list = get_pos_list(os.path.join(scene_path, 'patch_list.txt'))
        for img_num in range(len(noisy_paths)):

            noisy = np.array(cv2.imread(noisy_paths[img_num]))
            img = np.concatenate([noisy, gt], 2)
            [h, w, c] = img.shape

            patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=False)
            img_num_real = 0
            while 1:
                max_psnr = 0
                max_patch_num = 0
                if img_num_real >= 32 or img_num_real >= len(patch_list):
                    break
                for patch_num in range(len(patch_list)):
                    img2 = cv2.imread(os.path.join(path_real, str(total_num) + '.png'))
                    clean_patch = patch_list[patch_num][:, :, 3:6].transpose((1, 0, 2))
                    psnr = compare_psnr(img2, clean_patch)
                    if psnr > max_psnr:
                        max_psnr = psnr
                        max_patch_num = patch_num
                    print(psnr, max_psnr, patch_num, img_num_real, total_num)
                noisy_patch = patch_list[max_patch_num][:, :, 0:3].transpose((1, 0, 2))
                clean_patch = patch_list[max_patch_num][:, :, 3:6].transpose((1, 0, 2))
                img = np.concatenate([noisy_patch, clean_patch], 1)
                cv2.imwrite(os.path.join(dst_path_test,
                                         '{}_{}_img_{:03d}_patch_{:03d}.png'.format(
                                             total_num, scene_name, img_num + 1, max_patch_num + 1)), img)
                img_num_real += 1
                total_num += 1


def prepare_polyu_data(src_path, dst_path, patch_size):
    dst_path = make_dir(os.path.join(dst_path, 'polyu'))
    dst_path_test = os.path.join(dst_path, 'test')
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)

    #  prepare testing data
    print('PolyU test data processing...')
    noisy_paths = glob.glob(os.path.join(src_path, '*Real.JPG'))
    noisy_paths.sort()

    for img_num, noisy_path in enumerate(tqdm(noisy_paths)):

        gt_path = os.path.join(noisy_path.replace('Real.JPG', 'mean.JPG'))
        # pos_list = get_pos_list(os.path.join(noisy_path.replace('_Real.JPG', 'patch_list.txt')))
        noisy = np.array(cv2.imread(noisy_path))
        gt = np.array(cv2.imread(gt_path))
        img = np.concatenate([noisy, gt], 2)
        [h, w, c] = img.shape
        patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=False)
        random.shuffle(patch_list)
        for patch_num in range(len(patch_list[:32])):
            noisy_patch = patch_list[patch_num][:, :, 0:3]
            clean_patch = patch_list[patch_num][:, :, 3:6]
            img = np.concatenate([noisy_patch, clean_patch], 1)
            cv2.imwrite(os.path.join(dst_path_test, ' {}_img_{:03d}_patch_{:03d}.png'.format(
                os.path.join(os.path.basename(noisy_path).replace('_Real.JPG', '')),
                img_num + 1, patch_num + 1)), img)


def prepare_nind_data(src_path, dst_path, patch_size):
    dst_path = make_dir(os.path.join(dst_path, 'nind'))
    dst_path_test = os.path.join(dst_path, 'test')
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)

    scene_paths = glob.glob(os.path.join(src_path, '*'))
    scene_paths.sort()
    # prepare testing data
    print('NIND test data processing...')
    for scene_num, scene_path in enumerate(tqdm(scene_paths), 0):
        scene_name = os.path.basename(scene_path)
        gt_path = glob.glob(os.path.join(scene_path, '*gt.png'))
        noisy_paths = glob.glob(os.path.join(scene_path, '*ISO*'))
        noisy_paths.sort()
        gt = cv2.imread(gt_path[0])
        pos_list = get_pos_list(os.path.join(scene_path, 'patch_list.txt'))
        for img_num in range(len(noisy_paths)):
            noisy = np.array(cv2.imread(noisy_paths[img_num]))
            img = np.concatenate([noisy, gt], 2)
            [h, w, c] = img.shape
            patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=False)
            random.shuffle(patch_list)
            for patch_num in range(len(patch_list[:32])):
                noisy_patch = patch_list[patch_num][:, :, 0:3]
                clean_patch = patch_list[patch_num][:, :, 3:6]
                img = np.concatenate([noisy_patch, clean_patch], 1)
                cv2.imwrite(os.path.join(dst_path_test,
                                         '{}_img_{:03d}_patch_{:03d}.png'.format(scene_name,
                                                                                 img_num + 1,
                                                                                 patch_num + 1)), img)


def prepare_rid2021_data(src_path, dst_path, patch_size):
    dst_path = make_dir(os.path.join(dst_path, 'rid2021'))
    dst_path_test = os.path.join(dst_path, 'test')
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)

    # prepare training data
    print('RID2021 test data processing...')
    noisy_paths = glob.glob(os.path.join(src_path, '*_noisy.jpeg'))
    noisy_paths.sort()
    gt_paths = glob.glob(os.path.join(src_path, '*_gt.jpeg'))
    gt_paths.sort()
    for img_num, noisy_path in enumerate(tqdm(noisy_paths)):
        noisy = np.array(cv2.imread(noisy_path))
        gt = np.array(cv2.imread(noisy_path.replace('_noisy.jpeg', '_gt.jpeg')))
        pos_list = get_pos_list(noisy_path.replace('_noisy.jpeg', '_gt.jpegpatch_list.txt'))
        img = np.concatenate([noisy, gt], 2)
        [h, w, c] = img.shape
        patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, random_crop=False, pos_list=pos_list)
        random.shuffle(patch_list)
        for patch_num in range(len(patch_list[:32])):
            noisy_patch = patch_list[patch_num][:, :, 0:3]
            clean_patch = patch_list[patch_num][:, :, 3:6]
            img = np.concatenate([noisy_patch, clean_patch], 1)
            cv2.imwrite(os.path.join(dst_path_test,
                                     '{}_img_{:03d}_patch_{:03d}.png'.format(
                                         os.path.join(os.path.basename(noisy_path).replace('_noisy.jpeg', '')),
                                         img_num + 1, patch_num + 1)), img)


def main():
    parser = argparse.ArgumentParser(description='PyTorch prepare test data')
    parser.add_argument('--patch_size', type=int, default=256, help='size of cropped image')
    parser.add_argument('--data_set', type=str, default='sidd', help='the dataset to crop')
    parser.add_argument('--src_dir', type=str, default='/home/SENSETIME/yangmingzhuo/Documents/ECCV/split_dataset',
                        help='the path of dataset dir')
    parser.add_argument('--dst_dir', type=str, default='/home/SENSETIME/yangmingzhuo/Documents/ECCV/processed',
                        help='the path of destination dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use default=0')
    opt = parser.parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    patch_size = opt.patch_size

    root_dir = opt.src_dir
    if opt.data_set == 'sidd':
        print("start...")
        print("start...SIDD...")
        sidd_src_path_test = os.path.join(root_dir, 'sidd', 'test')
        prepare_sidd_data(sidd_src_path_test, opt.dst_dir, patch_size)
        print("end...SIDD")
    elif opt.data_set == 'renoir':
        print("start...RENOIR...")
        renoir_src_path_test = os.path.join(root_dir, 'renoir', 'test')
        prepare_renoir_data(renoir_src_path_test, opt.dst_dir, patch_size)
        print("end...RENOIR")
    elif opt.data_set == 'polyu':
        print("start...PolyU...")
        polyu_src_path_test = os.path.join(root_dir, 'polyu', 'test')
        prepare_polyu_data(polyu_src_path_test, opt.dst_dir, patch_size)
        print("end...PolyU")
    elif opt.data_set == 'nind':
        print("start...NIND...")
        nind_src_path_test = os.path.join(root_dir, 'nind', 'test')
        prepare_nind_data(nind_src_path_test, opt.dst_dir, patch_size)
        print("end...NIND")
    elif opt.data_set == 'rid2021':
        print("start...RID2021...")
        nind_src_path_test = os.path.join(root_dir, 'rid2021', 'test')
        prepare_rid2021_data(nind_src_path_test, opt.dst_dir, patch_size)
        print("end...RID2021")
    print('end')


if __name__ == "__main__":
    main()
