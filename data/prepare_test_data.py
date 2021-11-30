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

def prepare_sidd_data(src_path, dst_path, patch_size):
    dst_path = make_dir(os.path.join(dst_path, 'sidd'))
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    make_dir(dst_path_train)

    noisy_data_mat_file = os.path.join(src_files_test[0], 'ValidationNoisyBlocksSrgb.matlab_evaluate')
    clean_data_mat_file = os.path.join(src_files_test[0], 'ValidationGtBlocksSrgb.matlab_evaluate')
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.matlab_evaluate', '')
    clean_data_mat_name = os.path.basename(clean_data_mat_file).replace('.matlab_evaluate', '')
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


def prepare_renoir_data(src_files, dst_path_test, dst_path_train, patch_size, rand, rand_num_train=300,
                        rand_num_test=32):
    dst_path_test = make_dir(os.path.join(dst_path_test, 'renoir_patch_test'))
    dst_path_train = make_dir(os.path.join(dst_path_train, 'renoir_patch_train'))

    for camera, src_path in enumerate(src_files, 0):
        file_path = glob.glob(src_path + '*')
        file_path_test, file_path_train = split(file_path)
        # prepare training data
        print('RENOIR train data processing...')
        for scene_num, file_name in enumerate(tqdm(file_path_train), 0):
            if 'RENOIR' in file_name:
                ref_imgs = glob.glob(file_name + '/*Reference.bmp')
                full_imgs = glob.glob(file_name + '/*full.bmp')
                noisy_imgs = glob.glob(file_name + '/*Noisy.bmp')
                noisy_imgs.sort()
                ref = np.array(cv2.imread(ref_imgs[0])).astype(np.float32)
                full = np.array(cv2.imread(full_imgs[0])).astype(np.float32)
                gt = (ref + full) / 2
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for img_num in range(len(noisy_imgs)):
                    noisy = np.array(cv2.imread(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, rand, rand_num_train)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_train,
                                                 'camera_{:03d}_scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(
                                                     camera + 1, scene_num + 1, img_num + 1, patch_num + 1)), img)

        # prepare testing data
        print('RENOIR test data processing...')
        for scene_num, file_name in enumerate(tqdm(file_path_test), 0):
            if 'RENOIR' in file_name:
                ref_imgs = glob.glob(file_name + '/*Reference.bmp')
                full_imgs = glob.glob(file_name + '/*full.bmp')
                noisy_imgs = glob.glob(file_name + '/*Noisy.bmp')
                noisy_imgs.sort()
                ref = np.array(cv2.imread(ref_imgs[0])).astype(np.float32)
                full = np.array(cv2.imread(full_imgs[0])).astype(np.float32)
                gt = (ref + full) / 2
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for img_num in range(len(noisy_imgs)):
                    noisy = np.array(cv2.imread(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, rand, rand_num_test)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_test,
                                                 'camera_{:03d}_scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(
                                                     camera + 1, scene_num + 1, img_num + 1, patch_num + 1)), img)


def prepare_polyu_data(src_files, dst_path_test, dst_path_train, patch_size, rand, rand_num_train=300,
                       rand_num_test=32):
    dst_path_test = make_dir(os.path.join(dst_path_test, 'polyu_patch_test'))
    dst_path_train = make_dir(os.path.join(dst_path_train, 'polyu_patch_train'))

    for src_path in src_files:
        file_path = glob.glob(src_path + '*')
        file_path_test, file_path_train = split(file_path)
        # prepare training data
        print('PolyU train data processing...')
        for scene_num, file_name in enumerate(tqdm(file_path_train), 0):
            if 'PolyU' in file_name:
                noisy_paths = glob.glob(file_name + '/*.JPG')
                noisy_imgs = [noisy_paths[0], noisy_paths[33], noisy_paths[66], noisy_paths[99]]
                gt = np.array(cv2.imread(noisy_imgs[0])).astype(np.float32)
                for i in range(1, len(noisy_imgs)):
                    gt += np.array(cv2.imread(noisy_imgs[i])).astype(np.float32)
                gt = gt / 100
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for img_num in range(len(noisy_imgs)):
                    noisy = np.array(cv2.imread(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, rand, rand_num_train)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_train,
                                                 'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1,
                                                                                                   img_num + 1,
                                                                                                   patch_num + 1)), img)

        print('PolyU train data processing...')
        for scene_num, file_name in enumerate(tqdm(file_path_test), 0):
            if 'PolyU' in file_name:
                noisy_paths = glob.glob(file_name + '/*.JPG')
                noisy_imgs = [noisy_paths[0], noisy_paths[33], noisy_paths[66], noisy_paths[99]]
                gt = np.array(cv2.imread(noisy_imgs[0])).astype(np.float32)
                for i in range(1, len(noisy_imgs)):
                    gt += np.array(cv2.imread(noisy_imgs[i])).astype(np.float32)
                gt = gt / 100
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for img_num in range(len(noisy_imgs)):
                    noisy = np.array(cv2.imread(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, rand, rand_num_test)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_test,
                                                 'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1,
                                                                                                   img_num + 1,
                                                                                                   patch_num + 1)), img)


def main():
    parser = argparse.ArgumentParser(description='PyTorch prepare test data')
    parser.add_argument('--patch_size', type=int, default=256, help='size of cropped image')
    parser.add_argument('--data_set', type=str, default='sidd', help='the dataset to crop')
    parser.add_argument('--src_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/split_dataset/',
                        help='the path of dataset dir')
    parser.add_argument('--dst_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/processed/',
                        help='the path of destination dir')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    patch_size = opt.patch_size

    root_dir = opt.src_dir
    sidd_src_path_test = os.path.join(root_dir, 'sidd', 'test')
    renoir_src_path_test = os.path.join(root_dir, 'renoir', 'test')
    polyu_src_path_test = os.path.join(root_dir, 'polyu', 'test')
    if opt.data_set == 'sidd':
        print("start...")
        print("start...SIDD...")
        prepare_sidd_data(sidd_src_path_test, opt.dst_dir, patch_size)
        print("end...SIDD")
    elif opt.data_set == 'renoir':
        print("start...RENOIR...")
        prepare_renoir_data(renoir_src_path_test, opt.dst_dir, patch_size)
        print("end...RENOIR")
    elif opt.data_set == 'polyu':
        print("start...PolyU...")
        prepare_polyu_data(polyu_src_path_test, opt.dst_dir, patch_size)
        print("end...PolyU")
    print('end')


if __name__ == "__main__":
    main()
