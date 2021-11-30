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

# pre-selected train data
polyu_train_list = ['Canon5D2_bag', 'Canon5D2_bicyc', 'Canon5D2_circu', 'Canon5D2_desk', 'Canon5D2_fruit',
                    'Canon5D2_mouse', 'Canon5D2_plug', 'Canon5D2_robot', 'Canon5D2_toy',

                    'Canon80D_compr', 'Canon80D_corne', 'Canon80D_GO', 'Canon80D_print',

                    'Canon600_book', 'Canon600_toy',

                    'Nikon800_bulle', 'Nikon800_class', 'Nikon800_desch', 'Nikon800_door', 'Nikon800_flowe',
                    'Nikon800_map', 'Nikon800_photo', 'Nikon800_plaso', 'Nikon800_stair', 'Nikon800_wall',

                    'SonyA7II_class', 'SonyA7II_compu', 'SonyA7II_door', 'SonyA7II_plant', 'SonyA7II_stair',
                    'SonyA7II_toy', 'SonyA7II_water']

polyu_test_list = ['Canon5D2_chair', 'Canon5D2_recie', 'Canon80D_ball', 'Canon600_water', 'Nikon800_desk',
                   'Nikon800_plant', 'Nikon800_carbi', 'SonyA7II_book']


def prepare_sidd_data(src_path_test, src_path_train, dst_path):
    dst_path = make_dir(os.path.join(dst_path, 'sidd'))
    dst_path_train = os.path.join(dst_path, 'train')
    dst_path_test = os.path.join(dst_path, 'test')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)

    shutil.copytree(src_path_train, dst_path_train)
    shutil.copytree(src_path_test, dst_path_test)


def prepare_renoir_data(src_path, dst_path):
    dst_path = make_dir(os.path.join(dst_path, 'renoir'))
    dst_path_test = os.path.join(dst_path, 'test')
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)
    make_dir(dst_path_train)

    for camera, camera_path in enumerate(src_path, 0):
        file_path = os.listdir(camera_path)
        dst_camera_name = os.path.basename(camera_path)

        # divide the train and test data in random
        file_path_test, file_path_train = split(file_path)

        # prepare training data
        print('RENOIR train data processing...')
        for scene_num, src_scene_name in enumerate(tqdm(file_path_train), 0):
            shutil.copytree(os.path.join(camera_path, src_scene_name), os.path.join(dst_path_train, str(dst_camera_name) + '_' + str(src_scene_name)))

        # prepare testing data
        print('RENOIR test data processing...')
        for scene_num, src_scene_name in enumerate(tqdm(file_path_test), 0):
            shutil.copytree(os.path.join(camera_path, src_scene_name), os.path.join(dst_path_test, str(dst_camera_name) + '_' + str(src_scene_name)))


def prepare_polyu_data(src_path, dst_path):
    dst_path = make_dir(os.path.join(dst_path, 'polyu'))
    dst_path_test = os.path.join(dst_path, 'test')
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)
    make_dir(dst_path_train)

    print('PolyU train data processing...')
    for scene_num, scene_path in enumerate(tqdm(polyu_train_list), 0):
        noisy_paths = glob.glob(os.path.join(src_path, scene_path, '*.JPG'))
        os.makedirs(os.path.join(dst_path_train, scene_path))
        noisy_paths.sort()

        # generate ground truth
        gt = np.array(cv2.imread(noisy_paths[0])).astype(np.float32)
        for i in range(1, len(noisy_paths)):
            gt += np.array(cv2.imread(noisy_paths[i])).astype(np.float32)
        gt = gt / len(noisy_paths)
        gt = np.clip(gt, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_path_train, scene_path, 'mean.png'), gt)

        # select 4 images
        noisy_imgs = [noisy_paths[0], noisy_paths[33], noisy_paths[66], noisy_paths[99]]
        for i in range(len(noisy_imgs)):
            shutil.copy(noisy_imgs[i], os.path.join(dst_path_train, scene_path))

    print('PolyU test data processing...')
    for scene_num, scene_path in enumerate(tqdm(polyu_test_list), 0):
        noisy_paths = glob.glob(os.path.join(src_path, scene_path) + '/*.JPG')
        os.makedirs(os.path.join(dst_path_test, scene_path))
        noisy_paths.sort()

        # generate ground truth
        gt = np.array(cv2.imread(noisy_paths[0])).astype(np.float32)
        for i in range(1, len(noisy_paths)):
            gt += np.array(cv2.imread(noisy_paths[i])).astype(np.float32)
        gt = gt / len(noisy_paths)
        gt = np.clip(gt, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_path_test, scene_path, 'mean.png'), gt)

        # select 4 images
        noisy_imgs = [noisy_paths[0], noisy_paths[33], noisy_paths[66], noisy_paths[99]]
        for i in range(len(noisy_imgs)):
            shutil.copy(noisy_imgs[i], os.path.join(dst_path_test, scene_path))


def main():
    parser = argparse.ArgumentParser(description='PyTorch data split')
    parser.add_argument('--data_set', type=str, default='sidd', help='the dataset to crop')
    parser.add_argument('--data_set_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/dataset/',
                        help='the dataset dir')
    parser.add_argument('--dst_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/split_dataset/',
                        help='the destination dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use default=0')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    root_dir = opt.data_set_dir
    if opt.data_set == 'sidd':
        print("start...")
        print("start...SIDD...")
        sidd_src_path_test = os.path.join(root_dir, 'SIDD', 'test')
        sidd_src_path_train = os.path.join(root_dir, 'SIDD', 'SIDD_Medium_Srgb', 'Data')
        prepare_sidd_data(sidd_src_path_test, sidd_src_path_train, opt.dst_dir)
        print("end...SIDD")
    elif opt.data_set == 'renoir':
        print("start...RENOIR...")
        renoir_src_path_list = [os.path.join(root_dir, 'RENOIR', 'Mi3_Aligned'),
                                os.path.join(root_dir, 'RENOIR', 'T3i_Aligned'),
                                os.path.join(root_dir, 'RENOIR', 'S90_Aligned'), ]
        prepare_renoir_data(renoir_src_path_list, opt.dst_dir)
        print("end...RENOIR")
    elif opt.data_set == 'polyu':
        print("start...PolyU...")
        polyu_src_path = os.path.join(root_dir, 'PolyU')
        prepare_polyu_data(polyu_src_path, opt.dst_dir)
        print("end...PolyU")
    print('end')


if __name__ == "__main__":
    main()
