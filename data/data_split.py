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

renoir_list = ['Batch_005', 'Batch_009', 'Batch_010', 'Batch_015', 'Batch_019', 'Batch_021', 'Batch_025', 'Batch_031']




def split(full_list, shuffle=False, ratio=0.2):
    sublist_test = []
    sublist_train = []
    for i in full_list:
        if i in renoir_list:
            sublist_test.append(i)
        else:
            sublist_train.append(i)
    return sublist_test, sublist_train


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
        scene_paths = os.listdir(camera_path)
        dst_camera_name = os.path.basename(camera_path)

        # divide the train and test data in random
        scene_path_test, scene_path_train = split(scene_paths, shuffle=True)
        print('test scene:', scene_path_test)
        # prepare training data
        print('RENOIR train data processing...')
        for scene_num, src_scene_name in enumerate(tqdm(scene_path_train), 0):
            shutil.copytree(os.path.join(camera_path, src_scene_name),
                            os.path.join(dst_path_train, str(dst_camera_name) + '_' + str(src_scene_name)))

        # prepare testing data
        print('RENOIR test data processing...')
        for scene_num, src_scene_name in enumerate(tqdm(scene_path_test), 0):
            shutil.copytree(os.path.join(camera_path, src_scene_name),
                            os.path.join(dst_path_test, str(dst_camera_name) + '_' + str(src_scene_name)))


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

    real_file_paths = glob.glob(os.path.join(src_path, '*Real.JPG'))
    real_file_paths.sort()

    # divide the train and test data in random
    scene_paths_test, scene_paths_train = split(real_file_paths, shuffle=True)

    # prepare training data
    print('PolyU train data processing...')
    for scene_num, scene_path in enumerate(tqdm(scene_paths_train), 0):
        scene_name = os.path.basename(scene_path)
        shutil.copy(scene_path, os.path.join(dst_path_train, scene_name))
        scene_gt_path = os.path.join(scene_path.replace('Real', 'mean'))
        scene_name = os.path.basename(scene_gt_path)
        shutil.copy(scene_gt_path, os.path.join(dst_path_train, scene_name))

    # prepare testing data
    print('PolyU test data processing...')
    for scene_num, scene_path in enumerate(tqdm(scene_paths_test), 0):
        scene_name = os.path.basename(scene_path)
        shutil.copy(scene_path, os.path.join(dst_path_test, scene_name))
        scene_gt_path = os.path.join(scene_path.replace('Real', 'mean'))
        scene_name = os.path.basename(scene_gt_path)
        shutil.copy(scene_gt_path, os.path.join(dst_path_test, scene_name))


def prepare_nind_data(src_path, dst_path):
    dst_path = make_dir(os.path.join(dst_path, 'nind'))
    dst_path_test = os.path.join(dst_path, 'test')
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)
    make_dir(dst_path_train)
    for camera, camera_path in enumerate(src_path, 0):
        scene_paths = os.listdir(camera_path)
        dst_camera_name = os.path.basename(camera_path)

        # divide the train and test data in random
        scene_path_test, scene_path_train = split(scene_paths)

        # prepare training data
        print('NIND train data processing...')
        for scene_num, src_scene_name in enumerate(tqdm(scene_path_train), 0):
            shutil.copytree(os.path.join(camera_path, src_scene_name),
                            os.path.join(dst_path_train, str(dst_camera_name) + '_' + str(src_scene_name)))

        # prepare testing data
        print('NIND test data processing...')
        for scene_num, src_scene_name in enumerate(tqdm(scene_path_test), 0):
            shutil.copytree(os.path.join(camera_path, src_scene_name),
                            os.path.join(dst_path_test, str(dst_camera_name) + '_' + str(src_scene_name)))


def prepare_rid2021_data(src_path, dst_path):
    dst_path = make_dir(os.path.join(dst_path, 'rid2021'))
    dst_path_test = os.path.join(dst_path, 'test')
    dst_path_train = os.path.join(dst_path, 'train')
    if os.path.exists(dst_path_train):
        shutil.rmtree(dst_path_train)
    if os.path.exists(dst_path_test):
        shutil.rmtree(dst_path_test)
    make_dir(dst_path_test)
    make_dir(dst_path_train)

    real_file_paths = glob.glob(os.path.join(src_path, 'original', '*.jpeg'))
    real_file_paths.sort()

    # divide the train and test data in random
    scene_paths_test, scene_paths_train = split(real_file_paths, shuffle=True)

    # prepare training data
    print('RID2021 train data processing...')
    for scene_num, scene_path in enumerate(tqdm(scene_paths_train), 0):
        scene_name = os.path.basename(scene_path)
        name = scene_name.split('.')[0]
        shutil.copy(scene_path, os.path.join(dst_path_train, name + '_noisy.jpeg'))
        scene_gt_path = os.path.join(scene_path.replace('original', 'denoised'))
        scene_name = os.path.basename(scene_gt_path)
        name = scene_name.split('.')[0]
        shutil.copy(scene_gt_path, os.path.join(dst_path_train, name + '_gt.jpeg'))

    # prepare testing data
    print('RID2021 test data processing...')
    for scene_num, scene_path in enumerate(tqdm(scene_paths_test), 0):
        scene_name = os.path.basename(scene_path)
        name = scene_name.split('.')[0]
        shutil.copy(scene_path, os.path.join(dst_path_test, name + '_noisy.jpeg'))
        scene_gt_path = os.path.join(scene_path.replace('original', 'denoised'))
        scene_name = os.path.basename(scene_gt_path)
        name = scene_name.split('.')[0]
        shutil.copy(scene_gt_path, os.path.join(dst_path_test, name + '_gt.jpeg'))


def main():
    parser = argparse.ArgumentParser(description='PyTorch data split')
    parser.add_argument('--data_set', type=str, default='sidd', help='the dataset to crop')
    parser.add_argument('--data_set_dir', type=str, default='/home/SENSETIME/yangmingzhuo/Documents/ECCV/data',
                        help='the dataset dir')
    parser.add_argument('--dst_dir', type=str, default='/home/SENSETIME/yangmingzhuo/Documents/ECCV/split_dataset/',
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
    elif opt.data_set == 'nind':
        print("start...NIND...")
        nind_src_path_list = [os.path.join(root_dir, 'NIND', 'C500D_8bit'),
                              os.path.join(root_dir, 'NIND', 'XT1_8bit'),
                              os.path.join(root_dir, 'NIND', 'XT1_16bit'), ]
        prepare_nind_data(nind_src_path_list, opt.dst_dir)
        print("end...NIND")
    elif opt.data_set == 'rid2021':
        print("start...RID2021...")
        rid2021_src_path = os.path.join(root_dir, 'RID2021')
        prepare_rid2021_data(rid2021_src_path, opt.dst_dir)
        print("end...RID2021")
    print('end')


if __name__ == "__main__":
    main()
