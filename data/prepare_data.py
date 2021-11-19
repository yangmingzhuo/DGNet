import h5py
from PIL import Image
import os
import numpy as np
import glob
import random
from scipy.io import loadmat
from tqdm import tqdm
import cv2


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split(full_list, shuffle=False, ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_test = full_list[:offset]
    sublist_train = full_list[offset:]
    print("testing set: ", len(sublist_test), sublist_test)
    print("training set: ", len(sublist_train), sublist_train)
    return sublist_test, sublist_train


def crop_patch(img, img_size=(512, 512), patch_size=(300, 300), stride=300, random_crop=False):
    count = 0
    patch_list = []
    if random_crop:
        crop_num = 100
        pos = [(np.random.randint(0, img_size[0] - patch_size),
                np.random.randint(0, img_size[1] - patch_size))
               for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(0, img_size[1] - patch_size[1], stride) for y in
               range(0, img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt:yt + patch_size[0], xt:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list


def prepare_sidd_data(src_files_test, src_files_train, dst_path_test, dst_path_train, patch_size):
    dst_path_test = os.path.join(dst_path_test, 'sidd_patch_test')
    dst_path_train = os.path.join(dst_path_train, 'sidd_patch_train')
    make_dir(dst_path_test)
    make_dir(dst_path_train)
    count = 0

    noisy_data_mat_file = os.path.join(src_files_test[0], 'ValidationNoisyBlocksSrgb.mat')
    clean_data_mat_file = os.path.join(src_files_test[0], 'ValidationGtBlocksSrgb.mat')
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
            cv2.imwrite(os.path.join(dst_path_test, 'scene_{}_patch_{}.png'.format(image_index + 1, block_index + 1)), img)
            count += 1

    # prepare training data
    count = 0
    print('SIDD train data processing...')
    for src_path in src_files_train:
        file_path = glob.glob(src_path + '*')
        for scene_num, file_name in enumerate(tqdm(file_path), 0):
            if 'SIDD' in file_name:
                gt_imgs = glob.glob(file_name + '/*GT*.PNG')
                gt_imgs.sort()
                noisy_imgs = glob.glob(file_name + '/*NOISY*.PNG')
                noisy_imgs.sort()

                for img_num in range(len(noisy_imgs)):
                    gt = np.array(Image.open(gt_imgs[img_num]))
                    noisy = np.array(Image.open(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, False)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_train, 'scene_{}_img_{}_patch_{}.png'.format(scene_num + 1, img_num + 1, patch_num + 1)), img)
                        count += 1


def prepare_renoir_data(src_files, dst_path_test, dst_path_train, patch_size):
    dst_path_test = os.path.join(dst_path_test, 'renoir_patch_test')
    dst_path_train = os.path.join(dst_path_train, 'renoir_patch_train')
    make_dir(dst_path_test)
    make_dir(dst_path_train)

    count = 0
    for src_path in src_files:
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
                ref = np.array(Image.open(ref_imgs[0])).astype(np.float32)
                full = np.array(Image.open(full_imgs[0])).astype(np.float32)
                gt = (ref + full) / 2
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for img_num in range(len(noisy_imgs)):
                    noisy = np.array(Image.open(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, False)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_train, 'scene_{}_img_{}_patch_{}.png'.format(scene_num + 1, img_num + 1, patch_num + 1)),
                                    img)
                        count += 1

        # prepare testing data
        print('RENOIR test data processing...')
        for scene_num, file_name in enumerate(tqdm(file_path_test), 0):
            if 'RENOIR' in file_name:
                ref_imgs = glob.glob(file_name + '/*Reference.bmp')
                full_imgs = glob.glob(file_name + '/*full.bmp')
                noisy_imgs = glob.glob(file_name + '/*Noisy.bmp')
                noisy_imgs.sort()

                ref = np.array(Image.open(ref_imgs[0])).astype(np.float32)
                full = np.array(Image.open(full_imgs[0])).astype(np.float32)
                gt = (ref + full) / 2
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for img_num in range(len(noisy_imgs)):
                    noisy = np.array(Image.open(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, False)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_test, 'scene_{}_img_{}_patch_{}.png'.format(scene_num + 1, img_num + 1, patch_num + 1)), img)
                        count += 1



def main():
    random.seed(0)
    patch_size = 256
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    root_dir = "/mnt/lustre/share/yangmingzhuo/dataset/"
    src_path_list_1 = [os.path.join(root_dir, "test/SIDD/")]
    src_path_list_2 = [os.path.join(root_dir, "train/SIDD/SIDD_Medium_Srgb/Data/")]
    src_path_list_3 = [os.path.join(root_dir, "train/RENOIR/Mi3_Aligned/"),
                     os.path.join(root_dir, "train/RENOIR/T3i_Aligned/"),
                     os.path.join(root_dir, "train/RENOIR/S90_Aligned/"),
                     ]
    dst_dir = "/mnt/lustre/share/yangmingzhuo/processed"
    make_dir(dst_dir)
    dst_path_test = os.path.join(dst_dir, "test")
    dst_path_train = os.path.join(dst_dir, "train")
    make_dir(dst_path_test)
    make_dir(dst_path_train)
    print("start...")
    print("start...SIDD...")
    prepare_sidd_data(src_path_list_1, src_path_list_2, dst_path_test, dst_path_train, patch_size)
    print("end...SIDD")
    print("start...RENOIR...")
    prepare_renoir_data(src_path_list_3, dst_path_test, dst_path_train, patch_size)
    print("end...RENOIR")
    print('end')


if __name__ == "__main__":
    main()
