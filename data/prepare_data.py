import h5py
from PIL import Image
import os
import numpy as np
import glob
import random


def create_dir(path):
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


def crop_patch(img, img_size=(512, 512), patch_size=(150, 150), stride=150, random_crop=False):
    count = 0
    patch_list = []
    if random_crop:
        crop_num = 100
        pos = [(np.random.randint(patch_size, img_size[0] - patch_size),
                np.random.randint(patch_size, img_size[1] - patch_size))
               for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
               range(patch_size[0], img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list


def prepare_data(src_files, dst_path_test, dst_path_train):
    create_dir(dst_path_test)
    create_dir(dst_path_train)
    h5py_name_train = os.path.join(dst_path_train, "train.h5")
    h5py_name_test = os.path.join(dst_path_test, "val.h5")
    h5f_train = h5py.File(h5py_name_train, 'w')
    h5f_test = h5py.File(h5py_name_test, 'w')
    count = 0
    for src_path in src_files:
        file_path = glob.glob(src_path + '*')
        file_path_test, file_path_train = split(file_path)
        for file_name in file_path_train:
            if 'SIDD' in file_name:
                gt_imgs = glob.glob(file_name + '/*GT*.PNG')
                gt_imgs.sort()
                noisy_imgs = glob.glob(file_name + '/*NOISY*.PNG')
                noisy_imgs.sort()
                print('SIDD processing...' + str(count))
                for i in range(len(noisy_imgs)):
                    gt = np.array(Image.open(gt_imgs[i]))
                    noisy = np.array(Image.open(noisy_imgs[i]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (150, 150), 150, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f_train.create_dataset(str(count), shape=(300, 300, 6), data=data)
                        count += 1

            if 'RENOIR' in file_name:
                ref_imgs = glob.glob(file_name + '/*Reference.bmp')
                full_imgs = glob.glob(file_name + '/*full.bmp')
                noisy_imgs = glob.glob(file_name + '/*Noisy.bmp')
                noisy_imgs.sort()

                print('RENOIR processing...' + str(count) + file_name)
                ref = np.array(Image.open(ref_imgs[0])).astype(np.float32)
                full = np.array(Image.open(full_imgs[0])).astype(np.float32)
                gt = (ref + full) / 2
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for i in range(len(noisy_imgs)):
                    noisy = np.array(Image.open(noisy_imgs[i]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (150, 150), 150, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f_train.create_dataset(str(count), shape=(300, 300, 6), data=data)
                        count += 1

        for file_name in file_path_test:
            if 'SIDD' in file_name:
                gt_imgs = glob.glob(file_name + '/*GT*.PNG')
                gt_imgs.sort()
                noisy_imgs = glob.glob(file_name + '/*NOISY*.PNG')
                noisy_imgs.sort()
                print('SIDD processing...' + str(count))
                for i in range(len(noisy_imgs)):
                    gt = np.array(Image.open(gt_imgs[i]))
                    noisy = np.array(Image.open(noisy_imgs[i]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (150, 150), 150, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f_test.create_dataset(str(count), shape=(300, 300, 6), data=data)
                        count += 1

            if 'RENOIR' in file_name:
                ref_imgs = glob.glob(file_name + '/*Reference.bmp')
                full_imgs = glob.glob(file_name + '/*full.bmp')
                noisy_imgs = glob.glob(file_name + '/*Noisy.bmp')
                noisy_imgs.sort()

                print('RENOIR processing...' + str(count) + file_name)
                ref = np.array(Image.open(ref_imgs[0])).astype(np.float32)
                full = np.array(Image.open(full_imgs[0])).astype(np.float32)
                gt = (ref + full) / 2
                gt = np.clip(gt, 0, 255).astype(np.uint8)
                for i in range(len(noisy_imgs)):
                    noisy = np.array(Image.open(noisy_imgs[i]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (256, 256), 256, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f_test.create_dataset(str(count), shape=(300, 300, 6), data=data)
                        count += 1

    h5f_train.close()
    h5f_test.close()


if __name__ == "__main__":
    random.seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    src_path_list_1 = ["./train/SIDD/SIDD_Medium_Srgb/Data/"]
    src_path_list_2 = ["./train/RENOIR/Mi3_Aligned/",
                     "./train/RENOIR/T3i_Aligned/",
                     "./train/RENOIR/S90_Aligned/",
                     ]
    dst_path_test = "./test"
    dst_path_train = "./train"

    create_dir(dst_path_test)
    create_dir(dst_path_train)
    print("start...")
    # print("start...SIDD_RENOIR_h5...")
    # prepare_data(src_path_list, dst_path)
    # print("end...SIDD_RENOIR_h5")
    # print("start...SIDD_h5...")
    # prepare_data(src_path_list_1, dst_path_test, dst_path_train)
    # print("end...SIDD_h5")
    print("start...RENOIR_h5...")
    prepare_data(src_path_list_2, dst_path_test, dst_path_train)
    print("end...RENOIR_h5")
    print('end')
