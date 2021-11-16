import h5py
from PIL import Image
import os
import numpy as np
import glob
import random
from scipy.io import loadmat
from tqdm import tqdm


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
    create_dir(dst_path_test)
    create_dir(dst_path_train)
    h5py_name_train = os.path.join(dst_path_train, "sidd_train.h5")
    h5py_name_test = os.path.join(dst_path_test, "sidd_val.h5")
    h5f_train = h5py.File(h5py_name_train, 'w')
    h5f_test = h5py.File(h5py_name_test, 'w')
    count = 0

    noisy_data_mat_file = os.path.join(src_files_test[0], 'ValidationNoisyBlocksSrgb.mat')
    clean_data_mat_file = os.path.join(src_files_test[0], 'ValidationGtBlocksSrgb.mat')
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
    clean_data_mat_name = os.path.basename(clean_data_mat_file).replace('.mat', '')
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]
    clean_data_mat = loadmat(clean_data_mat_file)[clean_data_mat_name]

    # prepare training data
    for image_index in tqdm(range(noisy_data_mat.shape[0])):
        print('SIDD processing...' + str(count))
        for block_index in range(noisy_data_mat.shape[1]):
            noisy_image = noisy_data_mat[image_index, block_index, :, :, :]
            noisy_image = np.float32(noisy_image / 255.)
            clean_image = clean_data_mat[image_index, block_index, :, :, :]
            clean_image = np.float32(clean_image / 255.)
            img = np.concatenate([noisy_image, clean_image], 2)
            data = img.copy()
            h5f_test.create_dataset(str(count), shape=(256, 256, 6), data=data)
            count += 1

    # prepare testing data
    count = 0
    for src_path in src_files_train:
        file_path = glob.glob(src_path + '*')
        for file_name in file_path:
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
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f_train.create_dataset(str(count), shape=(patch_size, patch_size, c * 2), data=data)
                        count += 1


    h5f_train.close()
    h5f_test.close()


def prepare_renoir_data(src_files, dst_path_test, dst_path_train, patch_size):
    create_dir(dst_path_test)
    create_dir(dst_path_train)
    h5py_name_train = os.path.join(dst_path_train, "renoir_train.h5")
    h5py_name_test = os.path.join(dst_path_test, "renoir_val.h5")
    h5f_train = h5py.File(h5py_name_train, 'w')
    h5f_test = h5py.File(h5py_name_test, 'w')

    count = 0
    for src_path in src_files:
        file_path = glob.glob(src_path + '*')
        file_path_test, file_path_train = split(file_path)

        # prepare training data
        for file_name in file_path_train:
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
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f_train.create_dataset(str(count), shape=(patch_size, patch_size, c * 2), data=data)
                        count += 1

        # prepare testing data
        for file_name in file_path_test:
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
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, False)
                    for num in range(len(patch_list)):
                        data = patch_list[num].copy()
                        h5f_test.create_dataset(str(count), shape=(300, 300, 6), data=data)
                        count += 1

    h5f_train.close()
    h5f_test.close()


def main():
    random.seed(0)
    patch_size = 300
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    src_path_list_1 = ["./test/SIDD/"]
    src_path_list_2 = ["./train/SIDD/SIDD_Medium_Srgb/Data/"]
    src_path_list_3 = ["./train/RENOIR/Mi3_Aligned/",
                     "./train/RENOIR/T3i_Aligned/",
                     "./train/RENOIR/S90_Aligned/",
                     ]
    dst_path_test = "./test"
    dst_path_train = "./train"

    create_dir(dst_path_test)
    create_dir(dst_path_train)
    print("start...")
    # print("start...SIDD...")
    # prepare_sidd_data(src_path_list_1, src_path_list_2, dst_path_test, dst_path_train, patch_size)
    # print("end...SIDD")
    print("start...RENOIR...")
    prepare_renoir_data(src_path_list_3, dst_path_test, dst_path_train, patch_size)
    print("end...RENOIR")
    print('end')


if __name__ == "__main__":
    main()
