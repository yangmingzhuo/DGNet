import argparse
import os
import numpy as np
import glob
import random
import cv2
from scipy.io import loadmat
from tqdm import tqdm


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def split(full_list, shuffle=False, ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_test = full_list[:offset]
    sublist_train = full_list[offset:]
    return sublist_test, sublist_train


def crop_patch(img, img_size=(512, 512), patch_size=(256, 256), stride=256, random_crop=False, crop_num=100):
    count = 0
    patch_list = []
    if random_crop:
        pos = [(np.random.randint(0, img_size[1] - patch_size[1]), np.random.randint(0, img_size[0] - patch_size[0])) for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(0, img_size[1] - patch_size[1], stride) for y in
               range(0, img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt:yt + patch_size[0], xt:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list


def prepare_sidd_data(src_files_test, src_files_train, dst_path_test, dst_path_train, patch_size, rand, rand_num_train=300):
    dst_path_test = make_dir(os.path.join(dst_path_test, 'sidd_patch_test'))
    dst_path_train = make_dir(os.path.join(dst_path_train, 'sidd_patch_train'))

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
            cv2.imwrite(os.path.join(dst_path_test, 'scene_{:03d}_patch_{:03d}.png'.format(image_index + 1, block_index + 1)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # prepare training data
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
                    gt = np.array(cv2.imread(gt_imgs[img_num]))
                    noisy = np.array(cv2.imread(noisy_imgs[img_num]))
                    img = np.concatenate([noisy, gt], 2)
                    [h, w, c] = img.shape
                    patch_list = crop_patch(img, (h, w), (patch_size, patch_size), patch_size, rand, rand_num_train)
                    for patch_num in range(len(patch_list)):
                        noisy_patch = patch_list[patch_num][:, :, 0:3]
                        clean_patch = patch_list[patch_num][:, :, 3:6]
                        img = np.concatenate([noisy_patch, clean_patch], 1)
                        cv2.imwrite(os.path.join(dst_path_train, 'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1, img_num + 1, patch_num + 1)), img)


def prepare_renoir_data(src_files, dst_path_test, dst_path_train, patch_size, rand, rand_num_train=300, rand_num_test=32):
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
                        cv2.imwrite(os.path.join(dst_path_train, 'camera_{:03d}_scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(camera + 1, scene_num + 1, img_num + 1, patch_num + 1)), img)

        # prepare testing data
        print('RENOIR test data processing...')
        for scene_num, file_name in enumerate(tqdm(file_path_test), 0):
            print(file_name)
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
                        cv2.imwrite(os.path.join(dst_path_test, 'camera_{:03d}_scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(camera + 1, scene_num + 1, img_num + 1, patch_num + 1)), img)


def prepare_polyu_data(src_files, dst_path_test, dst_path_train, patch_size, rand, rand_num_train=300, rand_num_test=32):
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
                        cv2.imwrite(os.path.join(dst_path_train, 'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1, img_num + 1, patch_num + 1)), img)

        print('PolyU train data processing...')
        for scene_num, file_name in enumerate(tqdm(file_path_test), 0):
            if 'PolyU' in file_name:
                noisy_paths = glob.glob(file_name + '/*.JPG')
                noisy_imgs = [noisy_paths[0], noisy_paths[33], noisy_paths[66],  noisy_paths[99]]
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
                        cv2.imwrite(os.path.join(dst_path_test, 'scene_{:03d}_img_{:03d}_patch_{:03d}.png'.format(scene_num + 1, img_num + 1, patch_num + 1)), img)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped HR image')
    parser.add_argument('--data_set', type=str, default='sidd', help='the dataset to crop')
    parser.add_argument('--data_set_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/dataset/',
                        help='the dataset dir')
    parser.add_argument('--dst_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/',
                        help='the destination dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
    parser.add_argument('--random', action='store_true', help='whether to randomly crop images')
    parser.add_argument('--rand_num_train', type=int, default=300, help='training patch number to randomly crop images')
    parser.add_argument('--rand_num_test', type=int, default=32, help='testing patch number to randomly crop images')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    patch_size = opt.patch_size

    root_dir = opt.data_set_dir
    sidd_src_path_list_test = [os.path.join(root_dir, "test/SIDD/")]
    sidd_src_path_list_train = [os.path.join(root_dir, "train/SIDD/SIDD_Medium_Srgb/Data/")]
    renoir_src_path_list = [os.path.join(root_dir, "train/RENOIR/Mi3_Aligned/"),
                            os.path.join(root_dir, "train/RENOIR/T3i_Aligned/"),
                            os.path.join(root_dir, "train/RENOIR/S90_Aligned/"),
                            ]
    polyu_src_path_list = [os.path.join(root_dir, "train/PolyUDataset/")]
    if(opt.random):
        dst_dir = make_dir(os.path.join(opt.dst_dir, 'random_processed'))
    else:
        dst_dir = make_dir(os.path.join(opt.dst_dir, 'processed'))
    dst_path_test = make_dir(os.path.join(dst_dir, "test"))
    dst_path_train = make_dir(os.path.join(dst_dir, "train"))
    if opt.data_set == 'sidd':
        print("start...")
        print("start...SIDD...")
        prepare_sidd_data(sidd_src_path_list_test, sidd_src_path_list_train, dst_path_test, dst_path_train, patch_size, opt.random, opt.rand_num_train)
        print("end...SIDD")
    elif opt.data_set == 'renoir':
        print("start...RENOIR...")
        prepare_renoir_data(renoir_src_path_list, dst_path_test, dst_path_train, patch_size, opt.random, opt.rand_num_train, opt.rand_num_test)
        print("end...RENOIR")
    elif opt.data_set == 'polyu':
        print("start...PolyU...")
        prepare_polyu_data(polyu_src_path_list, dst_path_test, dst_path_train, patch_size, opt.random, opt.rand_num_train, opt.rand_num_test)
        print("end...PolyU")
    print('end')


if __name__ == "__main__":
    main()
