import os
import numpy as np
import random


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


def crop_patch(img, img_size=(512, 512), patch_size=(256, 256), stride=256, random_crop=False, crop_num=100,
               pos_list=[]):
    count = 0
    patch_list = []
    if random_crop:
        pos = [(np.random.randint(0, img_size[1] - patch_size[1]), np.random.randint(0, img_size[0] - patch_size[0]))
               for i in range(crop_num)]
    elif pos_list:
        pos = pos_list
    else:
        pos = [(x, y) for x in range(0, img_size[1] - patch_size[1], stride) for y in
               range(0, img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt:yt + patch_size[0], xt:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list


def get_pos_list(txt_path, max_num=999):
    pos_list = []
    with open(txt_path, "r") as f:
        f.readline()
        for i in range(max_num):
            line = f.readline()
            if line:
                line_data = line.split('\t')
                pos = (int(line_data[0]), int(line_data[1]))
                pos_list.append(pos)
            else:
                break
    return pos_list

def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real

