#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import os
import glob


def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


def cv_imwrite(filePathName, img):
    cv2.imencode(".jpg", img)[1].tofile(filePathName)


# draw line and write number, add txt
def draw_addTxt(img_path, patch_size=256, save_flag=True,
                jpg_name="processed", fontScale=2, txt_name='patch_list.txt'):
    process_flag = True
    # print('img_path={}'.format(img_path))
    img = cv_imread(img_path)
    # print(img)
    # print('img.shape={}'.format(img.shape))
    height, width, ch = img.shape

    # draw lines
    for width_index in range(patch_size, width, patch_size):
        ptStart = (width_index, 0)
        ptEnd = (width_index, height)
        # print('ptStart={}, ptEnd={}'.format(ptStart, ptEnd))
        cv2.line(img, ptStart, ptEnd, color=(0, 255, 0), thickness=2)
    # print('*'*80)
    for height_index in range(patch_size, height, patch_size):
        ptStart = (0, height_index)
        ptEnd = (width, height_index)
        # print('ptStart={}, ptEnd={}'.format(ptStart, ptEnd))
        cv2.line(img, ptStart, ptEnd, color=(0, 255, 0), thickness=2)

    # draw number
    num = 0
    for height_index in range(0, height - patch_size, patch_size):
        for width_index in range(0, width - patch_size, patch_size):
            num_x = width_index + int(patch_size / 2)
            num_y = height_index + int(patch_size / 2)
            # draw number
            cv2.putText(img, "{}".format(num), (num_x, num_y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, color=(0, 0, 255), thickness=2)
            num += 1
    dst_path, file_name = os.path.split(img_path)
    if process_flag:
        txt_str = ''
        count = 0
        for height_index in range(0, height - patch_size, patch_size):
            for width_index in range(0, width - patch_size, patch_size):
                num_x = width_index
                num_y = height_index
                # add txt
                txt_str += '{}\t{}\t{}\t{}\t{}\n'.format(num_x, num_y, patch_size,
                                                         patch_size, count)
                count += 1
        # save txt
        # print('txt_str=\n{}'.format(txt_str))
        txt_path = os.path.join(dst_path, txt_name)
        title_str = 'x\ty\tw\th\tindex\n'
        with open(txt_path, 'w') as f:
            f.write(title_str)
            f.write(txt_str)

        f.close()
        process_flag = False

    if save_flag:
        jpg_path = os.path.join(dst_path, '{}_{}.jpg'.format(file_name, jpg_name))
        print('jpg_path={}'.format(jpg_path))
        cv_imwrite(jpg_path, img)
    print('dst_path={}, success!'.format(dst_path))


if __name__ == "__main__":
    # file_path = glob.glob('/home/SENSETIME/yangmingzhuo/Documents/ECCV/split_dataset/renoir/test/' + '*')
    # # prepare training data
    # print('RENOIR test data processing...')
    # print(file_path)
    # for scene_num, file_name in enumerate(file_path, 0):
    #     ref_imgs = glob.glob(os.path.join(file_name, '*Reference.bmp'))
    #     draw_addTxt(ref_imgs[0])
    #     print('#' * 40)
    #
    # file_path = glob.glob('/home/SENSETIME/yangmingzhuo/Documents/ECCV/split_dataset/polyu/test/' + '*')
    # # prepare training data
    # print('polyu test data processing...')
    # print(file_path)
    # for scene_num, file_name in enumerate(file_path, 0):
    #     ref_imgs = glob.glob(os.path.join(file_name, 'mean.png'))
    #     draw_addTxt(ref_imgs[0])
    #     print('#' * 40)

    # file_path = glob.glob('/home/SENSETIME/yangmingzhuo/Documents/ECCV/split_dataset/nind/test/' + '*')
    # # prepare training data
    # print('nind test data processing...')
    # print(file_path)
    # for scene_num, file_name in enumerate(file_path, 0):
    #     ref_imgs = glob.glob(os.path.join(file_name, '*_gt.png'))
    #     draw_addTxt(ref_imgs[0])
    #     print('#' * 40)

    file_path = glob.glob('/home/SENSETIME/yangmingzhuo/Documents/ECCV/split_dataset/rid2021/test/' + '*_gt.jpeg')
    # prepare training data
    print('nind test data processing...')
    print(file_path)
    for scene_num, file_name in enumerate(file_path, 0):
        draw_addTxt(file_name, txt_name=os.path.basename(file_name) + 'patch_list.txt')
        print('#' * 40)

    # print("Draw and add txt Successfully!")
