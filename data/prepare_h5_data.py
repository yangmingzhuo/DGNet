import h5py
from PIL import Image
import os
import numpy as np
import glob
from tqdm import tqdm
from data_utils import *


def gen_h5_dataset(src_path):
    print('prepare train img')
    h5py_name = os.path.join(src_path, "train.h5")
    h5f = h5py.File(h5py_name, 'w')
    count = 0
    file_path = glob.glob(os.path.join(src_path, 'train', '*'))
    file_path.sort()
    for i, file_name in enumerate(tqdm(file_path)):
        img = np.array(Image.open(file_name))
        data = img.copy()
        h5f.create_dataset(str(count), shape=(256, 256, 6), data=data)
        count += 1
    print(count)
    h5f.close()

    print('prepare test img')
    h5py_name = os.path.join(src_path, "test.h5")
    h5f = h5py.File(h5py_name, 'w')
    count = 0
    file_path = glob.glob(os.path.join(src_path, 'test', '*'))
    file_path.sort()
    for i, file_name in enumerate(tqdm(file_path)):
        img = np.array(Image.open(file_name))
        data = img.copy()
        h5f.create_dataset(str(count), shape=(256, 256, 6), data=data)
        count += 1
    print(count)
    h5f.close()


def main():
    src_path = "/mnt/lustre/share/yangmingzhuo/processed"

    sidd_src_path = os.path.join(src_path, 'sidd')
    renoir_src_path = os.path.join(src_path, 'renoir')
    nind_src_path = os.path.join(src_path, 'nind')
    rid2021_src_path = os.path.join(src_path, 'rid2021_v2')
    print("start...sidd")
    gen_h5_dataset(sidd_src_path)
    print('end...sidd')
    print("start...renoir")
    gen_h5_dataset(renoir_src_path)
    print('end...renoir')
    print("start...nind")
    gen_h5_dataset(nind_src_path)
    print('end...nind')
    print("start...rid2021")
    gen_h5_dataset(rid2021_src_path)
    print('end...rid2021')


if __name__ == "__main__":
    main()
