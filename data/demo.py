import os
import glob

path = '/home/SENSETIME/yangmingzhuo/Documents/ECCV/processed/renoir/test'
paths = glob.glob('/home/SENSETIME/yangmingzhuo/Documents/ECCV/processed/renoir/test/*')
for p in paths:
    p.replace('Mi3_Aligned_Batch_001_img_001_patch_001')
