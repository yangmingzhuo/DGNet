import os
import glob
import shutil
import cv2
from PIL import Image
import numpy as np

XT1_8bit_list = ['droid,200,800,3200,6400',
                 'gnome,200,800,1600,6400',
                 'Ottignies,200,640,3200,6400',
                 'MuseeL-turtle,200,800,1250,6400',
                 'MuseeL-centrifuge,200,800,2000,6400',
                 'MuseeL-shell,200,400,800,6400',
                 'MuseeL-coral,200,800,5000,6400',
                 'MuseeL-head,200,640,3200,6400',
                 'MuseeL-heads,200,400,3200,6400',
                 'MuseeL-mask,200,640,4000,6400',
                 'MuseeL-pig,200,500,2000,6400',
                 'MuseeL-inspiredlotus,200,640,2500,6400',
                 'MuseeL-pinklotus,200,800,4000,6400',
                 'MuseeL-Armlessness,200,800,2000,6400',
                 'MuseeL-byMarcGroessens,200,400,3200,6400',
                 'MuseeL-Moschophore,200,500,4000,6400',
                 'MuseeL-AraPacis,200,800,4000,6400',
                 'MuseeL-stele,200,640,1000,6400',
                 'MuseeL-cross,200,500,4000,6400',
                 'MuseeL-fuite,200,320,4000,6400',
                 'MuseeL-RGB,200,250,1000,6400',
                 'MuseeL-Vincent,200,400,2500,6400',
                 'MuseeL-ambon,200,640,2500,6400',
                 'MuseeL-ram,200,800,1000,6400',
                 'MuseeL-pedestal,200,1250,5000,6400',
                 'MuseeL-theatre,200,400,2500,6400',
                 'MuseeL-text,200,400,5000,6400',
                 'MuseeL-painting,200,800,3200,6400',
                 'MuseeL-yombe,200,640,3200,6400',
                 'MuseeL-hanging,200,500,4000,6400',
                 'MuseeL-snakeAndMask,200,2000,6400,H1',
                 'MuseeL-coral2,200,6400,H1,H2',
                 'MuseeL-Vanillekipferl,200,6400,H1',
                 'MuseeL-clam,200,6400,H1',
                 'MuseeL-Ndengese,200,6400,H1',
                 'MuseeL-Bobo,200,2500,6400,H1,H2',
                 'threebicycles,200,6400',
                 'sevenbicycles,200,1600,6400',
                 'Stevin,200,4000,6400',
                 'wall,200,640,6400',
                 'Saint-Remi,200,6400,H1,H2,H3',
                 'Saint-Remi-C,200,6400,H1,H2',
                 'books,200,1600,6400,H1,H2',
                 'bloop,200,3200,6400,H1',
                 'schooltop,200,800,6400,H1,H2',
                 'Sint-Joris,200,1000,2500,6400,H1,H2,H3',
                 'claycreature,200,4000,6400,H1',
                 'claycreatures,200,1600,5000,6400,H1,H2,H3',
                 'claytools,200,5000,6400,H1,H2,H3',
                 'CourtineDeVillersDebris,200,2500,6400,H1,H2',
                 'Leonidas,200,400,3200,6400,H1',
                 'pastries,200,3200,6400,H1,H2',
                 'mugshot,200,6400,H1',
                 'holywater,200,1600,4000,6400,H1,H2,H3',
                 'chapel,200,1000,6400,H1,H2,H3',
                 'directions,200,640,640-2,1250,6400,6400-2,H1,H2,H3',
                 'drowning,200,800,6400,H1,H2,H3',
                 'parking-keyboard,200,400,800,1600,3200,6400,H1,H2,H3',
                 'semicircle,200,320,640,1250,2500,5000,6400,H1,H2,H3',
                 'stairs,200,250,320,640,1250,2500,5000,6400,H1,H2',
                 'stefantiek,200,250,500,2000,6400,H1,H2',
                 'tree1,200,400,1600,3200,6400,H1,H2,H3',
                 'tree2,200,800,1600,3200,6400,H1,H2,H3',
                 'ursulines-building,200,250,400,1000,4000,6400,H1',
                 'ursulines-can,200,200-2,400,800,1600,3200,6400,H1,H2',
                 'ursulines-red,200,250,500,4000,6400,H1,H2',
                 'vlc,200,250,500,1000,3200,6400,H1,H2,H3',
                 'whistle,200,250,500,1000,2000,4000,6400,H1,H2,H3,H4',
                 'Homarus-americanus,200,200-2,250,400,800,2000,3200,5000,6400,H1,H2',
                 'fruits,200,200-2,800,3200,5000,6400',
                 'MVB-Sainte-Anne,200,200-2,250,640,4000,6400,H1',
                 'MVB-JardinBotanique,200,200-2,400,1000,2500,3200,6400',
                 'MVB-Urania,200,320,500,1000,2500,5000,6400,H1,H2',
                 'MVB-1887GrandPlace,200,200-2,400,640,2000,5000,6400,H1,H2',
                 'MVB-heraldicLion,200,200-2,320,1000,3200,6400,H1,H2',
                 'MVB-LouveFire,200,200-2,400,800,1600,3200,6400,6400-2,H1,H2',
                 'MVB-Bombardement,200,200-2,320,800,5000,6400,H1,H2,H3',
                 'beads,200,500,1000,3200,6400',
                 'shells,200,200-2,250,320,1000,1600,2500,3200,5000,6400,H1,H2,H3']

XT1_16bit_list = ['soap,200,200-2,400,800,3200,6400,H1,H2,H3,H4',
                  'kibbles,200,200-2,800,5000,6400,H1,H2,H3',
                  'bertrixtree,200,400,640,2500,4000,6400,H1',
                  'BruegelLibraryS1,200,400,1000,2500,3200,5000,6400,H1,H2',
                  'BruegelLibraryS2,200,500,1250,2500,5000,6400,H1,H2,H3,H4',
                  'LaptopInLibrary,200,500,800,1600,2500,6400,H1,H2,H3',
                  'banana,200,250,500,800,1250,2000,4000,6400,H1,H2,H3',
                  'dustyrubberduck,200,1000,1250,2500,5000,6400,H1,H2',
                  'partiallyeatenbanana,200,640,1250,2500,4000,5000,6400,H1,H2,H3',
                  'corkboard,200,320,1000,2500,5000,6400,H1,H2,H3',
                  'fireextinguisher,200,200-2,200-3,800,3200,6400,H1,H2,H3',
                  'colorscreen,200,201,202,400,1000,3200,6400,H1']

C500D_8bit_list = [
    'MuseeL-Bobo-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-yombe-C500D,100,400,800,1600,3200,H1',
    'MuseeL-sol-C500D,100,200,400,800,3200,H1',
    'MuseeL-skull-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-Sepik-C500D,100,200,800,1600,3200,H1',
    'MuseeL-Saint-Pierre-C500D,100,100-2,200,400,800,1600,3200,H1',
    'MuseeL-mammal-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-idole-C500D,100,100-2,200,400,800,3200,H1',
    'MuseeL-CopteArch-C500D,100,100-2,200,400,1600,3200',
    'MuseeL-cross-C500D,100,200,400,800,1600,3200,H1',
    'MuseeL-fuite-C500D,100,200,400,800,1600,3200,H1']

src_path = '/home/SENSETIME/yangmingzhuo/Documents/ECCV/Dataset/NIND/'
num = 0
for imgs in XT1_8bit_list:
    scene_name = imgs.split(',')[0]
    iso_list = imgs.split(',')[1:]
    dst_path = os.path.join(src_path, 'pre', 'XT1_8bit', scene_name)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for iso in iso_list:
        if iso == '200':
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.jpg'))
            gt = np.array(cv2.imread(file_list[0])).astype(np.float32)
        elif iso == '200-2':
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.jpg'))
            gt = (gt + np.array(cv2.imread(file_list[0])).astype(np.float32)) / 2
        else:
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.jpg'))
            shutil.copy(file_list[0], dst_path)
        print(scene_name, iso, file_list)
        num += 1
    gt = np.clip(gt, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(dst_path, 'NIND_' + scene_name + '_gt.png'), gt)

for imgs in XT1_16bit_list:
    scene_name = imgs.split(',')[0]
    iso_list = imgs.split(',')[1:]
    dst_path = os.path.join(src_path, 'pre', 'XT1_16bit', scene_name)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for iso in iso_list:
        if iso == '200':
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.png'))
            gt = np.array(cv2.imread(file_list[0])).astype(np.float32)
        elif iso == '200-2':
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.png'))
            gt = (gt + np.array(cv2.imread(file_list[0])).astype(np.float32)) / 2
        else:
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.png'))
            shutil.copy(file_list[0], dst_path)
        print(scene_name, iso, file_list)
        num += 1
    gt = np.clip(gt, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(dst_path, 'NIND_' + scene_name + '_gt.png'), gt)

for imgs in C500D_8bit_list:
    scene_name = imgs.split(',')[0]
    iso_list = imgs.split(',')[1:]
    dst_path = os.path.join(src_path, 'pre', 'C500D_8bit', scene_name)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for iso in iso_list:
        if iso == '100':
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.jpg'))
            gt = np.array(cv2.imread(file_list[0])).astype(np.float32)
        elif iso == '100-2':
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.jpg'))
            gt = (gt + np.array(cv2.imread(file_list[0])).astype(np.float32)) / 2
        else:
            file_list = glob.glob(os.path.join(src_path, 'NIND_' + scene_name + '_ISO' + iso + '.jpg'))
            shutil.copy(file_list[0], dst_path)
        print(scene_name, iso, file_list)
        num += 1
    gt = np.clip(gt, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(dst_path, 'NIND_' + scene_name + '_gt.png'), gt)

print(num)
