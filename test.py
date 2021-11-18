import os
import numpy as np
from skimage import img_as_ubyte
import argparse
from model.ELD_UNet import ELD_UNet
from tqdm import tqdm
from scipy.io import loadmat, savemat
from utils.util import load_checkpoint, mkdir
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='./logs/', help="Checkpoints directory,  (default:./checkpoints)")
parser.add_argument('--data_folder', type=str, default='/mnt/lustre/share/yangmingzhuo/processed/SIDD', help='Location to save checkpoint models')
parser.add_argument('--out_folder', type=str, default='/mnt/lustre/share/yangmingzhuo/test_result/SIDD', help='Location to save checkpoint models')
parser.add_argument('--model', type=str, default='model_latest.pth', help='Location to save checkpoint models')
parser.add_argument('--Type', type=str, default='SIDD', help='To choose the testing benchmark dataset, SIDD or Dnd')
parser.add_argument('--gpus', default=1, type=str, help='number of gpus')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus


def denoise(model, noisy_image):
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()

        phi = model(noisy_image)
        torch.cuda.synchronize()
        im_denoise = phi.cpu().numpy()

    im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))
    im_denoise = img_as_ubyte(im_denoise.clip(0, 1))

    return im_denoise


def main():
    use_gpu = True
    # load the pretrained model
    print('Loading the Model')
    net = ELD_UNet()
    checkpoint = os.path.join(opt.pretrained, opt.model)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    load_checkpoint(net, checkpoint)
    net.eval()
    mkdir(opt.out_folder)

    # load SIDD benchmark dataset and information
    noisy_data_mat_file = os.path.join(opt.data_folder, 'ValidationNoisyBlocksSrgb.mat')
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]

    npose = (noisy_data_mat.shape[0])
    nsmile = noisy_data_mat.shape[1]
    poseSmile_cell = np.empty((npose, nsmile), dtype=object)

    for image_index in tqdm(range(noisy_data_mat.shape[0])):
        for block_index in range(noisy_data_mat.shape[1]):
            noisy_image = noisy_data_mat[image_index, block_index, :, :, :]
            noisy_image = np.float32(noisy_image / 255.)
            noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1))[np.newaxis,])
            img = denoise(net, noisy_image)
            save_file = os.path.join(opt.out_folder, '%04d_%02d.png' % (image_index + 1, block_index + 1))
            cv2.imwrite(save_file, cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_RGB2BGR))
            poseSmile_cell[image_index, block_index] = img

    submit_data = {
            'DenoisedBlocksSrgb': poseSmile_cell
        }

    savemat(
            os.path.join(opt.out_folder, 'SubmitSrgb.mat'),
            submit_data
        )

if __name__ == '__main__':
    main()