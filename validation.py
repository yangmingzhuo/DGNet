import os
import torch
import cv2
from utils.util import *
from data.dataloader import *
from torch.utils.data import DataLoader
import time
import numpy as np
from skimage import img_as_ubyte
import argparse
from model.ELD_UNet import ELD_UNet
from tqdm import tqdm
from scipy.io import loadmat, savemat
from skimage.measure import compare_ssim, compare_psnr


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='./', help="Checkpoints directory,  (default:./checkpoints)")
parser.add_argument('--data_folder', type=str, default='/mnt/lustre/share/yangmingzhuo/processed/SIDD', help='Location to save checkpoint models')
parser.add_argument('--out_folder', type=str, default='/mnt/lustre/share/yangmingzhuo/test_result/SIDD', help='Location to save checkpoint models')
parser.add_argument('--model', type=str, default='model_best.pth', help='Location to save checkpoint models')
parser.add_argument('--Type', type=str, default='SIDD', help='To choose the testing benchmark dataset, SIDD or Dnd')
parser.add_argument('--gpus', default=1, type=str, help='number of gpus')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus


def valid(epoch, data_loader, model, logger, writer):
    logger.info('Epoch[{}]: Validation start'.format(epoch))
    t0 = time.time()
    model.eval()
    psnr_val = AverageMeter()
    ssim_val = AverageMeter()

    for iteration, batch in enumerate(data_loader, 0):
        noisy = batch[0].cuda()
        target = batch[1].cuda()

        with torch.no_grad():
            prediction = model(noisy)
            prediction = torch.clamp(prediction, 0.0, 1.0)

        prediction = prediction.data.cpu().numpy().astype(np.float32)
        target = target.data.cpu().numpy().astype(np.float32)
        for i in range(prediction.shape[0]):
            psnr_val.update(compare_psnr(prediction[i, :, :, :], target[i, :, :, :], data_range=1.0), 1)
            ssim_val.update(compare_ssim(np.transpose(np.squeeze(prediction[i, :, :, :]), (1, 2, 0)), np.transpose(np.squeeze(target[i, :, :, :]), (1, 2, 0)), data_range=1.0, multichannel=True), 1)

    writer.add_scalar('Validation PSNR', psnr_val.avg, epoch)
    writer.add_scalar('Validation SSIM', ssim_val.avg, epoch)
    logger.info('Epoch[{}]: Validation end, Average PSNR: {:.4f} dB, Average SSIM: {:.4f}, Time: {:.4f}'.format(epoch, psnr_val.avg, ssim_val.avg, time.time() - t0))
    logger.info('------------------------------------------------------------------')
    return psnr_val.avg, ssim_val.avg


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate. default=0.0002')
    parser.add_argument('--lr_min', type=float, default=0.000001, help='minimum learning rate. default=0.000001')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size, default=1')
    parser.add_argument('--data_set', type=str, default='sidd', help='the exact dataset we want to train on')
    parser.add_argument('--random', action='store_true', help='whether to randomly crop images')

    # Global settings
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--gpus', default=1, type=str, help='id of gpus')
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/yangmingzhuo', help='the dataset dir')
    parser.add_argument('--log_dir', default='./logs/', help='Location to save checkpoint models')
    parser.add_argument('--model_type', type=str, default='ELU_UNet', help='the name of model')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
    parser.add_argument('--resume', action='store_true', help='Whether to resume the training')
    parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
    parser.add_argument('--weight_decay', type=float, default=0.00000001, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    opt = parser.parse_args()
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

    val_set = LoadDataset(src_path=os.path.join(opt.data_dir, data_process, 'test', opt.data_set + '_patch_test'), patch_size=opt.patch_size, train=False)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True)

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