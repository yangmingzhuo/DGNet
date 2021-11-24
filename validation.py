from __future__ import print_function

import time
import argparse
import torch.cuda.random
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from skimage.measure import compare_psnr, compare_ssim
from model.ELD_UNet import ELD_UNet
from data.dataloader import *
from utils.util import *

def valid(data_loader, model, logger):
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
            ssim_val.update(compare_ssim(np.transpose(np.squeeze(prediction[i, :, :, :]), (1, 2, 0)), np.transpose(np.squeeze(target[i, :, :, :]), (1, 2, 0)), data_range=1.0, gaussian_weights=True, use_sample_covariance=True, multichannel=True), 1)

    logger.info('Validation end, PSNR: {:.4f} dB, SSIM: {:.4f}, Time: {:.4f}'.format(psnr_val.avg, ssim_val.avg, time.time() - t0))
    logger.info('------------------------------------------------------------------')
    return psnr_val.avg, ssim_val.avg


def main():
    # Testing settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size, default=1')
    parser.add_argument('--data_set', type=str, default='sidd', help='the exact dataset we want to train on')
    parser.add_argument('--pretrained', type=str, default='/mnt/lustre/yangmingzhuo/DGNet/logs/model_ELU_UNet_ds_sidd_bs_32_ps_128_ep_200_lr_0.0002_rd_True/checkpoint', help="Checkpoints directory,  (default:./checkpoints)")

    # Global settings
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--gpus', default=1, type=str, help='id of gpus')
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/yangmingzhuo', help='the dataset dir')
    parser.add_argument('--log_dir', default='./logs/', help='Location to save checkpoint models')
    parser.add_argument('--model_type', type=str, default='ELU_UNet', help='the name of model')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    opt = parser.parse_args()

    # Initialize
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # Log setting
    mkdir(opt.log_dir)
    log_folder = os.path.join(opt.log_dir)
    logger = get_logger(log_folder, 'DGNet_log')

    # Load Dataset
    logger.info("Load data from: {}".format(opt.data_dir, 'random_processed', 'test', opt.data_set + '_patch_test'))
    val_set = LoadDataset(src_path=os.path.join(opt.data_dir, 'random_processed', 'test', opt.data_set + '_patch_test'), patch_size=opt.patch_size, train=False)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # Load Network
    logger.info('Using model {}'.format(opt.model_type))
    model = ELD_UNet()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Push all model to data parallel and then gpu")
    else:
        logger.info("Push all model to one gpu")
    model.cuda()

    # load pretrained model
    logger.info("Load model from: {}".format(opt.pretrained))
    path_chk_rest = get_last_path(opt.pretrained, '_best.pth')
    load_checkpoint(model, path_chk_rest)

    # Training
    valid(val_data_loader, model, logger)


if __name__ == '__main__':
    main()
