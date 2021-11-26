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
from utils.checkpoint import *


def valid(opt, epoch, data_loader, model, logger, writer):
    t0 = time.time()
    model.eval()
    psnr_val = AverageMeter()

    for iteration, (noisy, target) in enumerate(data_loader):
        noisy, target = noisy.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(noisy)
            prediction = torch.clamp(prediction, 0.0, 1.0)

        prediction = prediction.data.cpu().numpy().astype(np.float32)
        target = target.data.cpu().numpy().astype(np.float32)
        for i in range(prediction.shape[0]):
            psnr_val.update(compare_psnr(prediction[i, :, :, :], target[i, :, :, :], data_range=1.0), 1)

    writer.add_scalar('Validation_PSNR', psnr_val.avg, epoch)
    logger.info('||==> Validation epoch: [{:d}/{:d}]\tval_PSNR={:.4f}\tcost_time={:.4f}'
                .format(epoch, opt.nEpochs, psnr_val.avg, time.time() - t0))
    return psnr_val.avg


def main():
    # testing settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size, default=1')
    parser.add_argument('--data_set', type=str, default='sidd', help='the exact dataset we want to train on')
    parser.add_argument('--pretrained', type=str,
                        default='./logs/model_ELU_UNet_ds_sidd_bs_32_ps_128_ep_200_lr_0.0002_rd_True/checkpoint',
                        help="Checkpoints directory,  (default:./checkpoints)")

    # global settings
    parser.add_argument('--gpus', default=1, type=str, help='id of gpus')
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/yangmingzhuo', help='the dataset dir')
    parser.add_argument('--log_dir', default='./logs/', help='Location to save checkpoint models')
    parser.add_argument('--model_type', type=str, default='ELU_UNet', help='the name of model')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    opt = parser.parse_args()

    # initialize
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # log setting
    mkdir(opt.log_dir)
    log_folder = os.path.join(opt.log_dir)
    logger = get_logger(log_folder, 'DGNet_log')

    # load dataset
    logger.info("Load data from: {}".format(opt.data_dir, 'random_processed', 'test', opt.data_set + '_patch_test'))
    val_set = LoadDataset(src_path=os.path.join(opt.data_dir, 'random_processed', 'test', opt.data_set + '_patch_test'),
                          patch_size=opt.patch_size, train=False)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.test_batch_size, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)

    # load network
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
    model, psnr_best = load_single_model(opt.pretrain_model, model, logger)

    valid(val_data_loader, model, logger)
    dst_folder = make_dir(os.path.join(checkpoint_folder, opt.data_set))
    gen_mat(ELD_UNet(), os.path.join(checkpoint_folder, "model_best.pth"), dst_folder, val_data_loader, logger)

if __name__ == '__main__':
    main()