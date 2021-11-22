from __future__ import print_function

import os
import time
import argparse
import shutil
import torch.cuda.random
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter
from skimage.measure import compare_ssim, compare_psnr
from model.ELD_UNet import ELD_UNet
from data.dataloader import *
from utils.util import *


def train(epoch, model, data_loader, optimizer, scheduler, criterion, logger, writer):
    logger.info('------------------------------------------------------------------')
    logger.info('Epoch[{}]: Train start LearningRate {:.6f}'.format(epoch, scheduler.get_lr()[0]))
    t0 = time.time()
    epoch_loss = AverageMeter()
    model.train()
    for iteration, batch in enumerate(data_loader, 0):
        noisy = batch[0].cuda()
        target = batch[1].cuda()

        prediction = model(noisy)
        loss = criterion(prediction, target)
        epoch_loss.update(loss.data, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info('Epoch[{}]({:04d}/{:04d}): Loss: {:.4f}'.format(epoch, iteration, len(data_loader), loss.data))
    writer.add_scalar('Train Loss', epoch_loss.avg, epoch)
    logger.info('Epoch[{}]: Train end, Average Loss: {:.4f}, Time: {:.4f}, LearningRate {:.6f}'.format(epoch, epoch_loss.avg, time.time() - t0, scheduler.get_lr()[0]))
    logger.info('------------------------------------------------------------------')


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
            psnr_val.update(compare_psnr(prediction[i, :, :, :], target[i, :, :, :], 1), 1)
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
    log_folder = os.path.join(opt.log_dir, "model_{}_ds_{}_bs_{}_ps_{}_ep_{}_lr_{}_rd_{}".format(opt.model_type, opt.data_set, opt.batch_size, opt.patch_size, opt.nEpochs, opt.lr, opt.random))
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
    log_folder = mkdir(log_folder)
    checkpoint_folder = mkdir(os.path.join(log_folder, 'checkpoint'))
    writer = SummaryWriter(log_folder)
    logger = get_logger(log_folder, 'DGNet_log')

    # Load Dataset
    if opt.random:
        data_process = 'random_processed'
    else:
        data_process = 'processed'
    logger.info('Loading datasets {}, Batch Size: {}, Patch Size: {}'.format(opt.data_set, opt.batch_size, opt.patch_size))
    train_set = LoadDataset(src_path=os.path.join(opt.data_dir, data_process, 'train', opt.data_set + '_patch_train'), patch_size=opt.patch_size, train=True)
    train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_set = LoadDataset(src_path=os.path.join(opt.data_dir, data_process, 'test', opt.data_set + '_patch_test'), patch_size=opt.patch_size, train=False)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Load Network
    logger.info('Building model {}'.format(opt.model_type))
    model = ELD_UNet()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Push all model to data parallel and then gpu")
    else:
        logger.info("Push all model to one gpu")
    model.cuda()
    logger.info('----------------------- Networks architecture --------------------------')
    print_network(model, logger)
    logger.info('------------------------------------------------------------------------')

    # loss
    logger.info('==> Use L1 loss as criterion')
    criterion = nn.L1Loss()

    # Scheduler
    warmup_epochs = 3
    t_max = opt.nEpochs - warmup_epochs + 40
    logger.info('Optimizer: Adam Warmup epochs: {}, Learning rate: {}, Scheduler: CosineAnnealingLR, T_max: {}'.format(warmup_epochs, opt.lr, t_max))
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=opt.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    # resume
    if opt.resume:
        path_chk_rest = get_last_path(checkpoint_folder, '_latest.pth')
        load_checkpoint(model, path_chk_rest)
        start_epoch = load_start_epoch(path_chk_rest) + 1
        load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info('Resume start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]))
    else:
        start_epoch = opt.start_iter
        logger.info('Start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]))

    # Training
    psnr_best = 0
    ssim_best = 0
    epoch_best = 0
    for epoch in range(start_epoch, opt.nEpochs + 1):
        writer.add_scalar('Learning Rate', scheduler.get_lr()[0], epoch)
        train(epoch, model, train_data_loader, optimizer, scheduler, criterion, logger, writer)
        psnr, ssim = valid(epoch, val_data_loader, model, logger, writer)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'ptimizer': optimizer.state_dict()}, os.path.join(checkpoint_folder, "model_latest.pth"))
        if psnr > psnr_best:
            psnr_best = psnr
            ssim_best = ssim
            epoch_best = epoch
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(checkpoint_folder, "model_best.pth"))
        scheduler.step()

    logger.info("best_psnr = {}, best_ssim ={}".format(psnr_best, ssim_best, epoch_best))


if __name__ == '__main__':
    main()
