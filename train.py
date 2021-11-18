from __future__ import print_function
import os
import time
import pandas as pd
import argparse

import torch.cuda.random
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model.ELD_UNet import ELD_UNet
from data.dataloader import *
from utils.util import *
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate. default=0.0002')
parser.add_argument('--lr_min', type=float, default=0.0001, help='minimum learning rate. default=0.000001')
parser.add_argument('--resume', default=False, help='Whether to resume the training')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size, default=1')


# Global settings
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=1, type=str, help='id of gpus')
parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/processed', help='the dataset dir')
parser.add_argument('--log_folder', default='./logs/', help='Location to save checkpoint models')
parser.add_argument('--model_type', type=str, default='ELU_UNet', help='the name of model')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--ex_id', type=int, default=1, help='experiment id, default=1')
opt = parser.parse_args()


def train(epoch, model, data_loader, optimizer, criterion, logger, writer):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(data_loader, 0):
        target = batch[1]
        input = batch[0]

        input = input.cuda()
        target = target.cuda()

        model.zero_grad()
        optimizer.zero_grad()
        t0 = time.time()

        prediction = model(input)

        loss = criterion(prediction, target)/(input.size()[0]*2)

        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        logger.info("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(data_loader), loss.data, (t1 - t0)))
    avg_loss = epoch_loss / len(data_loader)
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(data_loader)))
    writer.add_scalar('Train_loss', avg_loss, epoch)


def valid(data_loader, model, logger):
    psnr_test = 0
    model.eval()
    for iteration, batch in enumerate(data_loader, 0):
        target = batch[1]
        input = batch[0]
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            prediction = model(input)
            prediction = torch.clamp(prediction, 0., 1.)
        psnr_test += batch_PSNR(prediction, target, 1.)
    logger.info("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(data_loader)))
    return psnr_test / len(data_loader)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # Logger
    mkdir(opt.log_folder)
    log_folder_name = "model_{}_bs_{}_ps_{}_ep_{}_lr_{}_id_{}".format(opt.model_type, opt.batch_size, opt.patch_size, opt.nEpochs, opt.lr, opt.ex_id)
    log_folder = os.path.join(opt.log_folder, log_folder_name)
    mkdir(log_folder)
    checkpoint_folder = os.path.join(log_folder, 'checkpoint')
    mkdir(checkpoint_folder)
    logger = get_logger(log_folder, 'DGNet_log')
    logger.info(opt)
    logger.info('===>Loading datasets')

    # Load Dataset
    train_set = load_dataset(src_path=os.path.join(opt.data_dir, 'train', 'sidd_patch_train'), patch_size=opt.patch_size, train=True)
    train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_set = load_dataset(src_path=os.path.join(opt.data_dir, 'test', 'sidd_patch_test'), patch_size=opt.patch_size, train=False)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Load Network
    logger.info('===> Building model {}'.format(opt.model_type))
    model = ELD_UNet()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info("push all model to dataparallel and then gpu")
    else:
        model.cuda()
        logger.info("push all model to one gpu")
    criterion = nn.MSELoss()
    logger.info('---------- Networks architecture -------------')
    print_network(model, logger)
    logger.info('----------------------------------------------')

    # Scheduler
    start_epoch = opt.start_iter
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=opt.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    if opt.resume:
        path_chk_rest = get_last_path(opt.trained_model, '_latest.pth')
        load_checkpoint(model, path_chk_rest)
        start_epoch = load_start_epoch(path_chk_rest) + 1
        load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info('--------------------------------------------------------------------------')
        logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
        logger.info('--------------------------------------------------------------------------')

    # Training
    PSNR = []
    writer = SummaryWriter(os.path.join(log_folder, 'logs'))
    for epoch in range(start_epoch, opt.nEpochs + 1):
        train(epoch, model, train_data_loader, optimizer, criterion, logger, writer)
        psnr = valid(val_data_loader, model, logger)
        PSNR.append(psnr)
        scheduler.step()
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(checkpoint_folder, "model_latest.pth"))
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        writer.add_scalar('Validation_PSNR', psnr, epoch)


if __name__ == '__main__':
    main()
