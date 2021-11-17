from __future__ import print_function
import os
import time
import socket
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model.ELD_UNet import ELD_UNet
from data.dataloader import *
from utils.util import *
from warmup_scheduler import GradualWarmupScheduler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate. default=0.0002')
parser.add_argument('--lr_min', type=float, default=0.000001, help='minimum learning rate. default=0.000001')
parser.add_argument('--data_augmentation', type=bool, default=True, help='if adopt augmentation when training')
parser.add_argument('--save_folder', default='./checkpoint/', help='Location to save checkpoint models')
parser.add_argument('--statistics', default='./statistics/', help='Location to save statistics')
parser.add_argument('--resume', default=False, help='Whether to resume the training')

# Testing settings
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size, default=1')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# Global settings
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
parser.add_argument('--data_dir', type=str, default='./data', help='the dataset dir')
parser.add_argument('--model_type', type=str, default='DGNet', help='the name of model')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True


def train(epoch, model, data_loader, optimizer, criterion, logger):
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
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(data_loader)))


def valid(data_set, model, logger):
    psnr_test= 0
    model.eval()
    for iteration, batch in enumerate(data_set, 0):
        target = batch[1]
        input = batch[0]

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            prediction = model(input)
            prediction = torch.clamp(prediction, 0., 1.)
        psnr_test += batch_PSNR(prediction, target, 1.)
    logger.info("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(data_set)))
    return psnr_test / len(data_set)


def main():
    # Logger
    logger = get_logger(opt.save_folder, 'DGNet_log')
    logger.info(opt)
    logger.info('===>Loading datasets')

    # Load Dataset
    train_set = Dataset_h5_real(src_path=os.path.join(opt.data_dir, 'train', 'sidd_train.h5'), patch_size=opt.patch_size, train=True)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_set = Dataset_h5_real(src_path=os.path.join(opt.data_dir, 'test', 'sidd_val.h5'), patch_size=opt.patch_size, train=False)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Load Network
    logger.info('===> Building model {}'.format(opt.model_type))
    model = ELD_UNet()
    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = nn.MSELoss()
    logger.info('---------- Networks architecture -------------')
    print_network(model, logger)
    logger.info('----------------------------------------------')

    # Scheduler
    start_epoch = opt.start_iter
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nEpochs - warmup_epochs + 40, eta_min=opt.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    if opt.resume:
        path_chk_rest = get_last_path(opt.statistics, '_latest.pth')
        load_checkpoint(model, path_chk_rest)
        start_epoch = load_start_epoch(path_chk_rest) + 1
        load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info('------------------------------------------------------------------------------')
        logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
        logger.info('------------------------------------------------------------------------------')

    # Training
    PSNR = []
    for epoch in range(start_epoch, opt.nEpochs + 1):
        train(epoch, model, training_data_loader, optimizer, criterion, logger)
        psnr = valid(testing_data_loader, model, logger)
        PSNR.append(psnr)
        data_frame = pd.DataFrame(
            data={'epoch': epoch, 'PSNR': PSNR}, index=range(1, epoch+1)
        )
        data_frame.to_csv(os.path.join(opt.statistics, 'training_logs.csv'), index_label='index')
        scheduler.step()
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(opt.statistics, "model_latest.pth"))


if __name__ == '__main__':
    main()
