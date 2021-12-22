import time
import argparse
import shutil
import torch.cuda.random
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr

from model.ELD_UNet import ELD_UNet
from model.DG_UNet import *
from data.dataloader import *
from utils.util import *
from utils.checkpoint import *
from utils.gen_mat import *
from loss.loss import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
torchvision.set_image_backend('accimage')


def train(opt, epoch, ad_net, data_loader, optimizer, scheduler, logger, writer):
    t0 = time.time()
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    ad_net.train()

    for iteration, (target, label) in enumerate(data_loader):
        target, label = target.cuda(), label.cuda()
        target_ad_out = ad_net(target)
        target_acc = accuracy(target_ad_out, label)

        loss = get_ad_loss(target_ad_out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.data, target.size(0))
        epoch_acc.update(target_acc, target.size(0))
        if iteration % opt.print_freq == 0:
            logger.info('Train epoch: [{:d}/{:d}]\titeration: [{:d}/{:d}]\tlr={:.6f}\tloss={:.4f}\tacc={:4f}'
                        .format(epoch, opt.nEpochs, iteration, len(data_loader), scheduler.get_lr()[0], epoch_loss.avg, target_acc))

    writer.add_scalar('Train_loss', epoch_loss.avg, epoch)
    writer.add_scalar('Learning_rate', scheduler.get_lr()[0], epoch)
    writer.add_scalar('Accuracy', epoch_acc.avg, epoch)
    logger.info('||==> Train epoch: [{:d}/{:d}]\tlr={:.6f}\tl1_loss={:.4f}\tacc={:4f}\tcost_time={:.4f}'
                .format(epoch, opt.nEpochs, scheduler.get_lr()[0], epoch_loss.avg, epoch_acc.avg, time.time() - t0))
    return epoch_acc.avg


def main():
    parser = argparse.ArgumentParser(description='PyTorch image denoising')
    # dataset settings
    parser.add_argument('--data_set1', type=str, default='renoir_v2', help='the exact dataset we want to train on')
    parser.add_argument('--data_set2', type=str, default='nind', help='the exact dataset we want to train on')
    parser.add_argument('--data_set3', type=str, default='rid2021_v2', help='the exact dataset we want to train on')
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/processed',
                        help='the dataset dir')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size: 32')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')

    # training settings
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate. default=0.0002')
    parser.add_argument('--lr_min', type=float, default=1e-3, help='minimum learning rate. default=0.000001')
    parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')

    # model settings
    parser.add_argument('--model_type', type=str, default='Discriminator_model_v2', help='the name of model')
    parser.add_argument('--pretrain_model', type=str, default='', help='pretrain model path')

    # general settings
    parser.add_argument('--gpus', default='0', type=str, help='id of gpus')
    parser.add_argument('--log_dir', default='./logs_disc/', help='Location to save checkpoint models')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--print_freq', type=int, default=10, help='print freq')
    parser.add_argument('--exp_id', type=int, default=0, help='experiment')
    opt = parser.parse_args()

    # initialize
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    epoch_acc_best = 0
    epoch_best = 0

    # log setting
    log_folder = os.path.join(opt.log_dir, "model_{}_gpu_{}_ds_{}_{}_{}_ps_{}_bs_{}_ep_{}_lr_{}_lr_min_{}_exp_id_{}"
                              .format(opt.model_type, opt.gpus, opt.data_set1, opt.data_set2, opt.data_set3,
                                      opt.patch_size, opt.batch_size, opt.nEpochs, opt.lr, opt.lr_min, opt.exp_id))
    output_process(log_folder)
    checkpoint_folder = make_dir(os.path.join(log_folder, 'checkpoint'))
    writer = SummaryWriter(log_folder)
    logger = get_logger(log_folder, 'DGNet_log')
    logger.info(opt)

    # load dataset
    logger.info('Loading datasets {} {} {}, Batch Size: {}, Patch Size: {}'.format(opt.data_set1, opt.data_set2,
                                                                                   opt.data_set3, opt.batch_size,
                                                                                   opt.patch_size))
    train_set = LoadMultiDataset_clean(src_path1=os.path.join(opt.data_dir, opt.data_set1, 'train'),
                                       src_path2=os.path.join(opt.data_dir, opt.data_set2, 'train'),
                                       src_path3=os.path.join(opt.data_dir, opt.data_set3, 'train'),
                                       patch_size=opt.patch_size,
                                       train=True)
    train_data_loader = DataLoaderX(dataset=train_set, batch_size=opt.batch_size, shuffle=True,
                                    num_workers=opt.num_workers, pin_memory=True)
    logger.info('Train dataset length: {}'.format(len(train_data_loader)))

    # load network
    logger.info('Building model {}'.format(opt.model_type))
    ad_net = Discriminator_model_v2()

    if torch.cuda.device_count() > 1:
        ad_net = torch.nn.DataParallel(ad_net)
        logger.info("Push model to data parallel and then gpu!")
    else:
        logger.info("Push model to one gpu!")
    ad_net.cuda()
    logger.info('model={}'.format(ad_net))

    # loss
    logger.info('==> Use CE loss as criterion')

    # optimizer and scheduler
    t_max = opt.nEpochs
    logger.info('Optimizer: Adam, Learning rate: {}, Scheduler: CosineAnnealingLR, T_max: {}'
                .format(opt.lr, t_max))

    optimizer = optim.Adam(ad_net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    logger.info('optimizer={}'.format(optimizer))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=opt.lr_min)

    logger.info('scheduler={}'.format(scheduler))
    scheduler.step()

    # resume
    if opt.pretrain_model != '':
        ad_net, start_epoch, optimizer, acc_best = load_ad_net_dp(opt.pretrain_model, ad_net, optimizer, logger)
        start_epoch += 1
        for i in range(1, start_epoch):
            scheduler.step()
        logger.info('Resume start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]))
    else:
        start_epoch = opt.start_epoch
        logger.info('Start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]))

    # training
    for epoch in range(start_epoch, opt.nEpochs + 1):
        # training
        epoch_acc = train(opt, epoch, ad_net, train_data_loader, optimizer, scheduler, logger, writer)

        if epoch_acc >= epoch_acc_best:
            epoch_acc_best = epoch_acc
            epoch_best = epoch
            save_ad_net(os.path.join(checkpoint_folder, "ad_net_best.pth"), epoch, ad_net, optimizer, epoch_acc_best, logger)
        # save model
        save_ad_net(os.path.join(checkpoint_folder, "ad_net_latest.pth"), epoch, ad_net, optimizer, epoch_acc, logger)

        scheduler.step()
        logger.info('||==> best_epoch = {}, best_acc = {}'.format(epoch_best, epoch_acc_best))

    # generate evaluate_mat for SSIM validation
    # gen_mat(ELD_UNet(), os.path.join(checkpoint_folder, "model_best.pth"), checkpoint_folder, val_data_loader,
    #        opt.test_batch_size, opt.test_patch_size, logger)


if __name__ == '__main__':
    main()
