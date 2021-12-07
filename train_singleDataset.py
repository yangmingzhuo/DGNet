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
from skimage.measure import compare_psnr

from model.ELD_UNet import ELD_UNet
from data.dataloader import *
from utils.util import *
from utils.checkpoint import *
from utils.gen_mat import *


def train(opt, epoch, model, data_loader, optimizer, scheduler, criterion, logger, writer):
    t0 = time.time()
    epoch_loss = AverageMeter()
    model.train()

    for iteration, (noisy, target) in enumerate(data_loader):
        noisy, target = noisy.cuda(), target.cuda()
        prediction = model(noisy)
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.data, noisy.size(0))
        if iteration % opt.print_freq == 0:
            logger.info('Train epoch: [{:d}/{:d}]\titeration: [{:d}/{:d}]\tlr={:.6f}\tl1_loss={:.4f}'
                        .format(epoch, opt.nEpochs, iteration, len(data_loader), scheduler.get_lr()[0], epoch_loss.avg))

    writer.add_scalar('Train_L1_loss', epoch_loss.avg, epoch)
    writer.add_scalar('Learning_rate', scheduler.get_lr()[0], epoch)
    logger.info('||==> Train epoch: [{:d}/{:d}]\tlr={:.6f}\tl1_loss={:.4f}\tcost_time={:.4f}'
                .format(epoch, opt.nEpochs, scheduler.get_lr()[0], epoch_loss.avg, time.time() - t0))


def valid(opt, epoch, data_loader, model, criterion, logger, writer):
    t0 = time.time()
    model.eval()
    psnr_val = AverageMeter()
    loss_val = AverageMeter()

    for iteration, (noisy, target) in enumerate(data_loader):
        noisy, target = noisy.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(noisy)
            prediction = torch.clamp(prediction, 0.0, 1.0)

        loss = criterion(prediction, target)
        prediction = prediction.data.cpu().numpy().astype(np.float32)
        target = target.data.cpu().numpy().astype(np.float32)
        for i in range(prediction.shape[0]):
            psnr_val.update(compare_psnr(prediction[i, :, :, :], target[i, :, :, :], data_range=1.0), 1)
        loss_val.update(loss.data, prediction.shape[0])

    writer.add_scalar('Validation_PSNR', psnr_val.avg, epoch)
    writer.add_scalar('Validation_loss', loss_val.avg, epoch)
    logger.info('||==> Validation epoch: [{:d}/{:d}]\tval_PSNR={:.4f}\tval_loss={:.4f}\tcost_time={:.4f}'
                .format(epoch, opt.nEpochs, psnr_val.avg, loss_val.avg, time.time() - t0))
    return psnr_val.avg


def main():
    parser = argparse.ArgumentParser(description='PyTorch image denoising')
    # dataset settings
    parser.add_argument('--data_set', type=str, default='sidd', help='the exact dataset we want to train on')
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/processed',
                        help='the dataset dir')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size: 32')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size, default=1')
    parser.add_argument('--test_patch_size', type=int, default=256, help='testing batch size, default=1')

    # training settings
    parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate. default=0.0002')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum learning rate. default=0.000001')
    parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight_decay')

    # model settings
    parser.add_argument('--model_type', type=str, default='ELU_UNet', help='the name of model')
    parser.add_argument('--pretrain_model', type=str, default='', help='pretrain model path')

    # general settings
    parser.add_argument('--gpus', default='1', type=str, help='id of gpus')
    parser.add_argument('--log_dir', default='./logs_v2/', help='Location to save checkpoint models')
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
    psnr_best = 0
    epoch_best = 0

    # log setting
    log_folder = os.path.join(opt.log_dir, "model_{}_gpu_{}_ds_{}_ps_{}_bs_{}_ep_{}_lr_{}_lr_min_{}_exp_id_{}"
                              .format(opt.model_type, opt.gpus, opt.data_set, opt.patch_size, opt.batch_size,
                                      opt.nEpochs, opt.lr, opt.lr_min, opt.exp_id))
    if opt.pretrain_model == '':
        output_process(log_folder, 'd')
        checkpoint_folder = make_dir(os.path.join(log_folder, 'checkpoint'))
    writer = SummaryWriter(log_folder)
    logger = get_logger(log_folder, 'DGNet_log')
    logger.info(opt)

    # load dataset
    logger.info('Loading datasets {}, Batch Size: {}, Patch Size: {}'.format(opt.data_set,
                                                                             opt.batch_size, opt.patch_size))
    train_set = LoadDataset(src_path=os.path.join(opt.data_dir, opt.data_set, 'train'), patch_size=opt.patch_size,
                            train=True)
    train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.num_workers, pin_memory=True)

    val_set = LoadDataset(src_path=os.path.join(opt.data_dir, opt.data_set, 'test'), patch_size=opt.test_patch_size,
                          train=False)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.test_batch_size, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)

    # load network
    logger.info('Building model {}'.format(opt.model_type))
    model = ELD_UNet()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Push model to data parallel and then gpu!")
    else:
        logger.info("Push model to one gpu!")
    model.cuda()
    logger.info('model={}'.format(model))

    # loss
    logger.info('==> Use L1 loss as criterion')
    criterion = nn.L1Loss()

    # optimizer and scheduler
    warmup_epochs = 3
    # t_max = opt.nEpochs - warmup_epochs + opt.nEpochs / 2
    t_max = opt.nEpochs - warmup_epochs
    logger.info('Optimizer: Adam Warmup epochs: {}, Learning rate: {}, Scheduler: CosineAnnealingLR, T_max: {}'
                .format(warmup_epochs, opt.lr, t_max))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    logger.info('optimizer={}'.format(optimizer))

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=opt.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    logger.info('scheduler={}'.format(scheduler))
    scheduler.step()

    # resume
    if opt.pretrain_model != '':
        model, start_epoch, optimizer, psnr_best = load_model(opt.pretrain_model, model, optimizer, logger)
        start_epoch += 1
        for i in range(1, start_epoch):
            scheduler.step()
        logger.info('Resume start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]))
    else:
        start_epoch = opt.start_iter
        logger.info('Start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]))

    # training
    for epoch in range(start_epoch, opt.nEpochs + 1):
        # training
        train(opt, epoch, model, train_data_loader, optimizer, scheduler, criterion, logger, writer)
        # validation
        if epoch > 100 or epoch < 3 or epoch % 5 == 0:
            psnr = valid(opt, epoch, val_data_loader, model, criterion, logger, writer)

        # save model
        save_model(os.path.join(checkpoint_folder, "model_latest.pth"), epoch, model, optimizer, psnr_best, logger)

        if psnr > psnr_best:
            psnr_best = psnr
            epoch_best = epoch
            save_model(os.path.join(checkpoint_folder, "model_best.pth"), epoch, model, optimizer, psnr_best, logger)
        scheduler.step()
        logger.info('||==> best_epoch = {}, best_psnr = {}'.format(epoch_best, psnr_best))

    # generate evaluate_mat for SSIM validation
    # gen_mat(ELD_UNet(), os.path.join(checkpoint_folder, "model_best.pth"), checkpoint_folder, val_data_loader,
    #        opt.test_batch_size, opt.test_patch_size, logger)


if __name__ == '__main__':
    main()
