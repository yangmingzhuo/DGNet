import time
import argparse

import torch.cuda.random
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr

from model.ELD_UNet import ELD_UNet
from utils.util import *
from model.DG_UNet import *
from loss.loss import *
from data.dataloader import *
from utils.gen_mat import *
from utils.checkpoint import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
torchvision.set_image_backend('accimage')


def train(opt, epoch, model, data_loader, optimizer, scheduler, criterion, logger, writer):
    t0 = time.time()
    epoch_l1_loss = AverageMeter()
    epoch_total_loss = AverageMeter()
    model.train()
    for iteration, batch in enumerate(data_loader):
        # load data
        (noisy, target, label) = batch
        noisy, target = noisy.cuda(opt.local_rank, non_blocking=True), target.cuda(opt.local_rank, non_blocking=True)

        # forward
        prediction = model(noisy)
        l1_loss = criterion(prediction, target)
        total_loss = l1_loss

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # output
        dist.barrier()
        reduced_l1_loss = reduce_mean(l1_loss, opt.nProcs)
        reduced_total_loss = reduce_mean(total_loss, opt.nProcs)

        epoch_l1_loss.update(reduced_l1_loss.item(), noisy.size(0))
        epoch_total_loss.update(reduced_total_loss.item(), noisy.size(0))

        if iteration % opt.print_freq == 0:
            ddp_logger_info(
                'Train epoch: [{:d}/{:d}]\titeration: [{:d}/{:d}]\tlr={:.6f}\tl1_loss={:.4f}\ttotal_loss={:.4f}'
                .format(epoch, opt.nEpochs, iteration, len(data_loader), scheduler.get_lr()[0], epoch_l1_loss.avg,
                            epoch_total_loss.avg), logger, opt.local_rank)

    ddp_writer_add_scalar('Train_L1_loss', epoch_l1_loss.avg, epoch, writer, opt.local_rank)
    ddp_writer_add_scalar('Train_total_loss', epoch_total_loss.avg, epoch, writer, opt.local_rank)
    ddp_writer_add_scalar('Learning_rate', scheduler.get_lr()[0], epoch, writer, opt.local_rank)
    ddp_logger_info(
        '||==> Train epoch: [{:d}/{:d}]\tlr={:.6f}\tl1_loss={:.4f}\ttotal_loss={:.4f}\tcost_time={:.4f}'
        .format(epoch, opt.nEpochs, scheduler.get_lr()[0], epoch_l1_loss.avg, epoch_total_loss.avg, time.time() - t0),
        logger, opt.local_rank)


def valid(opt, epoch, data_loader, model, criterion, logger, writer):
    t0 = time.time()
    model.eval()
    psnr_val = AverageMeter()
    loss_val = AverageMeter()

    for iteration, (noisy, target) in enumerate(data_loader):
        psnr = AverageMeter()
        noisy, target = noisy.cuda(opt.local_rank, non_blocking=True), target.cuda(opt.local_rank, non_blocking=True)
        with torch.no_grad():
            prediction = model(noisy)
            prediction = torch.clamp(prediction, 0.0, 1.0)

        loss = criterion(prediction, target)
        prediction = prediction.data.cpu().numpy().astype(np.float32)
        target = target.data.cpu().numpy().astype(np.float32)
        for i in range(prediction.shape[0]):
            psnr.update(compare_psnr(prediction[i, :, :, :], target[i, :, :, :], data_range=1.0), 1)

        dist.barrier()
        reduced_psnr = reduce_mean(torch.Tensor([psnr.avg]).cuda(opt.local_rank, non_blocking=True), opt.nProcs)
        reduced_loss = reduce_mean(loss, opt.nProcs)

        psnr_val.update(reduced_psnr.item(), prediction.shape[0])
        loss_val.update(reduced_loss.item(), prediction.shape[0])

    ddp_writer_add_scalar('Validation_PSNR', psnr_val.avg, epoch, writer, opt.local_rank)
    ddp_writer_add_scalar('Validation_loss', loss_val.avg, epoch, writer, opt.local_rank)
    ddp_logger_info('||==> Validation epoch: [{:d}/{:d}]\tval_PSNR={:.4f}\tval_loss={:.4f}\tcost_time={:.4f}'
                    .format(epoch, opt.nEpochs, psnr_val.avg, loss_val.avg, time.time() - t0),
                    logger, opt.local_rank)
    return psnr_val.avg


def main():
    parser = argparse.ArgumentParser(description='PyTorch image denoising')
    # dataset settings
    parser.add_argument('--data_set1', type=str, default='sidd', help='the exact dataset 1 we want to train on')
    parser.add_argument('--data_set2', type=str, default='renoir', help='the exact dataset 2 we want to train on')
    parser.add_argument('--data_set3', type=str, default='nind', help='the exact dataset 3 we want to train on')
    parser.add_argument('--data_set_test', type=str, default='rid2021', help='the exact dataset 4 we want to test on')
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/yangmingzhuo/processed', help='the dataset dir')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size: 32')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size, default=1')
    parser.add_argument('--test_patch_size', type=int, default=256, help='testing patch size, default=1')

    # training settings
    parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate. default=0.0002')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum learning rate. default=0.000001')
    parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight_decay')

    # model settings
    parser.add_argument('--model_type', type=str, default='ELU_UNet', help='the name of model')
    parser.add_argument('--pretrain_model', type=str, default='', help='pretrain model path')

    # general settings
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='id of gpus')
    parser.add_argument('--log_dir', default='./logs_v2/baseline', help='Location to save checkpoint models')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--print_freq', type=int, default=10, help='print freq')
    parser.add_argument('--exp_id', type=int, default=0, help='experiment')

    # distributed
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    opt = parser.parse_args()

    # initialize
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    logger = None
    writer = None
    psnr_best = 0
    epoch_best = 0

    # log setting
    if opt.local_rank == 0:
        log_folder = os.path.join(opt.log_dir,
                                  "model_{}_gpu_{}_ds_{}_{}_{}_td_{}_ps_{}_bs_{}_ep_{}_lr_{}_lr_min_{}_lam_exp_id_{}_ddp"
                                  .format(opt.model_type, opt.gpus, opt.data_set1, opt.data_set2, opt.data_set3,
                                          opt.data_set_test, opt.patch_size, opt.batch_size,
                                          opt.nEpochs, opt.lr, opt.lr_min, opt.exp_id))
        output_process(log_folder)
        checkpoint_folder = make_dir(os.path.join(log_folder, 'checkpoint'))
        writer = SummaryWriter(log_folder)
        logger = get_logger(log_folder, 'DGNet_log')
        logger.info(opt)

    # distributed init
    opt.nProcs = torch.cuda.device_count()
    dist.init_process_group(backend='nccl', world_size=opt.nProcs, rank=opt.local_rank)
    torch.cuda.set_device(device=opt.local_rank)

    # load dataset
    ddp_logger_info('Loading datasets {}, {}, {}, Batch Size: {}, Patch Size: {}'
                    .format(opt.data_set1, opt.data_set2, opt.data_set3, opt.batch_size, opt.patch_size),
                    logger, opt.local_rank)

    train_set = LoadMultiDataset(src_path1=os.path.join(opt.data_dir, opt.data_set1, 'train'),
                                 src_path2=os.path.join(opt.data_dir, opt.data_set2, 'train'),
                                 src_path3=os.path.join(opt.data_dir, opt.data_set3, 'train'),
                                 patch_size=opt.patch_size,
                                 train=True)
    train_sampler = DistributedSampler(train_set)
    train_data_loader = DataLoaderX(dataset=train_set, batch_size=opt.batch_size,
                                    num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    ddp_logger_info('Train dataset length: {} 1:{} 2:{} 3:{}'.format(len(train_data_loader), train_set.len1,
                                                                     train_set.len2, train_set.len3), logger, opt.local_rank)

    val_set = LoadDataset(src_path=os.path.join(opt.data_dir, opt.data_set_test, 'test'),
                          patch_size=opt.test_patch_size,
                          train=False)
    val_sampler = DistributedSampler(val_set)
    val_data_loader = DataLoaderX(dataset=val_set, batch_size=opt.test_batch_size, shuffle=False,
                                  num_workers=opt.num_workers, pin_memory=True, sampler=val_sampler)
    ddp_logger_info('Validation dataset length: {}'.format(len(val_data_loader)), logger, opt.local_rank)

    # load network
    ddp_logger_info('Building model {}'.format(opt.model_type), logger, opt.local_rank)
    model = ELD_UNet()
    ddp_logger_info("Push model to distribute data parallel!", logger, opt.local_rank)
    model.cuda(device=opt.local_rank)
    ddp_logger_info('model={}'.format(model), logger, opt.local_rank)
    model = DDP(model, device_ids=[opt.local_rank])

    # loss
    ddp_logger_info('Use L1 loss as criterion', logger, opt.local_rank)
    criterion = nn.L1Loss().cuda(device=opt.local_rank)

    # optimizer and scheduler
    warmup_epochs = 3
    t_max = opt.nEpochs - warmup_epochs
    ddp_logger_info('Optimizer: Adam Warmup epochs: {}, Learning rate: {}, Scheduler: CosineAnnealingLR, T_max: {}'
                    .format(warmup_epochs, opt.lr, t_max), logger, opt.local_rank)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ddp_logger_info('optimizer={}'.format(optimizer), logger, opt.local_rank)

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=opt.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    ddp_logger_info('scheduler={}'.format(scheduler), logger, opt.local_rank)

    # resume
    if opt.pretrain_model != '':
        model, start_epoch, optimizer, psnr_best = load_model(opt.pretrain_model, model,
                                                              optimizer, logger, opt.local_rank)
        start_epoch += 1
        for i in range(1, start_epoch):
            scheduler.step()
        ddp_logger_info('Resume start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]),
                        logger, opt.local_rank)
    else:
        start_epoch = opt.start_epoch
        ddp_logger_info(
            'Start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]), logger,
            opt.local_rank)

    # training
    for epoch in range(start_epoch, opt.nEpochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        scheduler.step()

        # training
        train(opt, epoch, model, train_data_loader, optimizer, scheduler, criterion, logger, writer)
        # validation
        psnr = valid(opt, epoch, val_data_loader, model, criterion, logger, writer)

        # save model
        if opt.local_rank == 0:
            if psnr >= psnr_best:
                psnr_best = psnr
                epoch_best = epoch
                save_model(os.path.join(checkpoint_folder, "model_best.pth"), epoch, model, optimizer, psnr_best, logger)
            save_model(os.path.join(checkpoint_folder, "model_latest.pth"), epoch, model, optimizer, psnr_best, logger)

        ddp_logger_info('||==> best_epoch = {}, best_psnr = {}'.format(epoch_best, psnr_best), logger, opt.local_rank)

    # generate evaluate_mat for SSIM validation
    # gen_mat(ELD_UNet(), os.path.join(checkpoint_folder, "model_best.pth"), checkpoint_folder, val_data_loader,
    #        opt.test_batch_size, opt.test_patch_size, logger)

    if opt.local_rank == 0:
        writer.close()


if __name__ == '__main__':
    main()
