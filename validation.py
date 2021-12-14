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
from utils.gen_mat import *
import cv2
from skimage import img_as_ubyte


def valid(opt, data_loader, model, logger):
    t0 = time.time()
    model.eval()
    psnr_val = AverageMeter()

    for iteration, (noisy, target) in enumerate(data_loader):
        noisy_num = torch.clamp(noisy, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
        with torch.no_grad():
            noisy, target = noisy.cuda(), target.cuda()
            prediction = model(noisy)
            prediction = torch.clamp(prediction, 0.0, 1.0)

        prediction_num = torch.clamp(prediction, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
        prediction = prediction.data.cpu().numpy().astype(np.float32)
        target_num = torch.clamp(target, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
        target = target.data.cpu().numpy().astype(np.float32)

        for i in range(prediction.shape[0]):
            psnr_val.update(compare_psnr(prediction[i, :, :, :], target[i, :, :, :], data_range=1.0), 1)

            if opt.save_imgs == 1 and iteration % 10 == 0:
                img = np.concatenate([noisy_num[i, :, :, :], prediction_num[i, :, :, :], target_num[i, :, :, :]], 1)
                save_file = os.path.join(os.path.dirname(opt.pretrained), '%04d_%02d.png' % (iteration + 1, i + 1))
                cv2.imwrite(save_file, cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_RGB2BGR))

    logger.info('||==> val_PSNR={:.4f}\tcost_time={:.4f}'
                .format(psnr_val.avg, time.time() - t0))
    return psnr_val.avg


def main():
    # testing settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size, default=1')
    parser.add_argument('--data_set', type=str, default='sidd', help='the exact dataset we want to train on')
    parser.add_argument('--pretrained', type=str, help="Checkpoints directory,  (default:./checkpoints)")
    parser.add_argument('--save_imgs', type=int, default=0, help='whether to save imgs')

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
    make_dir(opt.log_dir)
    log_folder = os.path.join(opt.log_dir)
    logger = get_logger(log_folder, 'DGNet_log')

    # load dataset
    logger.info("Load data from: {}".format(os.path.join(opt.data_dir, 'processed', opt.data_set, 'test')))
    val_set = LoadDataset(src_path=os.path.join(opt.data_dir, 'processed', opt.data_set, 'test'),
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
    model, psnr_best = load_single_model(opt.pretrained, model, logger)

    valid(opt, val_data_loader, model, logger)
    dst_folder = make_dir(os.path.join(os.path.dirname(opt.pretrained), opt.data_set))
    # gen_mat(ELD_UNet(), opt.pretrained, dst_folder, val_data_loader, opt.test_batch_size, opt.patch_size, logger)


if __name__ == '__main__':
    main()
