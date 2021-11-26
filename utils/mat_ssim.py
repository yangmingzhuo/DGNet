import os
import numpy as np
import torch
from utils.util import *
from utils.checkpoint import *
import scipy.io as sio


def gen_mat(model, pretrain_model, dst_path, data_loader, logger):
    model, psnr_best = load_single_model(pretrain_model, model, logger)
    logger.info('Resume start epoch: {}, Learning rate:{:.6f}'.format(start_epoch, scheduler.get_lr()[0]))
    store_data_prediction = np.zeros((len(data_loader) * 8, 256, 256, 3), float)
    store_data_clean = np.zeros((len(data_loader) * 8, 256, 256, 3), float)
    for iteration, (noisy, target) in enumerate(data_loader):
        noisy, target = noisy.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(noisy)
            prediction = prediction.data.cpu().numpy().astype(np.float32)
            target = target.data.cpu().numpy().astype(np.float32)
            for i in range(prediction.shape[0]):
                store_data_prediction[iteration * prediction.shape[0] + i, :, :, :] = prediction
                store_data_clean[iteration * prediction.shape[0] + i, :, :, :] = target

    sio.savemat(os.path.join(dst_path, 'denoised.mat'), {"denoised": store_data_prediction, })
    sio.savemat(os.path.join(dst_path, 'clean.mat'), {"clean": store_data_clean, })
