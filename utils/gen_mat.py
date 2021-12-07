import os
import numpy as np
import torch
import scipy.io as sio
from utils.util import *
from utils.checkpoint import *
import hdf5storage


def gen_mat(model, pretrain_model, dst_path, data_loader, batch_size, patch_size, logger):
    model, psnr_best = load_single_model(pretrain_model, model, logger)
    model.cuda()
    model.eval()
    store_data_prediction = np.zeros((len(data_loader) * batch_size, patch_size, patch_size, 3), float)
    store_data_clean = np.zeros((len(data_loader) * batch_size, patch_size, patch_size, 3), float)

    num = 0
    for iteration, (noisy, target) in enumerate(data_loader):
        noisy, target = noisy.cuda(), target.cuda()
        with torch.no_grad():
            prediction = torch.clamp(model(noisy), 0.0, 1.0)
            prediction = prediction.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            target = target.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            for i in range(prediction.shape[0]):
                store_data_prediction[iteration * batch_size + i, :, :, :] = prediction[i]
                store_data_clean[iteration * batch_size + i, :, :, :] = target[i]
                num += 1

    store_data_clean.resize((num, patch_size, patch_size, 3))
    store_data_prediction.resize((num, patch_size, patch_size, 3))
    if os.path.exists(os.path.join(dst_path, 'denoised_evaluate_mat.mat')):
        os.remove(os.path.join(dst_path, 'denoised_evaluate_mat.mat'))
    if os.path.exists(os.path.join(dst_path, 'clean_evaluate_mat.mat')):
        os.remove(os.path.join(dst_path, 'clean_evaluate_mat.mat'))
    hdf5storage.savemat(os.path.join(dst_path, 'denoised_evaluate_mat'), {"denoised": store_data_prediction}, do_compression=True, format='7.3')
    hdf5storage.savemat(os.path.join(dst_path, 'clean_evaluate_mat'), {"clean": store_data_clean}, do_compression=True, format='7.3')
    logger.info('Mat with best model and {} patches generated!'.format(num))
