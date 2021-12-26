import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ad_loss(discriminator_out, label):
    criterion_ce = nn.CrossEntropyLoss()
    loss = criterion_ce(discriminator_out, label)
    return loss


def get_patch_ad_loss(discriminator_out, label):
    criterion_ce = nn.CrossEntropyLoss()
    loss = criterion_ce(discriminator_out, label)
    return loss


def get_kl_loss(outputs, teacher_outputs, T=20.0):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    kl_loss = (T * T) * criterion_kl(F.log_softmax(outputs / T, dim=1),
                                                            F.softmax(teacher_outputs / T, dim=1))
    return kl_loss
