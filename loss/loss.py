import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ad_loss(discriminator_out, label, criterion_ce):
    loss = criterion_ce(discriminator_out, label)
    return loss


def get_kl_loss(outputs, teacher_outputs, criterion_kl, T=20.0):
    kl_loss = (T * T) * criterion_kl(F.log_softmax(outputs / T, dim=1),
                                                            F.softmax(teacher_outputs / T, dim=1))
    return kl_loss
