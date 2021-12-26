import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ad_loss(discriminator_out, label):
    criterion_ce = nn.CrossEntropyLoss()
    loss = criterion_ce(discriminator_out, label)
    return loss


def get_patch_ad_loss(discriminator_out, label, patch_num):
    patch_label = torch.cat([label for i in range(patch_num)], 1)
    print(patch_label)
    patch_label = patch_label.view(-1, 1)
    print(patch_label)
    criterion_ce = nn.CrossEntropyLoss()
    loss = criterion_ce(discriminator_out, patch_label)
    return loss


def get_kl_loss(outputs, teacher_outputs, T=20.0):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    kl_loss = (T * T) * criterion_kl(F.log_softmax(outputs / T, dim=1),
                                                            F.softmax(teacher_outputs / T, dim=1))
    return kl_loss
