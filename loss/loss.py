import torch
import torch.nn as nn


def get_ad_loss(discriminator_out, criterion, label):
    # generate ad_label
    ad_label1_index = torch.LongTensor(label[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    ad_label2_index = torch.LongTensor(label[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(label[2], 1).fill_(2)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label1, ad_label2, ad_label3], dim=0).view(-1)

    loss = criterion(discriminator_out, ad_label)
    return loss
