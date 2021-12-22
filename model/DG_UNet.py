import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
from torchvision.models import resnet


class GRL(torch.autograd.Function):
    def __init__(self, max_iter):
        self.iter_num = 0
        self.alpha = 10.0
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = 2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) \
                - (self.high - self.low) + self.low
        return -coeff * gradOutput


# model
# # v0
# class Discriminator_model_v1(nn.Module):
#     def __init__(self, in_channels=3, domain_num=3, start_channel=32):
#         super(Discriminator_model_v1, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, start_channel, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(start_channel, start_channel*2, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.conv3 = nn.Conv2d(start_channel*2, start_channel*4, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#         self.conv4 = nn.Conv2d(start_channel*4, start_channel*8, kernel_size=3, stride=1, padding=1)
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv5 = nn.Conv2d(start_channel*8, start_channel*16, kernel_size=3, stride=1, padding=1)
#         self.pool5 = nn.MaxPool2d(kernel_size=2)
#         self.conv6 = nn.Conv2d(start_channel*16, start_channel*16, kernel_size=3, stride=1, padding=1)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc7 = nn.Linear(start_channel*16, start_channel*16)
#         self.dropout7 = nn.Dropout(0.5)
#         self.fc8 = nn.Linear(start_channel*16, start_channel*16)
#         self.dropout8 = nn.Dropout(0.5)
#         self.fc9 = nn.Linear(start_channel*16, domain_num)
#
#     def forward(self, x):
#         conv1 = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
#         pool1 = self.pool1(conv1)
#         conv2 = F.leaky_relu(self.conv2(pool1), 0.2, inplace=True)
#         pool2 = self.pool2(conv2)
#         conv3 = F.leaky_relu(self.conv3(pool2), 0.2, inplace=True)
#         pool3 = self.pool3(conv3)
#         conv4 = F.leaky_relu(self.conv4(pool3), 0.2, inplace=True)
#         pool4 = self.pool4(conv4)
#
#         conv5 = F.leaky_relu(self.conv5(pool4), 0.2, inplace=True)
#         pool5 = self.pool5(conv5)
#         conv6 = F.leaky_relu(self.conv6(pool5), 0.2, inplace=True)
#
#         avg_pool = self.avg_pool(conv6)
#         avg_pool = avg_pool.view(avg_pool.size(0), -1)
#         fc7 = F.leaky_relu(self.fc7(avg_pool), 0.2, inplace=True)
#         fc7 = self.dropout7(fc7)
#         fc8 = F.leaky_relu(self.fc8(fc7), 0.2, inplace=True)
#         fc8 = self.dropout8(fc8)
#         adversarial_out = self.fc9(fc8)
#
#         return adversarial_out


# v1
class Discriminator_model_v1(nn.Module):
    def __init__(self, in_channels=3, domain_num=3, start_channel=64):
        super(Discriminator_model_v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, start_channel, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(start_channel, start_channel, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(start_channel, start_channel*2, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(start_channel*2, start_channel*2, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(start_channel*2, start_channel*4, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(start_channel*4, start_channel*4, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(start_channel*4, start_channel*8, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(start_channel*8, start_channel*8, kernel_size=3, stride=2, padding=1)
        self.conv5_2 = nn.Conv2d(start_channel*8, start_channel*16, kernel_size=3, stride=1, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc6 = nn.Linear(start_channel*16, start_channel*8)
        self.dropout6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(start_channel*8, start_channel*8)
        self.dropout7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(start_channel*8, domain_num)

    def forward(self, x):
        conv1 = F.leaky_relu(self.conv1(x), 0.2, inplace=True)

        conv2_1 = F.leaky_relu(self.conv2_1(conv1), 0.2, inplace=True)
        conv2_2 = F.leaky_relu(self.conv2_2(conv2_1), 0.2, inplace=True)

        conv3_1 = F.leaky_relu(self.conv3_1(conv2_2), 0.2, inplace=True)
        conv3_2 = F.leaky_relu(self.conv3_2(conv3_1), 0.2, inplace=True)

        conv4_1 = F.leaky_relu(self.conv4_1(conv3_2), 0.2, inplace=True)
        conv4_2 = F.leaky_relu(self.conv4_2(conv4_1), 0.2, inplace=True)

        conv5_1 = F.leaky_relu(self.conv5_1(conv4_2), 0.2, inplace=True)
        conv5_2 = F.leaky_relu(self.conv5_2(conv5_1), 0.2, inplace=True)

        avg_pool = self.avg_pool(conv5_2)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)

        fc6 = F.leaky_relu(self.fc6(avg_pool), 0.2, inplace=True)
        fc6 = self.dropout6(fc6)

        fc7 = F.leaky_relu(self.fc7(fc6), 0.2, inplace=True)
        fc7 = self.dropout8(fc7)

        fc8 = self.fc8(fc7)
        return fc8


# v2
class Discriminator_model_v2(nn.Module):
    def __init__(self, in_channels=3, domain_num=3, start_channel=64):
        super(Discriminator_model_v2, self).__init__()
        norm = spectral_norm
        self.conv1 = nn.Conv2d(in_channels, start_channel, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = norm(nn.Conv2d(start_channel, start_channel, kernel_size=3, stride=2, padding=1))
        self.conv2_2 = norm(nn.Conv2d(start_channel, start_channel*2, kernel_size=3, stride=1, padding=1))

        self.conv3_1 = norm(nn.Conv2d(start_channel*2, start_channel*2, kernel_size=3, stride=2, padding=1))
        self.conv3_2 = norm(nn.Conv2d(start_channel*2, start_channel*4, kernel_size=3, stride=1, padding=1))

        self.conv4_1 = norm(nn.Conv2d(start_channel*4, start_channel*4, kernel_size=3, stride=2, padding=1))
        self.conv4_2 = norm(nn.Conv2d(start_channel*4, start_channel*8, kernel_size=3, stride=1, padding=1))

        self.conv5_1 = norm(nn.Conv2d(start_channel*8, start_channel*8, kernel_size=3, stride=2, padding=1))
        self.conv5_2 = norm(nn.Conv2d(start_channel*8, start_channel*16, kernel_size=3, stride=1, padding=1))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc6 = norm(nn.Linear(start_channel*16, start_channel*8))
        self.dropout6 = nn.Dropout(0.5)
        self.fc7 = norm(nn.Linear(start_channel*8, start_channel*8))
        self.dropout7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(start_channel*8, domain_num)

    def forward(self, x):
        conv1 = F.leaky_relu(self.conv1(x), 0.2, inplace=True)

        conv2_1 = F.leaky_relu(self.conv2_1(conv1), 0.2, inplace=True)
        conv2_2 = F.leaky_relu(self.conv2_2(conv2_1), 0.2, inplace=True)

        conv3_1 = F.leaky_relu(self.conv3_1(conv2_2), 0.2, inplace=True)
        conv3_2 = F.leaky_relu(self.conv3_2(conv3_1), 0.2, inplace=True)

        conv4_1 = F.leaky_relu(self.conv4_1(conv3_2), 0.2, inplace=True)
        conv4_2 = F.leaky_relu(self.conv4_2(conv4_1), 0.2, inplace=True)

        conv5_1 = F.leaky_relu(self.conv5_1(conv4_2), 0.2, inplace=True)
        conv5_2 = F.leaky_relu(self.conv5_2(conv5_1), 0.2, inplace=True)

        avg_pool = self.avg_pool(conv5_2)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)

        fc6 = F.leaky_relu(self.fc6(avg_pool), 0.2, inplace=True)
        fc6 = self.dropout6(fc6)

        fc7 = F.leaky_relu(self.fc7(fc6), 0.2, inplace=True)
        fc7 = self.dropout8(fc7)

        fc8 = self.fc8(fc7)
        return fc8


# Encoder
class Discriminator_encoder_v1(nn.Module):
    def __init__(self, max_iter, in_channels=512, domain_num=3, start_channel=32):
        super(Discriminator_encoder_v1, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc2 = nn.Linear(in_channels, start_channel*16)
        self.dropout2 = nn.Dropout(0.5)
        self.grl_layer3 = GRL(max_iter)
        self.fc4 = nn.Linear(start_channel*16, start_channel*16)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(start_channel*16, domain_num)

    def forward(self, x):
        pool1 = self.avg_pool1(x)
        pool1 = pool1.view(pool1.size(0), -1)

        fc2 = F.leaky_relu(self.fc2(pool1), 0.2, inplace=True)
        fc2 = self.dropout2(fc2)
        grl3 = self.grl_layer3(fc2)
        fc4 = F.leaky_relu(self.fc4(grl3), 0.2, inplace=True)
        fc4 = self.dropout4(fc4)
        adversarial_out = self.fc5(fc4)

        return adversarial_out


class Discriminator_encoder_v2(nn.Module):
    def __init__(self, in_channels=512, domain_num=3):
        super(Discriminator_encoder_v2, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(in_channels, in_channels)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(in_channels, in_channels)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(in_channels, domain_num)

    def forward(self, x):
        conv1 = F.leaky_relu(self.conv1_1(x), 0.2, inplace=True)
        pool1 = self.pool1(conv1)
        conv1 = F.leaky_relu(self.conv1_2(pool1), 0.2, inplace=True)

        pool2 = self.avg_pool2(conv1)
        pool2 = pool2.view(pool2.size(0), -1)
        fc3 = F.leaky_relu(self.fc3(pool2), 0.2, inplace=True)
        fc3 = self.dropout3(fc3)
        fc4 = F.leaky_relu(self.fc4(fc3), 0.2, inplace=True)
        fc4 = self.dropout4(fc4)
        adversarial_out = self.fc5(fc4)

        return adversarial_out


# print(Discriminator_model_v1())
# print(Discriminator_model_v2())