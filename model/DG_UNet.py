import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRL(torch.autograd.Function):
    def __init__(self, max_iter):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput


class Discriminator(nn.Module):
    def __init__(self, max_iter, in_channels=3, domain_num=3, start_channel=32):
        super(Discriminator, self).__init__()
        self.grl_layer = GRL(max_iter)
        self.conv1_1 = nn.Conv2d(in_channels, start_channel, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(start_channel, start_channel, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(start_channel, start_channel*2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(start_channel*2, start_channel*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(start_channel*2, start_channel*4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(start_channel*4, start_channel*4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(start_channel*4, start_channel*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(start_channel*8, start_channel*8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc5 = nn.Linear(start_channel*8, start_channel*16)
        self.dropout5 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(start_channel*16, start_channel*16)
        self.dropout6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(start_channel*16, domain_num)

    def forward(self, x):
        x = self.grl_layer(x)
        conv1 = F.leaky_relu(self.conv1_1(x), 0.2, inplace=True)
        conv1 = F.leaky_relu(self.conv1_2(conv1), 0.2, inplace=True)
        pool1 = self.pool1(conv1)

        conv2 = F.leaky_relu(self.conv2_1(pool1), 0.2, inplace=True)
        conv2 = F.leaky_relu(self.conv2_2(conv2), 0.2, inplace=True)
        pool2 = self.pool1(conv2)

        conv3 = F.leaky_relu(self.conv3_1(pool2), 0.2, inplace=True)
        conv3 = F.leaky_relu(self.conv3_2(conv3), 0.2, inplace=True)
        pool3 = self.pool1(conv3)

        conv4 = F.leaky_relu(self.conv4_1(pool3), 0.2, inplace=True)
        conv4 = F.leaky_relu(self.conv4_2(conv4), 0.2, inplace=True)
        pool4 = self.pool1(conv4)

        pool4 = self.avg_pool(pool4)
        pool4 = pool4.view(pool4.size(0), -1)
        fc5 = F.leaky_relu(self.fc5(pool4), 0.2, inplace=True)
        fc5 = self.dropout5(fc5)

        fc6 = F.leaky_relu(self.fc6(fc5), 0.2, inplace=True)
        fc6 = self.dropout6(fc6)
        adversarial_out = self.fc7(fc6)
        return adversarial_out



