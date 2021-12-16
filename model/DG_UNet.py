import torch
import torch.nn as nn
import torch.nn.functional
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
    def __init__(self, max_iter):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        # self.batch_norm2d = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.gen_feature = nn.Sequential(
            self.conv1,
            # self.batch_norm2d,
            self.relu1,
            self.avg_pool,
        )

        self.fc1 = nn.Linear(512, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)

        self.relu2 = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)

        self.ad_net = nn.Sequential(
            self.fc1,
            self.relu2,
            self.drop_out,
            self.fc2
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        feature = self.gen_feature(feature)
        feature = feature.view(feature.size(0), -1)
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out

