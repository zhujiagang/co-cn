# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init


class conv5_fc8(nn.Module):
    def __init__(self):
        super(conv5_fc8, self).__init__()
        stride = 1

        self.conv_5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 3),
                                padding=(1, 1),
                                stride=stride,
                                bias=False),
                      nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_6 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1),
                                stride=stride,
                                bias=False),
                      nn.MaxPool2d(kernel_size=2, stride=2))

        self.bn_5 = nn.BatchNorm2d(128)
        self.bn_6 = nn.BatchNorm2d(256)

        for m in self.conv_5:
            if isinstance(m, nn.Conv2d):
                conv_init(m)
        for m in self.conv_6:
            if isinstance(m, nn.Conv2d):
                conv_init(m)


    def forward(self, x):
        N, C, T, V = x.size()

        x = F.relu(self.bn_5(self.conv_5(x)))
        x = F.relu(self.bn_6(self.conv_6(x)))
        return x