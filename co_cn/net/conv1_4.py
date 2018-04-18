# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init


class conv1_4(nn.Module):
    def __init__(self,
                 in_channels):
        super(conv1_4, self).__init__()

        stride = 1
        kernel_size = 1
        self.conv_1 =  nn.Conv2d(in_channels, 64, kernel_size= 1,
                                padding=(0, 0),
                                stride=stride,
                                bias=False)
        self.conv_2 =  nn.Conv2d(64, 32, kernel_size= (3, 1),
                                padding=(0, 0),
                                stride=stride,
                                bias=False)
        self.conv_3 =  nn.Sequential(nn.Conv2d(25, 32, kernel_size= (3, 3),
                                padding=(0, 0),
                                stride=stride,
                                bias=False),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_4 =  nn.Sequential(nn.Conv2d(32, 64, kernel_size= (3, 3),
                                padding=(0, 0),
                                stride=stride,
                                bias=False),
                                nn.MaxPool2d(kernel_size=2, stride=2))

        self.bn_1 = nn.BatchNorm2d(64)
        self.bn_2 = nn.BatchNorm2d(32)
        self.bn_3 = nn.BatchNorm2d(32)
        self.bn_4 = nn.BatchNorm2d(64)

        conv_init(self.conv_1)
        conv_init(self.conv_2)
        for m in self.conv_3:
            if isinstance(m, nn.Conv2d):
                conv_init(m)
        for m in self.conv_4:
            if isinstance(m, nn.Conv2d):
                conv_init(m)


    def forward(self, x):
        N, C, T, V = x.size()
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = x.permute(0, 3, 2, 1).contiguous()
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        return x