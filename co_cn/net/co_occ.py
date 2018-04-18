import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .net import import_class

from .conv1_4 import conv1_4
from .conv5_fc8 import conv5_fc8

class Model(nn.Module):
    """ Spatial temporal graph convolutional networks
                        for skeleton-based action recognition.

    Input shape:
        Input shape should be (N, C, T, V, M)
        where N is the number of samples,
              C is the number of input channels,
              T is the length of the sequence,
              V is the number of joints or graph nodes,
          and M is the number of people.
    
    Arguments:
        About shape:
            channel (int): Number of channels in the input data
            num_class (int): Number of classes for classification
            window_size (int): Length of input sequence
            num_point (int): Number of joints or graph nodes
            num_person (int): Number of people
        About net:
            use_data_bn: If true, the data will first input to a batch normalization layer
            backbone_config: The structure of backbone networks
        About graph convolution:
            graph: The graph of skeleton, represtented by a adjacency matrix
            graph_args: The arguments of graph
            mask_learning: If true, use mask matrixes to reweight the adjacency matrixes
            use_local_bn: If true, each node in the graph have specific parameters of batch normalzation layer
        About temporal convolution:
            multiscale: If true, use multi-scale temporal convolution
            temporal_kernel_size: The kernel size of temporal convolution
            dropout: The drop out rate of the dropout layer in front of each temporal convolution layer

    """

    def __init__(self,
                 channel,
                 num_class,
                 window_size,
                 num_point,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A).float().cuda(0)

        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale

        # Different bodies share batchNorma parameters or not
        self.M_dim_bn = True

        if self.M_dim_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)
            self.diff_data_bn = nn.BatchNorm1d(channel * num_point * num_person)
        else:
            self.data_bn = nn.BatchNorm1d(channel * num_point)
            self.diff_data_bn = nn.BatchNorm1d(channel * num_point)

        kwargs = dict(
            dropout=dropout,
            kernel_size=temporal_kernel_size)

        # head
        self.conv_frame = conv1_4(3)
        self.conv_diff = conv1_4(3)

        self.tail = conv5_fc8()

        self.fc7 = nn.Linear(4608, 256)
        self.fc8 = nn.Linear(256, num_class)
        # conv_init(self.fcn)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x1 = x[:, :, 0:T-1]
        x2 = x[:, :, 1:T]

        x_diff = x2-x1
        x_diff = torch.cat((x_diff, x_diff[:, :, -1].unsqueeze(2)), 2)

        # data bn
        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)

            x = self.data_bn(x)
            # to (N*M, C, T, V)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # data bn
        if self.use_data_bn:
            if self.M_dim_bn:
                x_diff = x_diff.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x_diff = x_diff.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)

            x_diff = self.diff_data_bn(x_diff)
            # to (N*M, C, T, V)
            x_diff = x_diff.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x_diff = x_diff.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # model
        x1 = self.conv_diff(x_diff)
        x2 = self.conv_frame(x)
        x = torch.cat((x1, x2), 1)
        x = self.tail(x)

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).permute(0, 2, 3, 1).contiguous()#
        x = F.max_pool2d(x, kernel_size=(1,2)).view(N, -1)

        # C fcn
        x = self.fc8(F.relu(self.fc7(x)))
        x = x.view(N, self.num_class)

        return x
