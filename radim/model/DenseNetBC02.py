import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, nin, reps, k=12, bottleneck=True, nonlin=F.relu):
        super(DenseBlock, self).__init__()

        self.bottleneck = bottleneck
        self.nonlin = nonlin

        if self.bottleneck:
            self.bn_bottleneck = nn.ModuleList([nn.BatchNorm2d(nin + k * i) for i in range(reps)])
            self.conv_bottleneck = nn.ModuleList([nn.Conv2d(nin + k * i, 4 * k, kernel_size=(1, 1), stride=1, padding=0) for i in range(reps)])
            self.bn = nn.ModuleList([nn.BatchNorm2d(4 * k) for i in range(reps)])
            self.conv = nn.ModuleList([nn.Conv2d(4 * k, k, kernel_size=(3, 3), stride=1, padding=1) for i in range(reps)])
        else:
            self.bn = nn.ModuleList([nn.BatchNorm2d(nin + k * i) for i in range(reps)])
            self.conv = nn.ModuleList([nn.Conv2d(nin + k * i, k, kernel_size=(3, 3), stride=1, padding=1) for i in range(reps)])

    def forward(self, x):
        for i in range(len(self.conv)):
            xb = x
            if self.bottleneck:
                xb = self.conv_bottleneck[i](self.nonlin(self.bn_bottleneck[i](x)))
            xd = self.conv[i](self.nonlin(self.bn[i](xb)))
            x = torch.cat((x, xd), dim=1)

        return x


class TransitionLayer(nn.Module):
    def __init__(self, nin, nout):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm2d(nin)
        self.conv = nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.avg_pool(x)

        return x


class DenseNetBC(nn.Module):
    def __init__(self, use_cuda, input_image_channels_num, k=16, theta=0.5):
        super(DenseNetBC, self).__init__()

        self.use_cuda = use_cuda

        # theta is a compression factor
        self.theta = theta

        conv_init_mean = 0
        conv_init_std = .1

        xavier_normal_gain = 1

        self.bn_00 = nn.BatchNorm2d(input_image_channels_num)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        # Convolution 1
        input_count = input_image_channels_num
        output_count = 4 * k
        self.drop_01 = nn.Dropout(p=0.0)
        self.conv_01 = nn.Conv2d(input_count, output_count, kernel_size=(3, 3), stride=2, padding=0)
        nn.init.xavier_normal_(self.conv_01.weight, gain=xavier_normal_gain)
        self.bn_01 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        dense_in_dim = output_count
        dense_reps = 5
        dense_out_dim = dense_in_dim + dense_reps * k
        trans_out_dim = int(self.theta * dense_out_dim)
        self.dense_01 = DenseBlock(dense_in_dim, dense_reps, k=k)
        self.transition_01 = TransitionLayer(dense_out_dim, trans_out_dim)

        dense_in_dim = trans_out_dim
        dense_reps = 5
        dense_out_dim = dense_in_dim + dense_reps * k
        trans_out_dim = int(self.theta * dense_out_dim)
        self.dense_02 = DenseBlock(dense_in_dim, dense_reps, k=k)
        self.transition_02 = TransitionLayer(dense_out_dim, trans_out_dim)

        self.linear_00 = torch.nn.Linear(684, 10)
        nn.init.kaiming_normal_(self.linear_00.weight, mode='fan_in')

    def load_state_dict_partly(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                logging.getLogger('training').warning(' Did not find param %s' % name)
                continue
            if param.size() != own_state[name].size():
                logging.getLogger('training').warning(' Skipping incompatible sized param %s' % name)
                continue
            own_state[name].copy_(param)

    def forward(self, x):
        nonlin = F.relu

        x = self.bn_00(x)
        x = nonlin(self.bn_01(self.conv_01(self.drop_01(x))))

        x = self.dense_01(x)
        x = self.transition_01(x)

        x = self.dense_02(x)
        x = self.transition_02(x)

        x = x.reshape((x.size()[0], -1))

        x = nonlin(self.linear_00(x))

        x = F.softmax(x, dim=1)

        return x
