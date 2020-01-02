import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


class SeparableConv2D(nn.Module):
    def __init__(self, nin, nout, kernel_size=(3, 3), padding=1, stride=1):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, stride=stride, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MobileNet(nn.Module):
    def __init__(self, use_cuda, input_image_channels_num):
        super(MobileNet, self).__init__()

        self.use_cuda = use_cuda

        conv_init_mean = 0
        conv_init_std = .1

        xavier_normal_gain = 1

        self.bn_00 = nn.BatchNorm2d(input_image_channels_num)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        # Convolution 1
        input_count = input_image_channels_num
        output_count = 32
        self.drop_01 = nn.Dropout(p=0.0)
        self.conv_01 = SeparableConv2D(input_count, output_count, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.xavier_normal_(self.conv_01.depthwise.weight, gain=xavier_normal_gain)
        nn.init.xavier_normal_(self.conv_01.pointwise.weight, gain=xavier_normal_gain)
        self.bn_01 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        # Convolution 2
        input_count = output_count
        output_count = 64
        self.drop_02 = nn.Dropout(p=0.1)
        self.conv_02 = SeparableConv2D(input_count, output_count, kernel_size=(3, 3), stride=2, padding=1)
        nn.init.xavier_normal_(self.conv_02.depthwise.weight, gain=xavier_normal_gain)
        nn.init.xavier_normal_(self.conv_02.pointwise.weight, gain=xavier_normal_gain)
        self.bn_02 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_02.weight, conv_init_mean, conv_init_std)

        # Convolution 3-8
        reps = 5
        input_count = [output_count, 128, 128, 128, 128]
        output_count = 128
        self.drop_03_08 = nn.ModuleList([nn.Dropout(p=0.1) for i in range(reps)])
        self.conv_03_08 = nn.ModuleList([nn.Conv2d(input_count[i], output_count, kernel_size=(3, 3), stride=1, padding=1) for i in range(reps)])
        for i in range(reps):
            nn.init.xavier_normal_(self.conv_03_08[i].weight, gain=xavier_normal_gain)
        self.bn_03_08 = nn.ModuleList([nn.BatchNorm2d(output_count) for i in range(reps)])
        for i in range(reps):
            nn.init.normal_(self.bn_03_08[i].weight, conv_init_mean, conv_init_std)

        # Convolution 9
        input_count = output_count
        output_count = 128
        self.drop_09 = nn.Dropout(p=0.2)
        self.conv_09 = SeparableConv2D(input_count, output_count, kernel_size=(3, 3), stride=2, padding=0)
        nn.init.xavier_normal_(self.conv_09.depthwise.weight, gain=xavier_normal_gain)
        nn.init.xavier_normal_(self.conv_09.pointwise.weight, gain=xavier_normal_gain)
        self.bn_09 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_09.weight, conv_init_mean, conv_init_std)

        # Convolution 10
        input_count = output_count
        output_count = 256
        self.drop_10 = nn.Dropout(p=0.3)
        self.conv_10 = SeparableConv2D(input_count, output_count, kernel_size=(3, 3), stride=2, padding=0)
        nn.init.xavier_normal_(self.conv_10.depthwise.weight, gain=xavier_normal_gain)
        nn.init.xavier_normal_(self.conv_10.pointwise.weight, gain=xavier_normal_gain)
        self.bn_10 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_10.weight, conv_init_mean, conv_init_std)

        self.linear_00 = torch.nn.Linear(1024, 10)
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
        nonlin = F.elu

        x = self.bn_00(x)
        x = nonlin(self.bn_01(self.conv_01(self.drop_01(x))))
        x = nonlin(self.bn_02(self.conv_02(self.drop_02(x))))

        for i in range(5):
            x = nonlin(self.bn_03_08[i](self.conv_03_08[i](self.drop_03_08[i](x))))

        x = nonlin(self.bn_09(self.conv_09(self.drop_09(x))))
        x = nonlin(self.bn_10(self.conv_10(self.drop_10(x))))

        x = x.reshape((x.size()[0], -1))

        x = nonlin(self.linear_00(x))

        x = F.softmax(x, dim=1)

        return x
