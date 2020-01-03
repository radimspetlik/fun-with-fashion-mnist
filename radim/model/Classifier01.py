import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, use_cuda, input_image_channels_num):
        super(Classifier, self).__init__()

        self.use_cuda = use_cuda

        conv_init_mean = 0
        conv_init_std = .1

        xavier_normal_gain = 1

        self.bn_00 = nn.BatchNorm2d(input_image_channels_num)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        # Convolution 1
        input_count = input_image_channels_num
        output_count = 32
        self.drop_01 = nn.Dropout(p=0.1)
        self.conv_01 = nn.Conv2d(input_count, output_count, kernel_size=(3, 3), stride=2, padding=0)
        nn.init.xavier_normal_(self.conv_01.weight, gain=xavier_normal_gain)
        self.bn_01 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        # Convolution 2
        input_count = output_count
        output_count = 64
        self.drop_02 = nn.Dropout(p=0.2)
        self.conv_02 = nn.Conv2d(input_count, output_count, kernel_size=(3, 3), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_02.weight, gain=xavier_normal_gain)
        self.bn_02 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_02.weight, conv_init_mean, conv_init_std)

        # Convolution 2
        input_count = output_count
        output_count = 128
        self.drop_03 = nn.Dropout(p=0.3)
        self.conv_03 = nn.Conv2d(input_count, output_count, kernel_size=(3, 3), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_03.weight, gain=xavier_normal_gain)
        self.bn_03 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_03.weight, conv_init_mean, conv_init_std)

        self.linear_00 = torch.nn.Linear(10368, 100)
        nn.init.kaiming_normal_(self.linear_00.weight, mode='fan_in')

        self.linear_01 = torch.nn.Linear(100, 10)
        nn.init.kaiming_normal_(self.linear_01.weight, mode='fan_in')

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
        x = nonlin(self.bn_03(self.conv_03(self.drop_03(x))))

        x = x.reshape((x.size()[0], -1))

        x = nonlin(self.linear_00(x))
        x = self.linear_01(x)

        x = F.softmax(x, dim=1)

        return x
