from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


class BottleneckBlock(nn.Module):
    # factor of bottleneck
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality):
        super(BottleneckBlock, self).__init__()

        bottleneck_channels = cardinality * out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)

        y += self.shortcut(x)
        y = F.relu(y)  # apply ReLU after addition
        return y


class DeconvBottleneckBlock(nn.Module):
    # factor of bottleneck
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality):
        super(DeconvBottleneckBlock, self).__init__()

        bottleneck_channels = cardinality * out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.ConvTranspose2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=stride - 1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    output_padding=stride - 1,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y)  # apply ReLU after addition
        return y


class ResEncoder(nn.Module):
    def __init__(self, output_dim, in_channels=3, cardinality=1, resolution=256, base_channels=64, reduce=False, reduce_to=None, **ignorekwargs):
        super(ResEncoder, self).__init__()

        # depth = config['depth']
        self.cardinality = cardinality
        self.output_dim = output_dim
        self.in_channels = in_channels

        # 2: initial conv layer + last fc layer
        # 9: 3 layers/block * 3 stages
        # --> 9 * blocks/stage = layers
        # n_blocks_per_stage = (depth - 2) // 9
        # assert n_blocks_per_stage * 9 + 2 == depth

        block = BottleneckBlock
        n_channels = [
            base_channels, base_channels * block.expansion,
                           base_channels * 2 * block.expansion,
                           base_channels * 4 * block.expansion,
                           base_channels * 8 * block.expansion
        ]

        self.conv = nn.Conv2d(
            in_channels,
            n_channels[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(n_channels[0], n_channels[1], 3, stride=1)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2], 4, stride=2)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3], 6, stride=2)
        self.stage4 = self._make_stage(n_channels[3], n_channels[4], 3, stride=2)

        # compute conv feature size
        self.reduce = reduce

        if reduce:
            assert reduce_to is not None
            self.reduce = reduce
            self.fc = nn.Linear(output_dim*output_dim*2048, reduce_to)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                s = stride
            else:
                s = 1
                in_channels = out_channels

            stage.add_module(
                block_name,
                BottleneckBlock(
                    in_channels,
                    out_channels,
                    s,
                    self.cardinality))
        return stage

    def _forward_conv(self, x):
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        # x = F.relu(self.bn(self.conv(x)))
        x = self.conv(x)
        #         print(x.shape)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        x = self.bn(x)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        x = F.relu(x)
        # print(torch.cuda.memory_allocated("cuda:1")/1024//1024, torch.cuda.memory_reserved("cuda:1")/1024//1024)
        x = self.maxpool(x)

        #         print(x.shape)
        x = self.stage1(x)
        #         print(x.shape)
        x = self.stage2(x)
        #         print(x.shape)
        x = self.stage3(x)
        #         print(x.shape)
        x = self.stage4(x)
        #         print(x.shape)
        x = F.adaptive_avg_pool2d(x, output_size=self.output_dim)
        #         print(x.shape)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        if self.reduce:
            x = self.fc(x)
        return x


class ResDecoder(nn.Module):
    def __init__(self, output_dim, in_channels = 3, cardinality = 1, resolution=256, base_channels=64, reduce = False, reduce_to = None, **ignorekwargs):
        super(ResDecoder, self).__init__()

        self.output_dim = output_dim
        self.cardinality = cardinality
        self.conv_shape = [7,7]
        self.reduce = reduce
        self.in_channels = in_channels
        # 2: initial conv layer + last fc layer
        # 9: 3 layers/block * 3 stages
        # --> 9 * blocks/stage = layers
        # n_blocks_per_stage = (depth - 2) // 9
        # assert n_blocks_per_stage * 9 + 2 == depth

        block = DeconvBottleneckBlock
        n_channels = [
            base_channels, base_channels * block.expansion,
                           base_channels * 2 * block.expansion,
                           base_channels * 4 * block.expansion,
                           base_channels * 8 * block.expansion
        ]
        n_channels.reverse()

        self.stage1 = self._make_stage(n_channels[0], n_channels[1], 3, stride=2)
        self.stage2 = self._make_stage(n_channels[1], n_channels[2], 6, stride=2)
        self.stage3 = self._make_stage(n_channels[2], n_channels[3], 4, stride=2)
        self.stage4 = self._make_stage(n_channels[3], n_channels[4], 3, stride=1)

        self.conv_out = nn.ConvTranspose2d(
            n_channels[-1],
            self.in_channels,
            kernel_size=6,
            stride=2,
            padding=2,
            bias=False)

        if self.reduce:
            self.fc = nn.Linear(reduce_to, output_dim*output_dim*2048)

            # initialize weights
        self.apply(initialize_weights)

    #         with torch.no_grad():
    #             self.feature_size = self.forward(
    #                 torch.zeros(*self.input_shape)).shape
    #             print('Decoder:', self.feature_size)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                s = stride
            else:
                s = 1
                in_channels = out_channels

            stage.add_module(
                block_name,
                DeconvBottleneckBlock(
                    in_channels,
                    out_channels,
                    s,
                    self.cardinality))
        return stage

    def _forward_conv(self, x):
        #         print(x.shape)
        x = self.stage1(x)
        #         print(x.shape)
        x = self.stage2(x)
        #         print(x.shape)
        x = self.stage3(x)
        #         print(x.shape)
        x = self.stage4(x)
        #         print(x.shape)
        return x

    def forward(self, x):

        if self.reduce:
            x = self.fc(x)

        x = x.view(x.size(0), -1, self.output_dim, self.output_dim)
        x = F.interpolate(x, size=self.conv_shape, align_corners=False, mode='bilinear')

        x = self._forward_conv(x)
        # x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.interpolate(x, scale_factor=2, align_corners=False, mode='bilinear')
        x = self.conv_out(x)

        return x

