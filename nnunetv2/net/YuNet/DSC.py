import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torchsummary import summary
import numpy as np

class ConvBatchNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 pad = 0,dilation=1,
                 groups=1, has_bn=True, has_relu=True, inplace=True

                 ):
        super(ConvBatchNormReLU, self).__init__()
        self.is_3d = True
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.is_3d = False

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=conv_bias
        )
        ops.append(self.conv)

        if self.is_3d:
            if has_bn:
                self.bn = nn.BatchNorm3d(output_channels)
                ops.append(self.bn)

        else:
            if has_bn:
                self.bn = nn.BatchNorm2d(output_channels)
                ops.append(self.bn)

        if has_relu:
            self.relu = nn.ReLU(inplace=inplace)
            ops.append(self.relu)
        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class DSC(nn.Module):
    def __init__(self, in_channels, conv_op: Type[_ConvNd]):
        super(DSC, self).__init__()
        self.is_3d = True
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.is_3d = False
        self.conv3x3 = conv_op(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3, padding=1)
        if self.is_3d:
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = nn.BatchNorm2d

        self.bn = nn.ModuleList([norm_layer(in_channels), norm_layer(in_channels), norm_layer(in_channels)])
        self.conv1x1 = nn.ModuleList([
            conv_op(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
            conv_op(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)
        ])
        self.conv3x3_1 = nn.ModuleList([
            conv_op(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
            conv_op(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)
        ])
        self.conv3x3_2 = nn.ModuleList([
            conv_op(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
            conv_op(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)
        ])
        self.conv_last = ConvBatchNormReLU(conv_op, input_channels=in_channels, output_channels=in_channels, kernel_size=1, stride=1, pad=0,
                                    dilation=1)
        self.norm = nn.Sigmoid()
        self.conv1 = conv_op(in_channels * 2, 1, kernel_size=1, padding=0)
        self.dconv1 = conv_op(in_channels * 2, in_channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)
        if self.is_3d:
            branches_2 = F.conv3d(x, self.conv3x3.weight, padding=2, dilation=2)
        else:
            branches_2 = F.conv2d(x, self.conv3x3.weight, padding=2, dilation=2)

        branches_2 = self.bn[1](branches_2)

        if self.is_3d:
            branches_3 = F.conv3d(x, self.conv3x3.weight, padding=4, dilation=4)
        else:
            branches_3 = F.conv2d(x, self.conv3x3.weight, padding=4, dilation=4)
        branches_3 = self.bn[2](branches_3)

        feat = torch.cat([branches_1, branches_2], dim=1)

        feat_g = feat
        feat_g1 = self.relu(self.conv1(feat_g))
        feat_g1 = self.norm(feat_g1)

        out1 = feat_g * feat_g1
        out1 = self.dconv1(out1)

        feat = self.relu(self.conv1x1[0](feat))
        feat = self.relu(self.conv3x3_1[0](feat))
        att = self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, ...].unsqueeze(1)
        att_2 = att[:, 1, ...].unsqueeze(1)

        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2 + out1

        feat1 = torch.cat([fusion_1_2, branches_3], dim=1)

        feat_g = feat1
        feat_g1 = self.relu(self.conv1(feat_g))
        feat_g1 = self.norm(feat_g1)
        out2 = feat_g * feat_g1
        out2 = self.dconv1(out2)

        feat1 = self.relu(self.conv1x1[1](feat1))
        feat1 = self.relu(self.conv3x3_1[1](feat1))
        att1 = self.conv3x3_2[1](feat1)
        att1 = F.softmax(att1, dim=1)

        att_1_2 = att1[:, 0, ...].unsqueeze(1)
        att_3 = att1[:, 1, ...].unsqueeze(1)

        ax = self.relu(self.gamma * (att_1_2 * fusion_1_2 + att_3 * branches_3 + out2) + (1 - self.gamma) * x)
        ax = self.conv_last(ax)

        return ax


if __name__ == '__main__':
    # For 2D
    # data = torch.rand((2, 4, 512, 512))
    # dsc_2d = DSC(in_channels=4, conv_op=nn.Conv2d)
    # output = dsc_2d(data)
    # print(output.size())
    # For 3D
    data = torch.rand((2, 4, 128, 128, 128))
    dsc_3d = DSC(in_channels=4, conv_op=nn.Conv3d)
    output = dsc_3d(data)
    print(output.shape)

    # For 3D模型打印
    dsc_3d = DSC(in_channels=4, conv_op=nn.Conv3d)
    dsc_3d = dsc_3d.cuda()
    summary(dsc_3d, (4, 128, 128, 128),batch_size=2)
