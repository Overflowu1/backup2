import torch
import torch.nn as nn
from typing import Type, Union, List, Tuple


class HWB(nn.Module):
    def __init__(self,
                 conv_op: Type[nn.Module],
                 n_feat,
                 o_feat,
                 kernel_size,
                 bias,
                 act=nn.ReLU(inplace=True)
                 ):
        super(HWB, self).__init__()
        self.is_3d = True
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.is_3d = False


        if self.is_3d:
            self.dwt = DWT3D()
            self.iwt = IWT3D()
        else:
            self.dwt = DWT()
            self.iwt = IWT()

        if n_feat % 2 == 0:
            modules_body = [
                conv_op(n_feat * 2, n_feat, kernel_size, padding=1, bias=bias),
                act,
                conv_op(n_feat, n_feat * 2, kernel_size, padding=1, bias=bias)
            ]

        else:
            modules_body = [
                conv_op(n_feat * 2 + 2, n_feat + 1, kernel_size, padding=1, bias=bias),
                act,
                conv_op(n_feat + 1, n_feat * 2 + 2, kernel_size, padding=1, bias=bias),
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer(conv_op)
        if n_feat % 2 == 0:
            self.WCA = CALayer(conv_op, n_feat * 2)
        else:
            self.WCA = CALayer(conv_op, n_feat * 2 + 2)

        if n_feat % 2 == 0:
            self.conv1x1 = conv_op(n_feat * 4, n_feat * 2, kernel_size=1, bias=bias)
        else:
            self.conv1x1 = conv_op(n_feat * 4 + 4, n_feat * 2 + 2, kernel_size=1, bias=bias)
        self.conv3x3 = conv_op(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = conv_op(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):

        batch, channel, depth, height, width = x.size()
        residual = x
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path_in)
        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt
        wavelet_path = self.iwt(res)
        if depth % 2 != 0:
            wavelet_path = wavelet_path[:, :, :-1, :, :]
        if height % 2 != 0:
            wavelet_path = wavelet_path[:, :, :, :-1, :]
        if width % 2 != 0:
            wavelet_path = wavelet_path[:, :, :, :, :-1]

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        # out =self.conv3x3(out)
        out += self.conv1x1_final(residual)

        return out


class SALayer(nn.Module):
    """Spatial-attention module."""

    def __init__(self, conv_op, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.cv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CALayer(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, conv_op, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Conv3d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))

import torch.nn.functional as F
def dwt3d_init(x):
    # 如果输入张量的每个维度不是偶数，则进行填充
    batch_size, channels, depth, height, width = x.size()

    if depth % 2 != 0:
        x = F.pad(x, (0, 0, 0, 0, 0, 1))  # 在depth维度上填充1
    if height % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1, 0, 0))  # 在height维度上填充1
    if width % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0, 0, 0))  # 在width维度上填充1
    x001 = x[:, :, 0::2, :, :] / 2
    x002 = x[:, :, 1::2, :, :] / 2
    x011 = x001[:, :, :, 0::2, :] / 2
    x012 = x002[:, :, :, 0::2, :] / 2
    x021 = x001[:, :, :, 1::2, :] / 2
    x022 = x002[:, :, :, 1::2, :] / 2
    x111 = x011[:, :, :, :, 0::2]
    x112 = x012[:, :, :, :, 0::2]
    x121 = x021[:, :, :, :, 0::2]
    x122 = x022[:, :, :, :, 0::2]
    x211 = x011[:, :, :, :, 1::2]
    x212 = x012[:, :, :, :, 1::2]
    x221 = x021[:, :, :, :, 1::2]
    x222 = x022[:, :, :, :, 1::2]

    x_LL = x111 + x112 + x121 + x122 + x211 + x212 + x221 + x222
    x_HL = -x111 - x112 + x121 + x122 - x211 - x212 + x221 + x222
    x_LH = -x111 + x112 - x121 + x122 - x211 + x212 - x221 + x222
    x_HH = x111 - x112 - x121 + x122 + x211 - x212 - x221 + x222

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt3d_init(x):
    r = 2
    in_batch, in_channel, in_depth, in_height, in_width = x.size()
    out_batch, out_channel, out_depth, out_height, out_width = in_batch, int(
        in_channel / 4), r * in_depth, r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_depth, out_height, out_width], device=x.device)

    h[:, :, 0::2, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2, 0::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2, 0::2] = x1 + x2 + x3 + x4

    h[:, :, 0::2, 0::2, 1::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2, 1::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT3D(nn.Module):
    def __init__(self):
        super(DWT3D, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt3d_init(x)


class IWT3D(nn.Module):
    def __init__(self):
        super(IWT3D, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt3d_init(x)


import torch
import torch.nn as nn


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda()  #

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)


# # HWB layer for 2D data
# hwb_2d = HWB(conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, dwt_op=DWT2D, iwt_op=IWT2D,
#              n_feat=64, o_feat=64, kernel_size=3, reduction=16, bias=True, act=nn.ReLU())

# HWB layer for 3D data
# hwb_3d = HWB(conv_op=nn.Conv3d,n_feat=64, o_feat=64, kernel_size=3, reduction=16, bias=True, act=nn.ReLU())

if __name__ == '__main__':
    data = torch.rand((2, 2, 125, 125, 128)).to('cuda:0')
    hwb_3d = HWB(conv_op=nn.Conv3d, n_feat=2, o_feat=5, kernel_size=3, bias=True, act=nn.ReLU()).to('cuda:0')

    output = hwb_3d(data)
    print(output.shape)
