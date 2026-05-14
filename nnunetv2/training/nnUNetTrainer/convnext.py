# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class ConvTransposeModel(nn.Module):
    def __init__(self):
        super(ConvTransposeModel, self).__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose3d(in_channels=6, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_transpose(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W, D = x.size()

        proj_query = self.query_conv(x).view(batch_size, -1, H * W * D).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W * D)
        energy = torch.bmm(proj_query, proj_key)

        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(batch_size, -1, H * W * D)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W, D)

        out = self.gamma * out + x
        return out
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
	(1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
	(2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
	We use (2) as we find it slightly faster in PyTorch

	Args:
		dim (int): Number of input channels.
		drop_path (float): Stochastic depth rate. Default: 0.0
		layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
	"""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W,D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x


class BlockUp(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.ConvTranspose3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim // 4)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim // 4, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
		A PyTorch impl of : `A ConvNet for the 2020s`  -
		  https://arxiv.org/pdf/2201.03545.pdf

	Args:
		in_chans (int): Number of input image channels. Default: 3
		num_classes (int): Number of classes for classification head. Default: 1000
		depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
		dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
		drop_path_rate (float): Stochastic depth rate. Default: 0.
		layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
		head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
	"""

    # dims = [24, 48, 96, 192, 384, 768], dimsup = [768, 384, 192, 96],
    def __init__(self, in_chans=3, num_classes=4,
                 depths=[3, 3, 9, 3], dims=[12, 24, 48, 96], dimsup=[48, 24, 12, 6], dimst=[96, 48, 24],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.skip_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.upsample_layers = nn.ModuleList()
        stem2 = nn.Sequential(
            nn.ConvTranspose3d(96, 48, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            LayerNorm(48, eps=1e-6, data_format="channels_first")
        )
        self.upsample_layers.append(stem2)

        for i in range(3):
            upsample_layer = nn.Sequential(
                LayerNorm(dimsup[i], eps=1e-6, data_format="channels_first"),
                nn.ConvTranspose3d(dimsup[i], dimsup[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            )
            self.upsample_layers.append(upsample_layer)
        self.stages2 = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates2 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur2 = 0
        for i in range(4):
            stage2 = nn.Sequential(
                *[BlockUp(dim=dimsup[i], drop_path=dp_rates[cur2 + j],
                          layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages2.append(stage2)
            cur2 += depths[i]
        self.conv1 = nn.ConvTranspose3d(6, 4, kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(0, 0, 0))
        self.conv2 = nn.Conv3d(4, 4, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(4)
        self.relu1 = nn.ReLU(inplace=True)

        # self.norm = nn.LayerNorm(dimsup[0], eps=1e-6)  # final norm layer
        self.norm = nn.LayerNorm(dims[0], eps=1e-6)
        self.head = nn.Linear(dims[0], 1)
        self.poolt = nn.ModuleList([CBAM(dims) for dims in dimst])
        # self.poolt = nn.AvgPool3d(kernel_size=1,stride=1)
        self.attention_layer = SelfAttention(96)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        skip_layers = []
        for i in range(4):
            # print(f"ÏÂ²ÉÑùÇ°µÚ {i + 1}£ºx shape: {x.shape}")  # ´òÓ¡ÏÂ²ÉÑùÇ°µÄÐÎ×´
            x = self.downsample_layers[i](x)
            # print(f"ÏÂ²ÉÑùºóµÚ {i+1}£ºx shape: {x.shape}")  # ´òÓ¡ÏÂ²ÉÑùÇ°µÄÐÎ×´
            x = self.stages[i](x)
            skip = x.detach().clone().to(x.device)
            skip_layers.append(skip.to(x.device))
            # print("stage x shape:", x.shape)  # ´òÓ¡ÊäÈëÕÅÁ¿xµÄÐÎ×´
        # return self.norm(x.mean([-3, -2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        return x, skip_layers

    def up_features(self, x, skip_layers):
        j = 3
        k = 0
        for i in range(3):
            # print(x.shape[1])
            # print("2",self.poolt[k](skip_layers[j]).shape)
            x = torch.cat([x, self.poolt[k](skip_layers[j])], dim=1)
            inputc = x.shape[1]
            outputc = self.poolt[k](skip_layers[j]).shape[1]
            ConvCut = nn.Conv3d(inputc, outputc, kernel_size=3, stride=1, padding=1).to(x.device)
            # print("3", x.shape)
            x = ConvCut(x)
            # print(x.shape)
            k += 1
            j = j - 1
            # print(f"ÉÏ²ÉÑùÇ°µÚ {i + 1}£ºx shape: {x.shape}")  # ´òÓ¡ÏÂ²ÉÑùÇ°µÄÐÎ×´
            x = self.upsample_layers[i](x)
            # print(f"ÉÏ²ÉÑùºóµÚ {i+1}²ã£ºx shape: {x.shape}")  # ´òÓ¡ÏÂ²ÉÑùÇ°µÄÐÎ×´
            x = self.stages2[i](x)
            # print("2",x.shape)
            # print("stage x shape:", x.shape)  # ´òÓ¡ÊäÈëÕÅÁ¿xµÄÐÎ×´
        # return self.norm(x.mean([-3, -2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

        # print(x)
        x = self.upsample_layers[3](x)
        # print(f"ÉÏ²ÉÑùºóµÚ {i+1}²ã£ºx shape: {x.shape}")  # ´òÓ¡ÏÂ²ÉÑùÇ°µÄÐÎ×´
        x = self.stages2[3](x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x



    def forward(self, x):
        # print("Input x shape:", x.shape)  # ´òÓ¡ÊäÈëÕÅÁ¿xµÄÐÎ×´
        x, skip_layers = self.forward_features(x)
        # print("After downsample:", x.shape)  # ´òÓ¡ÏÂ²ÉÑùºóµÄÐÎ×´

        # for i, skip_layer in enumerate(skip_layers):
        # print("µÚ", i, "²ã", skip_layer.shape)
        # print(skip_layer)
        #attention_output = self.attention_layer(x)
        x = self.up_features(x, skip_layers)
        # print("After upsample:", x.shape)  # ´òÓ¡ÉÏ²ÉÑùºóµÄÐÎ×´

        # x = self.head(x)
        #print("Input x shape:", x.shape)  # ´òÓ¡ÊäÈëÕÅÁ¿xµÄÐÎ×´
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # ÐÞ¸ÄÎª AdaptiveAvgPool3d
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        fc_out = self.fc(avg_out)
        channel_attention = self.sigmoid(fc_out).view(x.size(0), x.size(1), 1, 1, 1)
        return x * channel_attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)  # ÐÞ¸ÄÎª Conv3d
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        min_out = torch.min(x, dim=1, keepdim=True)[0]
        concat = torch.cat([max_out, min_out], dim=1)
        spatial_attention = self.conv(concat)
        spatial_attention = self.sigmoid(spatial_attention)
        return x * spatial_attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        # x = x * channel_att * spatial_att
        return x


# È»ºóÔÚ ConvNeXt ÀàÖÐµÄ³õÊ¼»¯ CBAM Ä£¿éÊ±£¬ÐèÒª´«µÝºÏÊÊµÄÍ¨µÀÊý£¨channels£©²ÎÊý
# ±ÈÈç¼ÙÉè in_chans = 32£¬¼´ÊäÈëÍ¨µÀÊýÎª 32£¬ÄÇÃ´


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
	The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
	shape (batch_size, height, width, channels) while channels_first corresponds to inputs
	with shape (batch_size, channels, height, width).
	"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            # s = s.unsqueeze(-1)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
#
#
# import torch
#
#
# def custom_softmax(input_tensor, dim=None, temperature=None):
#     if temperature is not None:
#         input_tensor = input_tensor / temperature
#     min_value = input_tensor.min().item()
#     if min_value < 0:
#         input_tensor = input_tensor - min_value
#     max_values, _ = input_tensor.max(dim=dim, keepdim=True)
#     exp_values = torch.exp(input_tensor - max_values)
#     sum_exp_values = exp_values.sum(dim=dim, keepdim=True)
#     softmax_output = exp_values / sum_exp_values
#
#     return softmax_output
#
#
# x = torch.rand((2, 1, 80, 192, 160))
# o = torch.cat([x] * 3, dim=1)
# o1 = torch.softmax(o, dim=1)
# y = torch.softmax(x, dim=1)
# z = custom_softmax(x, dim=1)
# z2 = torch.sigmoid(y)
# z3 = torch.squeeze(x, dim=1)
# for dim in range(2, 5):
#     print(dim)
#     z4 = torch.softmax(x, dim=dim)
#     z4 += z4
#     print(z4)
#
#
#
# volume_data = torch.randn(2, 3, 64, 128, 128)
#
# # ¶ÔÃ¿¸öÌåËØ½øÐÐ min-max ¹éÒ»»¯
# min_value = volume_data.min()
# max_value = volume_data.max()
# normalized_volume = (volume_data - min_value) / (max_value - min_value)
#
# print("Ô­Ê¼ÌåÊý¾ÝÐÎ×´:", volume_data.shape)
# print("¾­¹ý min-max ¹éÒ»»¯ºóµÄÌåÊý¾ÝÐÎ×´:", normalized_volume.shape)
# +y = CBAM(3)
# z = y(x)
# print(z)
from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss



class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()

        return -dc


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)
    # return torch.softmax(x,dim=1)[:,1,:,:].unsqueeze(1)


def my_get_dice_loss(preds: torch.tensor, target: torch.tensor):
    weight = 0.5
    pred = torch.softmax(preds, dim=1)[:, 1, :, :, :].unsqueeze(1)
    # gt = target.long()
    # target_one_hot = torch.zeros(pred.shape, device=pred.device, dtype=torch.bool)
    # target_one_hot.scatter_(1, gt, 1)
    targets_one_hot = target.squeeze(dim=1)
    #criterion = nn.CrossEntropyLoss(reduction='mean')
    #ce_loss = criterion(pred, target_one_hot)
    ce_loss = nn.CrossEntropyLoss(pred, target, reduction='mean')
    loss_fn = MemoryEfficientSoftDiceLoss()
    dc_loss = loss_fn(pred, target)
    result = dc_loss + ce_loss
    return result
# def custom_softmax(input_tensor:torch.Tensor, dim=None, temperature=None):
#     x = input_tensor.clone()
#     if temperature is not None:
#         input_tensor = input_tensor / temperature
#
#     # Ensure the input tensor is non-negative
#     min_value = input_tensor.min().item()
#     if min_value < 0:
#         input_tensor = input_tensor - min_value
#
#     # Compute softmax
#     max_values, lastshu = x.max(dim=dim, keepdim=True)
#     exp_values = torch.exp(input_tensor - max_values)
#     print(exp_values.shape)
#     sum_exp_values = exp_values.sum(dim=dim, keepdim=True)
#     print(sum_exp_values.shape)
#     softmax_output = exp_values / sum_exp_values
#
#     return softmax_output

# def my_get_dice_loss(preds: torch.tensor, target: torch.tensor):
#     pred = torch.softmax(preds, dim=1)[:, 1, :, :, :].unsqueeze(1)
#     # print("Maximum value:", max_value.item())
#     # print("Minimum value:", min_value.item())
#     # pred = torch.softmax(preds,dim=(2, 3, 4))
#     # pred2 = custom_softmax(preds, dim=1, temperature=1.0)
#     gt = target.long()
#     target = torch.zeros(pred.shape, device=pred.device, dtype=torch.bool)
#     target.scatter_(1, gt, 1)
#     inter = (pred * target)
#     union = (pred + target)
#     # pred:BCHW, target: BCHW
#     # if (len(pred.shape) == 5) and (len(target.shape) == 5):
#     inter = (pred * target).sum(dim=(2, 3, 4))  # Sum along D, H, and W dimensions
#     union = (pred + target).sum(dim=(2, 3, 4))  # Sum along D, H, and W dimensions
#     dice_loss = 1 - 2 * (inter + 1) / (union + 2)
#     return dice_loss.mean()



if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((2, 3, 80, 192, 160), device=cuda0)
        z = torch.randint(0, 4, (2, 1, 80, 192, 160), device=cuda0)
        print(z.shape)
        # x = torch.rand((1, 3, 512, 512, 250), device=cuda0)
        model = ConvNeXt()
        model.cuda()
        y = model(x)
        #z = my_get_dice_loss(preds=y, target=y2)
        pred = torch.softmax(y, dim=1)
        ce_loss = nn.CrossEntropyLoss()(pred, z.squeeze(dim=1))
        print(ce_loss)
        # y = torch.tensor([item.cpu().detach().numpy() for item in y]).cuda()
        # print(y.shape)
        # print(y)
        # pred = torch.softmax(y, dim=1)[:, 1, :, :, :].unsqueeze(1)
        # print(pred.shape)
# import torch
# x = torch.rand((2, 1, 80, 192, 160))
# y = torch.rand((2, 12, 80, 192, 160))
# z = torch.softmax(y, dim=1)[:, 1, :, :, :].unsqueeze(1)
# x1 = x + y
# x2 = x * y
