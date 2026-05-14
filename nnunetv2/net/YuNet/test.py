import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from nnunetv2.net.YuNet.vision_lstm import ViLBlock, SequenceTraversal
from nnunetv2.net.YuNet.ASFEB import ASFEB


# 新增：引入自定义的 ARConv
from nnunetv2.net.YuNet.ARConv import ARConv

class UpsampleLayer(nn.Module):
    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        pool_op_kernel_size,
        mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class ViLLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlock(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )

    @torch.cuda.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_vil = self.vil(x_flat)
        out = x_vil.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class BasicResBlock(nn.Module):
    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        norm_op,
        norm_op_kwargs,
        kernel_size=3,
        padding=1,
        stride=1,
        use_1x1conv=False,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True}
    ):
        super().__init__()
        self.conv1 = conv_op(input_channels, output_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels,
                                 kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class UNetResEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        # 处理可选参数为列表
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(strides) == n_stages

        # 将每个 kernel_size 标量转换为和 conv_op 维度一致的列表（如 2D 转成 [3,3]）
        self.kernel_sizes = [
            maybe_convert_scalar_to_list(conv_op, ks)
            for ks in kernel_sizes
        ]
        # 根据 kernel_sizes 计算 padding
        self.conv_pad_sizes = [
            [k // 2 for k in ks]
            for ks in self.kernel_sizes
        ]

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # Stem 部分也使用 ARConv
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=self.kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=self.kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        # 后续各 stage
        input_ch = stem_channels
        stages = []
        for s in range(n_stages):
            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        conv_op=conv_op,
                        input_channels=input_ch,
                        output_channels=features_per_stage[s],
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        kernel_size=self.kernel_sizes[s],
                        padding=self.conv_pad_sizes[s],
                        stride=strides[s],
                        use_1x1conv=True,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs
                    ),
                    *[
                        BasicBlockD(
                            conv_op=conv_op,
                            input_channels=features_per_stage[s],
                            output_channels=features_per_stage[s],
                            kernel_size=self.kernel_sizes[s],
                            stride=1,
                            conv_bias=conv_bias,
                            norm_op=norm_op,
                            norm_op_kwargs=norm_op_kwargs,
                            nonlin=nonlin,
                            nonlin_kwargs=nonlin_kwargs,
                        )
                        for _ in range(n_blocks_per_stage[s] - 1)
                    ]
                )
            )
            input_ch = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, st) for st in strides]
        self.return_skips = return_skips

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips if self.return_skips else skips[-1]

    def compute_conv_feature_map_size(self, input_size):
        size = self.stem.compute_conv_feature_map_size(input_size) if self.stem else 0
        for s, stage in enumerate(self.stages):
            size += stage.compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return size


# Decoder 部分保持原样
class UNetResDecoder(nn.Module):
    # ...（同原版，略去）
    pass


class UXlstmBot(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        # 编码器块数列表化
        n_blocks_per_stage = n_conv_per_stage if isinstance(n_conv_per_stage, list) else [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # 构造编码器，强制使用 ARConv
        self.encoder = UNetResEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=ARConv,                  # ← 这里替换为 ARConv
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.xlstm = ViLLayer(dim=self.encoder.output_channels[-1])
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        # 也可以解开下面两行，先在最深层经 ViLBlock 处理
        # skips[-1] = self.xlstm(skips[-1])
        out = self.decoder(skips)
        return out

    def compute_conv_feature_map_size(self, input_size):
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    # -------- 配置项 --------
    input_channels = 3
    n_stages = 4
    features_per_stage = [16, 32, 64, 128]
    kernel_sizes = 3
    strides = 2
    n_conv_per_stage = 2
    num_classes = 1
    n_conv_per_stage_decoder = 2
    norm_op = nn.BatchNorm2d
    norm_op_kwargs = {'eps': 1e-5, 'momentum': 0.1}
    nonlin = nn.ReLU
    nonlin_kwargs = {'inplace': True}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 ARConv 单层
    conv = ARConv(input_channels, features_per_stage[0], kernel_size=3, padding=1).to(device)
    x = torch.randn(2, input_channels, 64, 64, device=device)
    y = conv(x)
    print(f"ARConv 输出形状: {y.shape}")  # 期望 (2,16,64,64)

    # 测试整个 UXlstmBot
    model = UXlstmBot(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=False,
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
        deep_supervision=False
    ).to(device)

    x = torch.randn(1, input_channels, 256, 256, device=device)
    print(f"输入形状: {x.shape}")
    out = model(x)
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            print(f"输出[{i}] 形状: {o.shape}")
    else:
        print(f"输出形状: {out.shape}")
