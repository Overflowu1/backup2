"""
UXlstmBotArconv: nnU-Net style residual encoder-decoder with
1) corrected geometry-adaptive anisotropic ARConv blocks,
2) bidirectional residual Vision-LSTM bottleneck,
3) ASFEB-enhanced decoder skip fusion.

This file keeps the public class name ``UXlstmBotArconv`` so the existing trainer can
instantiate it without changing nnU-Net plans.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_pool_op,
    maybe_convert_scalar_to_list,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from nnunetv2.net.YuNet.ARConv3D import ARConv3D
from nnunetv2.net.YuNet.ASFEB import ASFEB
from nnunetv2.net.YuNet.vision_lstm import SequenceTraversal, ViLBlock


SizeLike = Union[int, Sequence[int]]


def _to_list(value: SizeLike, dim: int) -> List[int]:
    if isinstance(value, int):
        return [int(value)] * dim
    value = list(value)
    if len(value) != dim:
        raise ValueError(f"Expected value with length {dim}, got {value}")
    return [int(v) for v in value]


def _conv_out_size(input_size: Sequence[int], kernel_size: SizeLike, stride: SizeLike, padding: SizeLike) -> List[int]:
    dim = len(input_size)
    kernel_size = _to_list(kernel_size, dim)
    stride = _to_list(stride, dim)
    padding = _to_list(padding, dim)
    return [
        int((i + 2 * p - k) // s + 1)
        for i, k, s, p in zip(input_size, kernel_size, stride, padding)
    ]


def _feature_count(channels: int, spatial_size: Sequence[int]) -> np.int64:
    return np.int64(channels) * np.prod(spatial_size, dtype=np.int64)


def _has_compute_feature_map_size(module: nn.Module) -> bool:
    return callable(getattr(module, "compute_conv_feature_map_size", None))


class ConvFeatureSequential(nn.Sequential):
    """nn.Sequential with a best-effort nnU-Net feature-map-size hook."""

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        output = np.int64(0)
        current_size = list(input_size)
        for module in self:
            if _has_compute_feature_map_size(module):
                output += module.compute_conv_feature_map_size(current_size)
            if hasattr(module, "get_output_spatial_size"):
                current_size = module.get_output_spatial_size(current_size)
        return output


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        pool_op_kernel_size: SizeLike,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        return self.conv(x)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        scale = _to_list(self.pool_op_kernel_size, len(input_size))
        out_size = [int(i * s) for i, s in zip(input_size, scale)]
        return _feature_count(self.output_channels, out_size)


class ViLLayer(nn.Module):
    """Bidirectional residual ViL bottleneck for 3D feature maps."""

    def __init__(self, dim: int, layer_scale_init: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.vil_forward = ViLBlock(dim=self.dim, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
        self.vil_backward = ViLBlock(dim=self.dim, direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        self.layer_scale = nn.Parameter(torch.ones(1) * float(layer_scale_init))

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        batch_size, channels = x.shape[:2]
        if channels != self.dim:
            raise AssertionError(f"ViLLayer expected {self.dim} channels, got {channels}")

        spatial_shape = x.shape[2:]
        n_tokens = int(np.prod(spatial_shape))
        x_flat = x.reshape(batch_size, channels, n_tokens).transpose(1, 2).contiguous()

        x_norm = self.norm(x_flat)
        y_forward = self.vil_forward(x_norm)
        y_backward = self.vil_backward(x_norm)
        # ViLBlock is internally residual. Subtract x_norm to keep this wrapper residual
        # around the original feature tensor instead of replacing it with normalized tokens.
        delta = 0.5 * ((y_forward - x_norm) + (y_backward - x_norm))
        y = x_flat + self.layer_scale * delta

        y = y.transpose(1, 2).contiguous().reshape(batch_size, channels, *spatial_shape)
        return y.to(dtype=original_dtype)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        return _feature_count(self.dim, input_size)


class BasicResBlock(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        kernel_size: SizeLike = 3,
        padding: SizeLike = 1,
        stride: SizeLike = 1,
        use_1x1conv: bool = False,
        conv_bias: bool = False,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
    ) -> None:
        super().__init__()
        nonlin_kwargs = dict(nonlin_kwargs or {"inplace": True})
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, stride=1, padding=padding, bias=conv_bias)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride, bias=conv_bias) if use_1x1conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv3(x) if self.conv3 is not None else x
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = y + residual
        return self.act2(y)

    def get_output_spatial_size(self, input_size: Sequence[int]) -> List[int]:
        return _conv_out_size(input_size, self.kernel_size, self.stride, self.padding)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        out_size = self.get_output_spatial_size(input_size)
        # conv1 + conv2 + optional projection.
        output = _feature_count(self.output_channels, out_size) * 2
        if self.conv3 is not None:
            output += _feature_count(self.output_channels, out_size)
        return output


class BasicResArconvBlock(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        kernel_size: SizeLike = 3,
        padding: SizeLike = 1,
        stride: SizeLike = 1,
        use_1x1conv: bool = False,
        conv_bias: bool = False,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        arconv_d_max: float = 2.0,
        arconv_h_max: float = 2.0,
        arconv_w_max: float = 2.0,
        arconv_reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        nonlin_kwargs = dict(nonlin_kwargs or {"inplace": True})
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        if conv_op != nn.Conv3d:
            raise NotImplementedError("BasicResArconvBlock currently supports 3D convolution only.")

        self.conv1 = ARConv3D(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            d_max=arconv_d_max,
            h_max=arconv_h_max,
            w_max=arconv_w_max,
            conv_bias=conv_bias,
            reg_weight=arconv_reg_weight,
        )
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride, bias=conv_bias) if use_1x1conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv3(x) if self.conv3 is not None else x
        y = self.norm1(self.conv1(x))
        y = y + residual
        return self.act1(y)

    def get_output_spatial_size(self, input_size: Sequence[int]) -> List[int]:
        return _conv_out_size(input_size, self.kernel_size, self.stride, self.padding)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        out_size = self.get_output_spatial_size(input_size)
        output = np.int64(0)
        if _has_compute_feature_map_size(self.conv1):
            output += self.conv1.compute_conv_feature_map_size(input_size)
        else:
            output += _feature_count(self.output_channels, out_size)
        if self.conv3 is not None:
            output += _feature_count(self.output_channels, out_size)
        return output


class UNetResEncoder(nn.Module):
    def __init__(
        self,
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
        pool_type: str = "conv",
        arconv_stage_idxs: Union[Tuple[int, ...], List[int]] = (1, 2),
        arconv_reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        self.conv_pad_sizes = []
        dim = convert_conv_op_to_dim(conv_op)
        for kernel in kernel_sizes:
            kernel_list = _to_list(kernel, dim)
            self.conv_pad_sizes.append([i // 2 for i in kernel_list])

        self.pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != "conv" else None
        self.arconv_stage_idxs = set(int(i) for i in arconv_stage_idxs)

        stem_channels = int(stem_channels or features_per_stage[0])
        self.stem = ConvFeatureSequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                use_1x1conv=True,
                conv_bias=conv_bias,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(n_blocks_per_stage[0] - 1)
            ],
        )

        current_channels = stem_channels
        stages = []
        for stage_id in range(n_stages):
            use_arconv = stage_id in self.arconv_stage_idxs
            first_block_cls = BasicResArconvBlock if use_arconv else BasicResBlock
            first_block_kwargs = {}
            if use_arconv:
                first_block_kwargs["arconv_reg_weight"] = arconv_reg_weight

            stage = ConvFeatureSequential(
                first_block_cls(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=current_channels,
                    output_channels=features_per_stage[stage_id],
                    kernel_size=kernel_sizes[stage_id],
                    padding=self.conv_pad_sizes[stage_id],
                    stride=strides[stage_id],
                    use_1x1conv=True,
                    conv_bias=conv_bias,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    **first_block_kwargs,
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[stage_id],
                        output_channels=features_per_stage[stage_id],
                        kernel_size=kernel_sizes[stage_id],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                    for _ in range(n_blocks_per_stage[stage_id] - 1)
                ],
            )
            stages.append(stage)
            current_channels = features_per_stage[stage_id]

        self.stages = nn.ModuleList(stages)
        self.output_channels = list(features_per_stage)
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips if self.return_skips else skips[-1]

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        output = self.stem.compute_conv_feature_map_size(input_size)
        current_size = list(input_size)
        for stage in self.stages:
            output += stage.compute_conv_feature_map_size(current_size)
            first = stage[0]
            if hasattr(first, "get_output_spatial_size"):
                current_size = first.get_output_spatial_size(current_size)
        return output


class UNetResDecoder(nn.Module):
    def __init__(
        self,
        encoder: UNetResEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.encoder = encoder
        self.num_classes = int(num_classes)
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as encoder resolution stages - 1"
        )

        stages = []
        upsample_layers = []
        seg_layers = []
        asfeb_modules = []

        for stage_id in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-stage_id]
            input_features_skip = encoder.output_channels[-(stage_id + 1)]
            stride_for_upsampling = encoder.strides[-stage_id]

            upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling,
                    mode="nearest",
                )
            )

            asfeb_modules.append(
                ASFEB(
                    input_features_skip,
                    input_features_skip,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                )
            )

            stages.append(
                ConvFeatureSequential(
                    BasicResBlock(
                        conv_op=encoder.conv_op,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                        input_channels=2 * input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(stage_id + 1)],
                        padding=encoder.conv_pad_sizes[-(stage_id + 1)],
                        stride=1,
                        use_1x1conv=True,
                        conv_bias=encoder.conv_bias,
                    ),
                    *[
                        BasicBlockD(
                            conv_op=encoder.conv_op,
                            input_channels=input_features_skip,
                            output_channels=input_features_skip,
                            kernel_size=encoder.kernel_sizes[-(stage_id + 1)],
                            stride=1,
                            conv_bias=encoder.conv_bias,
                            norm_op=encoder.norm_op,
                            norm_op_kwargs=encoder.norm_op_kwargs,
                            nonlin=encoder.nonlin,
                            nonlin_kwargs=encoder.nonlin_kwargs,
                        )
                        for _ in range(n_conv_per_stage[stage_id - 1] - 1)
                    ],
                )
            )
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.asfeb_modules = nn.ModuleList(asfeb_modules)

    def forward(self, skips: List[torch.Tensor]):
        low_res_input = skips[-1]
        seg_outputs = []

        for stage_id in range(len(self.stages)):
            x = self.upsample_layers[stage_id](low_res_input)
            skip_features = self.asfeb_modules[stage_id](skips[-(stage_id + 2)])
            x = torch.cat((x, skip_features), dim=1)
            x = self.stages[stage_id](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[stage_id](x))
            elif stage_id == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            low_res_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def _encoder_skip_sizes(self, input_size: Sequence[int]) -> List[List[int]]:
        sizes = []
        current_size = list(input_size)
        for stage in self.encoder.stages:
            first = stage[0]
            if hasattr(first, "get_output_spatial_size"):
                current_size = first.get_output_spatial_size(current_size)
            sizes.append(list(current_size))
        return sizes

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        skip_sizes = self._encoder_skip_sizes(input_size)
        output = np.int64(0)
        for stage_id in range(len(self.stages)):
            low_res_size = skip_sizes[-(stage_id + 1)]
            skip_size = skip_sizes[-(stage_id + 2)]
            output += self.upsample_layers[stage_id].compute_conv_feature_map_size(low_res_size)
            output += self.asfeb_modules[stage_id].compute_conv_feature_map_size(skip_size)
            output += self.stages[stage_id].compute_conv_feature_map_size(skip_size)
            if self.deep_supervision or stage_id == (len(self.stages) - 1):
                output += _feature_count(self.num_classes, skip_size)
        return output


class UXlstmBotArconv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
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
        stem_channels: int = None,
        arconv_stage_idxs: Union[Tuple[int, ...], List[int]] = (1, 2),
        arconv_reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        n_blocks_per_stage = list(n_conv_per_stage) if not isinstance(n_conv_per_stage, int) else [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        else:
            n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)

        # Keep the original efficiency trick: reduce conv depth at very low resolutions.
        for stage_id in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[stage_id] = 1
        for stage_id in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[stage_id] = 1

        assert len(n_blocks_per_stage) == n_stages, (
            f"n_blocks_per_stage must have {n_stages} entries, got {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            f"n_conv_per_stage_decoder must have {n_stages - 1} entries, got {n_conv_per_stage_decoder}"
        )

        self.encoder = UNetResEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels,
            arconv_stage_idxs=arconv_stage_idxs,
            arconv_reg_weight=arconv_reg_weight,
        )

        self.xlstm = ViLLayer(dim=features_per_stage[-1])
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        skips[-1] = self.xlstm(skips[-1])
        return self.decoder(skips)

    def get_arconv_regularization_loss(self) -> torch.Tensor:
        regs = []
        device = next(self.parameters()).device
        for module in self.modules():
            if isinstance(module, ARConv3D) and module.reg_loss is not None:
                regs.append(module.reg_loss)
        if len(regs) == 0:
            return torch.zeros((), device=device)
        return torch.stack([r.to(device) for r in regs]).sum()

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        output = self.encoder.compute_conv_feature_map_size(input_size)
        skip_sizes = self.decoder._encoder_skip_sizes(input_size)
        if len(skip_sizes) > 0:
            output += self.xlstm.compute_conv_feature_map_size(skip_sizes[-1])
        output += self.decoder.compute_conv_feature_map_size(input_size)
        return output
