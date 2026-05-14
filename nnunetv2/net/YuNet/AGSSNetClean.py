"""
AGSSNetClean with predicted structure-field SCI.

Strict design:
- The network predicts D_core / D_surface from image features.
- The predicted structure probability is used by SCI.
- Ground-truth D_core / D_surface must only be used in the loss, never as network input.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op
from nnunetv2.net.YuNet.ArconvNet import (
    BasicBlockD,
    BasicResBlock,
    ConvFeatureSequential,
    UNetResEncoder,
    UpsampleLayer,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _no_autocast_context_for(x: torch.Tensor):
    if torch.is_tensor(x) and x.is_cuda:
        try:
            return torch.autocast(device_type="cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _conv_norm_act(
    conv_op: Type[_ConvNd],
    in_ch: int,
    out_ch: int,
    norm_op: Optional[Type[nn.Module]],
    norm_op_kwargs: Optional[dict],
    nonlin: Optional[Type[nn.Module]],
    nonlin_kwargs: Optional[dict],
    kernel_size: int = 3,
) -> nn.Sequential:
    padding = kernel_size // 2
    layers: List[nn.Module] = [
        conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
    ]
    if norm_op is not None:
        layers.append(norm_op(out_ch, **(norm_op_kwargs or {})))
    if nonlin is not None:
        layers.append(nonlin(**(nonlin_kwargs or {"inplace": True})))
    return nn.Sequential(*layers)

def _light_bottleneck_refine(
    conv_op: Type[_ConvNd],
    channels: int,
    norm_op: Optional[Type[nn.Module]],
    norm_op_kwargs: Optional[dict],
    nonlin: Optional[Type[nn.Module]],
    nonlin_kwargs: Optional[dict],
    reduction: int = 4,
) -> nn.Sequential:
    """Cheap high-resolution refinement: 1x1 reduce -> depthwise 3x3 -> 1x1 expand.

    This replaces full C->C 3x3 convolutions in ACFA/SCI/structure stems so
    the paper module remains high-resolution but has bottleneck-level cost.
    """
    mid = max(int(channels) // max(int(reduction), 1), 8)
    nonlin = nonlin or nn.LeakyReLU
    nonlin_kwargs = nonlin_kwargs or {"inplace": True}
    layers: List[nn.Module] = [
        conv_op(channels, mid, kernel_size=1, stride=1, padding=0, bias=True),
    ]
    if norm_op is not None:
        layers.append(norm_op(mid, **(norm_op_kwargs or {})))
    layers.append(nonlin(**nonlin_kwargs))
    layers.append(conv_op(mid, mid, kernel_size=3, stride=1, padding=1, groups=mid, bias=True))
    if norm_op is not None:
        layers.append(norm_op(mid, **(norm_op_kwargs or {})))
    layers.append(nonlin(**nonlin_kwargs))
    layers.append(conv_op(mid, channels, kernel_size=1, stride=1, padding=0, bias=True))
    return nn.Sequential(*layers)


def assemble_anatomy_probs(region_logits: torch.Tensor, side_logits: torch.Tensor) -> torch.Tensor:
    """Assemble 4-class anatomy probabilities: bg, sacrum, left, right. FP32-safe."""
    out_dtype = region_logits.dtype
    with _no_autocast_context_for(region_logits):
        eps = 1e-7
        region_logits_f = torch.nan_to_num(region_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        side_logits_f = torch.nan_to_num(side_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        pr = torch.softmax(region_logits_f, dim=1)
        ps = torch.softmax(side_logits_f, dim=1)
        hip = pr[:, 2:3]
        side_lr = ps[:, 1:3]
        side_lr = side_lr / side_lr.sum(1, keepdim=True).clamp_min(eps)
        probs = torch.cat(
            [pr[:, 0:1], pr[:, 1:2], hip * side_lr[:, 0:1], hip * side_lr[:, 1:2]],
            dim=1,
        )
        probs = probs / probs.sum(1, keepdim=True).clamp_min(eps)
        probs = probs.clamp_min(eps)
    return probs.to(dtype=out_dtype)


def anatomy_prior_and_uncertainty(region_logits: torch.Tensor, side_logits: torch.Tensor):
    probs = assemble_anatomy_probs(region_logits, side_logits)
    with _no_autocast_context_for(probs):
        probs_f = probs.float().clamp_min(1e-7)
        prior = probs_f[:, 1:].sum(1, keepdim=True).clamp(0, 1)
        entropy = -(probs_f * torch.log(probs_f)).sum(1, keepdim=True)
        entropy = entropy / math.log(max(int(probs_f.shape[1]), 2))
    return prior.to(dtype=probs.dtype), entropy.clamp(0, 1).to(dtype=probs.dtype), probs


def assemble_semantic_logits(
    region_logits: torch.Tensor,
    side_logits: torch.Tensor,
    frac_logits: torch.Tensor,
    frac_gamma: float = 1.0,
) -> torch.Tensor:
    """Assemble final 5-class semantic logits in FP32 to avoid AMP log underflow."""
    out_dtype = region_logits.dtype
    with _no_autocast_context_for(region_logits):
        eps = 1e-7
        region_logits_f = torch.nan_to_num(region_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        side_logits_f = torch.nan_to_num(side_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        frac_logits_f = torch.nan_to_num(frac_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        p_anat = assemble_anatomy_probs(region_logits_f, side_logits_f).float()
        gamma = float(frac_gamma)
        if (not math.isfinite(gamma)) or gamma <= 0:
            gamma = 1.0
        p_frac = torch.sigmoid(frac_logits_f).clamp(eps, 1.0 - eps)
        if abs(gamma - 1.0) > 1e-6:
            p_frac = p_frac.pow(gamma).clamp(eps, 1.0 - eps)
        p_nonfrac = 1.0 - p_frac

        probs = torch.cat(
            [
                p_anat[:, 0:1] * p_nonfrac,
                p_anat[:, 1:2] * p_nonfrac,
                p_anat[:, 2:3] * p_nonfrac,
                p_anat[:, 3:4] * p_nonfrac,
                p_frac,
            ],
            dim=1,
        )
        probs = probs / probs.sum(1, keepdim=True).clamp_min(eps)
        logits = torch.log(probs.clamp_min(eps))
        logits = torch.nan_to_num(logits, nan=-20.0, neginf=-20.0, posinf=0.0)
    return logits.to(dtype=out_dtype)


# ---------------------------------------------------------------------------
# ACFA and SCI
# ---------------------------------------------------------------------------

class AnatomyConditionedFractureAttention(nn.Module):
    """Use predicted anatomy probability to refine fracture features."""

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        anat_classes: int = 4,
        reduction: int = 4,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        mid = max(channels // max(int(reduction), 1), 8)
        nonlin = nonlin or nn.LeakyReLU
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}

        def _norm(c: int) -> nn.Module:
            return norm_op(c, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity()

        self.anat_embed = nn.Sequential(
            conv_op(anat_classes, mid, kernel_size=1, bias=True),
            _norm(mid),
            nonlin(**nonlin_kwargs),
            conv_op(mid, channels, kernel_size=1, bias=True),
        )
        self.gate = nn.Sequential(
            conv_op(channels * 2, mid, kernel_size=1, padding=0, bias=True),
            _norm(mid),
            nonlin(**nonlin_kwargs),
            conv_op(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = _light_bottleneck_refine(
            conv_op, channels, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, reduction=reduction
        )
        self.res_scale_raw = nn.Parameter(torch.tensor(-4.0))
        self.last_gate_mean: Optional[torch.Tensor] = None
        self.last_gate_fg_mean: Optional[torch.Tensor] = None
        self.last_gate_bg_mean: Optional[torch.Tensor] = None
        self.last_res_scale: Optional[torch.Tensor] = None

    def reset_parameters_for_stable_start(self) -> None:
        nn.init.constant_(self.res_scale_raw, -4.0)

    def forward(self, feat: torch.Tensor, anatomy_probs: torch.Tensor) -> torch.Tensor:
        out_dtype = feat.dtype
        anat_p = anatomy_probs.detach().float()
        if anat_p.shape[2:] != feat.shape[2:]:
            anat_p = F.interpolate(anat_p, size=feat.shape[2:], mode="trilinear", align_corners=False)
        anat_ctx = self.anat_embed(anat_p.to(dtype=out_dtype))
        gate = self.gate(torch.cat([feat, anat_ctx], dim=1))
        scale = 0.30 * torch.sigmoid(self.res_scale_raw)
        refined = self.refine(feat)
        out = feat + scale * gate * refined
        with torch.no_grad():
            self.last_gate_mean = gate.detach().mean()
            self.last_res_scale = scale.detach()
            fg = anat_p[:, 1:].sum(dim=1, keepdim=True) > 0.5
            gate_scalar = gate.detach().mean(dim=1, keepdim=True)
            self.last_gate_fg_mean = gate_scalar[fg].mean() if bool(fg.any()) else None
            bg = ~fg
            self.last_gate_bg_mean = gate_scalar[bg].mean() if bool(bg.any()) else None
        return out.to(dtype=out_dtype)


class StructureConditionedInjection(nn.Module):
    """Inject predicted structure-field probabilities into image features."""

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
    ) -> None:
        super().__init__()
        nonlin = nonlin or nn.LeakyReLU
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}
        mid = max(channels // 4, 8)
        self.struct_proj = nn.Sequential(
            conv_op(2, mid, kernel_size=1, stride=1, padding=0, bias=True),
            norm_op(mid, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity(),
            nonlin(**nonlin_kwargs),
            conv_op(mid, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.gate = nn.Sequential(
            conv_op(channels * 2, mid, kernel_size=1, stride=1, padding=0, bias=True),
            norm_op(mid, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity(),
            nonlin(**nonlin_kwargs),
            conv_op(mid, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.refine = _light_bottleneck_refine(
            conv_op, channels, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, reduction=4
        )
        self.res_scale_raw = nn.Parameter(torch.tensor(-4.0))
        self.last_gate_mean: Optional[torch.Tensor] = None
        self.last_res_scale: Optional[torch.Tensor] = None

    def reset_parameters_for_stable_start(self) -> None:
        nn.init.constant_(self.res_scale_raw, -4.0)

    def forward(self, feat: torch.Tensor, struct_prob: torch.Tensor) -> torch.Tensor:
        if struct_prob.shape[2:] != feat.shape[2:]:
            struct_prob = F.interpolate(struct_prob, size=feat.shape[2:], mode="trilinear", align_corners=False)
        struct_prob = torch.nan_to_num(struct_prob.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        sf = self.struct_proj(struct_prob.to(dtype=feat.dtype))
        gate = self.gate(torch.cat([feat, sf], dim=1))
        scale = (0.30 * torch.sigmoid(self.res_scale_raw)).to(dtype=feat.dtype)
        out = feat + scale * gate * self.refine(feat)
        with torch.no_grad():
            self.last_gate_mean = gate.detach().mean()
            self.last_res_scale = scale.detach()
        return out




class ECASkipBlock(nn.Module):
    """Extremely lightweight channel attention for decoder skip features."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        k = max(int(kernel_size), 1)
        if k % 2 == 0:
            k += 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_axes = tuple(range(2, x.ndim))
        pooled = x.mean(dim=spatial_axes, keepdim=False).unsqueeze(1)  # B,1,C
        weight = torch.sigmoid(self.conv(pooled)).squeeze(1)
        weight = weight.view(x.shape[0], x.shape[1], *([1] * (x.ndim - 2)))
        return x * weight.to(dtype=x.dtype)


class LowResolutionARConvLite(nn.Module):
    """Low-resolution adaptive receptive-field bottleneck.

    It keeps ARConv's adaptive multi-kernel idea but removes grid_sample/offsets and
    applies the module only to the encoder bottleneck, where spatial cost is low.
    """

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        mid = max(self.channels // max(int(reduction), 1), 8)
        dim = 3 if conv_op == nn.Conv3d else 2
        if dim == 3:
            kernels = [(3, 3, 3), (1, 3, 3), (3, 1, 3), (3, 3, 1)]
        else:
            kernels = [(3, 3), (1, 3), (3, 1)]
        nonlin = nonlin or nn.LeakyReLU
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}

        def _norm(c: int) -> nn.Module:
            return norm_op(c, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity()

        self.reduce = nn.Sequential(
            conv_op(self.channels, mid, kernel_size=1, stride=1, padding=0, bias=True),
            _norm(mid),
            nonlin(**nonlin_kwargs),
        )
        self.branches = nn.ModuleList([
            conv_op(mid, mid, kernel_size=k, stride=1, padding=tuple(i // 2 for i in k), groups=mid, bias=True)
            for k in kernels
        ])
        self.routing = conv_op(mid, len(kernels), kernel_size=1, stride=1, padding=0, bias=True)
        self.expand = nn.Sequential(
            _norm(mid),
            nonlin(**nonlin_kwargs),
            conv_op(mid, self.channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.res_scale_raw = nn.Parameter(torch.tensor(-4.0))
        self.last_routing_entropy: Optional[torch.Tensor] = None
        self.last_routing_max_prob: Optional[torch.Tensor] = None
        self.last_res_scale: Optional[torch.Tensor] = None

    def reset_parameters_for_stable_start(self) -> None:
        nn.init.constant_(self.res_scale_raw, -4.0)
        nn.init.zeros_(self.routing.weight)
        nn.init.zeros_(self.routing.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        routing = torch.softmax(self.routing(z), dim=1)
        mixed = z.new_zeros(z.shape)
        for branch_id, branch in enumerate(self.branches):
            mixed = mixed + routing[:, branch_id:branch_id + 1] * branch(z)
        delta = self.expand(mixed)
        scale = (0.30 * torch.sigmoid(self.res_scale_raw)).to(dtype=x.dtype)
        with torch.no_grad():
            r = routing.detach().float().clamp_min(1e-7)
            self.last_routing_entropy = (-(r * torch.log(r)).sum(1).mean() / math.log(len(self.branches))).detach()
            self.last_routing_max_prob = r.max(1).values.mean().detach()
            self.last_res_scale = scale.detach()
        return x + scale * delta.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Clean Decoder
# ---------------------------------------------------------------------------

class CleanDecoder(nn.Module):
    """Simplified decoder: upsample + concat skip + conv block with optional ECA skip calibration."""

    def __init__(
        self,
        encoder: UNetResEncoder,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        use_skip_eca: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        feature_channels = []
        skip_eca = []
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
            feature_channels.append(input_features_skip)
            skip_eca.append(ECASkipBlock(input_features_skip) if use_skip_eca else nn.Identity())
        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.skip_eca = nn.ModuleList(skip_eca)
        self.feature_channels_high_to_low = list(reversed(feature_channels))
        self.highres_channels = int(feature_channels[-1])

    def forward(self, skips: List[torch.Tensor]) -> List[torch.Tensor]:
        low_res_input = skips[-1]
        decoder_features = []
        for stage_id in range(len(self.stages)):
            x = self.upsample_layers[stage_id](low_res_input)
            skip = self.skip_eca[stage_id](skips[-(stage_id + 2)])
            x = torch.cat((x, skip), dim=1)
            x = self.stages[stage_id](x)
            decoder_features.append(x)
            low_res_input = x
        return decoder_features[::-1]


# ---------------------------------------------------------------------------
# AGSS-Clean Network
# ---------------------------------------------------------------------------

class AGSSNetClean(nn.Module):
    """Clean hierarchical network with predicted-structure SCI."""

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
        dropout_op: Union[None, Type[nn.Module]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        stem_channels: int = None,
        use_acfa: bool = True,
        use_sci: bool = True,
        sci_detach_struct: bool = True,
        use_hierarchical_assembly: bool = True,
        use_struct_head: bool = True,
        use_sacfrac_head: bool = True,
        use_arconv_lite: bool = True,
        use_skip_eca: bool = True,
        use_lr_coord: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        else:
            features_per_stage = list(features_per_stage)
        n_blocks_per_stage = list(n_conv_per_stage) if not isinstance(n_conv_per_stage, int) else [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        else:
            n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)
        for stage_id in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[stage_id] = 1
        for stage_id in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[stage_id] = 1

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
            arconv_stage_idxs=(),
        )
        self.bottleneck_arconv_lite = (
            LowResolutionARConvLite(conv_op, features_per_stage[-1], norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
            if use_arconv_lite else nn.Identity()
        )
        self.decoder = CleanDecoder(self.encoder, n_conv_per_stage_decoder, use_skip_eca=use_skip_eca)
        self.num_classes = int(num_classes)
        self.use_arconv_lite = bool(use_arconv_lite)
        self.use_skip_eca = bool(use_skip_eca)
        self.use_lr_coord = bool(use_lr_coord)
        self.use_acfa = bool(use_acfa)
        self.use_sci = bool(use_sci)
        self.sci_detach_struct = bool(sci_detach_struct)
        self.use_hierarchical_assembly = bool(use_hierarchical_assembly)
        self.use_sacfrac_head = bool(use_sacfrac_head)
        # Struct head is required by SCI and/or direct structure-field supervision.
        # In flat/no-struct ablations it can be disabled to avoid unused parameters.
        self.use_struct_head = bool(use_struct_head or self.use_sci)
        high_ch = self.decoder.highres_channels

        if self.use_struct_head:
            self.struct_stem = _light_bottleneck_refine(
                conv_op, high_ch, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, reduction=4
            )
            self.struct_head = conv_op(high_ch, 2, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.struct_stem = None
            self.struct_head = None

        self.sci = StructureConditionedInjection(conv_op, high_ch, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) if (self.use_sci and self.use_struct_head) else None

        self.flat_head = conv_op(high_ch, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True) if not self.use_hierarchical_assembly else None

        self.frac_head = conv_op(high_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.region_head = conv_op(high_ch, 3, kernel_size=1, stride=1, padding=0, bias=True)
        side_in_ch = high_ch + (1 if self.use_lr_coord else 0)
        self.side_head = conv_op(side_in_ch, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.sacfrac_head = (
            conv_op(high_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
            if self.use_sacfrac_head else None
        )

        self.acfa = AnatomyConditionedFractureAttention(
            conv_op=conv_op,
            channels=high_ch,
            anat_classes=4,
            reduction=4,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        ) if self.use_acfa else None


    def _append_lr_coord(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.use_lr_coord:
            return feat
        width = int(feat.shape[-1])
        if width <= 1:
            coord_1d = torch.zeros(width, device=feat.device, dtype=feat.dtype)
        else:
            coord_1d = torch.linspace(-1.0, 1.0, width, device=feat.device, dtype=feat.dtype)
        view_shape = [1, 1] + [1] * (feat.ndim - 3) + [width]
        coord = coord_1d.view(*view_shape).expand(feat.shape[0], 1, *feat.shape[2:])
        return torch.cat([feat, coord], dim=1)

    def forward(self, x: torch.Tensor, return_dict: bool = True, struct_fields=None):
        # struct_fields is intentionally ignored. GT structure must not be used as network input.
        skips = self.encoder(x)
        skips[-1] = self.bottleneck_arconv_lite(skips[-1])
        features = self.decoder(skips)
        high_feat = features[0]

        struct_logits = None
        struct_prob = None
        if self.use_struct_head:
            struct_feat = self.struct_stem(high_feat)
            struct_logits = self.struct_head(struct_feat)
            struct_prob = torch.sigmoid(
                torch.nan_to_num(struct_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            ).to(dtype=high_feat.dtype)

        if self.use_sci and self.sci is not None and struct_prob is not None:
            sci_input = struct_prob.detach() if self.sci_detach_struct else struct_prob
            feat = self.sci(high_feat, sci_input)
        else:
            feat = high_feat

        # Same-backbone flat baseline: bypass hierarchical heads and semantic assembly.
        if not self.use_hierarchical_assembly:
            high_seg = self.flat_head(feat)
            if not return_dict:
                return high_seg
            out = {"seg": high_seg}
            if struct_logits is not None:
                out["struct"] = struct_logits
            if self.sci is not None:
                out.update({
                    "sci_gate_mean": self.sci.last_gate_mean,
                    "sci_res_scale": self.sci.last_res_scale,
                })
            return out

        high_region = self.region_head(feat)
        high_side = self.side_head(self._append_lr_coord(feat))

        _, _, anat_probs = anatomy_prior_and_uncertainty(high_region, high_side)
        if self.use_acfa and self.acfa is not None:
            frac_feat = self.acfa(feat, anat_probs)
        else:
            frac_feat = feat
        high_frac = self.frac_head(frac_feat)
        high_sacfrac = self.sacfrac_head(frac_feat) if self.sacfrac_head is not None else None
        high_seg = assemble_semantic_logits(high_region, high_side, high_frac)

        if not return_dict:
            return high_seg
        out = {
            "seg": high_seg,
            "frac": high_frac,
            "region": high_region,
            "side": high_side,
            "acfa_gate_mean": self.acfa.last_gate_mean if self.acfa is not None else None,
            "acfa_gate_fg_mean": self.acfa.last_gate_fg_mean if self.acfa is not None else None,
            "acfa_gate_bg_mean": self.acfa.last_gate_bg_mean if self.acfa is not None else None,
            "acfa_res_scale": self.acfa.last_res_scale if self.acfa is not None else None,
        }
        if high_sacfrac is not None:
            out["sacfrac"] = high_sacfrac
        if struct_logits is not None:
            out["struct"] = struct_logits
        if self.sci is not None:
            out.update({
                "sci_gate_mean": self.sci.last_gate_mean,
                "sci_res_scale": self.sci.last_res_scale,
            })
        if self.use_arconv_lite and isinstance(self.bottleneck_arconv_lite, LowResolutionARConvLite):
            out.update({
                "arconv_lite_routing_entropy": self.bottleneck_arconv_lite.last_routing_entropy,
                "arconv_lite_routing_max_prob": self.bottleneck_arconv_lite.last_routing_max_prob,
                "arconv_lite_res_scale": self.bottleneck_arconv_lite.last_res_scale,
            })
        return out


def get_agss_clean_from_plans(
    plans_manager,
    dataset_json,
    configuration_manager,
    num_input_channels,
    use_acfa: bool = True,
    use_sci: bool = True,
    sci_detach_struct: bool = True,
    use_hierarchical_assembly: bool = True,
    use_struct_head: bool = True,
    use_sacfrac_head: bool = True,
    use_arconv_lite: bool = True,
    use_skip_eca: bool = True,
    use_lr_coord: bool = True,
) -> AGSSNetClean:
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)
    return AGSSNetClean(
        input_channels=num_input_channels,
        n_stages=len(configuration_manager.conv_kernel_sizes),
        features_per_stage=[
            min(configuration_manager.UNet_base_num_features * 2 ** i, configuration_manager.unet_max_num_features)
            for i in range(len(configuration_manager.conv_kernel_sizes))
        ],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
        num_classes=label_manager.num_segmentation_heads,
        n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=__import__("dynamic_network_architectures.building_blocks.helper", fromlist=["get_matching_instancenorm"]).get_matching_instancenorm(conv_op),
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        use_acfa=use_acfa,
        use_sci=use_sci,
        sci_detach_struct=sci_detach_struct,
        use_hierarchical_assembly=use_hierarchical_assembly,
        use_struct_head=use_struct_head,
        use_sacfrac_head=use_sacfrac_head,
        use_arconv_lite=use_arconv_lite,
        use_skip_eca=use_skip_eca,
        use_lr_coord=use_lr_coord,
    )
