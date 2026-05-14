
"""
AGSS-Net v3.1 hybrid: hierarchical sacrum-hip-side parsing + semantic-structural fracture learning.

Hybrid additions over v3:
1. Coordinate-map support is enabled in the dataloader/trainer (borrowed from v2):
   z/y/x normalised coordinates are appended to the input image channels.
2. ChannelSE skip attention is added to the decoder skip pathway (borrowed from v2)
   to improve left/right hip discrimination while retaining the hierarchical v3 logic.
3. The hierarchical fused output remains the main prediction, while the raw semantic
   auxiliary head is kept only as an auxiliary supervision branch.

Main semantic labels:
    0 background
    1 sacrum
    2 left_hip
    3 right_hip
    4 fracture

Key changes compared with the earlier AGSS version:
1. The final 5-class output is no longer an unconstrained flat semantic head.
   It is assembled from:
      - region head: background / sacrum / hip-union
      - side head: background / left / right
      - binary fracture head
   This directly addresses left-right confusion and the conflict between
   anatomy labels and fracture override.
2. A dedicated sacrum-fracture head is added because sacrum is an unpaired
   structure and cannot rely on contralateral symmetry.
3. UM-Fusion uses hierarchical anatomy probabilities (region+side) rather than
   the flat 5-class semantic head as prior/uncertainty.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.net.YuNet.ARConv3D import ARConv3D
from nnunetv2.net.YuNet.ArconvNet import (
    BasicBlockD,
    BasicResBlock,
    ConvFeatureSequential,
    UNetResEncoder,
    UpsampleLayer,
    ViLLayer,
    _feature_count,
)
from nnunetv2.net.YuNet.ASFEB import ASFEB


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
    layers: List[nn.Module] = [conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
    if norm_op is not None:
        layers.append(norm_op(out_ch, **(norm_op_kwargs or {})))
    if nonlin is not None:
        layers.append(nonlin(**(nonlin_kwargs or {"inplace": True})))
    return nn.Sequential(*layers)




class ChannelSEBlock(nn.Module):
    """Lightweight channel SE attention on decoder skip features.

    This is taken from the v2 branch and integrated into the hierarchical v3
    decoder to improve left/right hip discrimination.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, *([1] * (x.ndim - 2)))
        return x * w

def assemble_anatomy_probs(region_logits: torch.Tensor, side_logits: torch.Tensor) -> torch.Tensor:
    """
    Assemble 4-class anatomy probabilities:
        0 background
        1 sacrum
        2 left_hip
        3 right_hip
    from:
        region head: bg/sacrum/hip_union
        side head: bg-or-nonhip/left/right
    """
    pr = torch.softmax(region_logits, dim=1)
    ps = torch.softmax(side_logits, dim=1)
    hip = pr[:, 2:3]
    side_lr = ps[:, 1:3]
    side_lr = side_lr / side_lr.sum(1, keepdim=True).clamp_min(1e-6)
    probs = torch.cat([pr[:, 0:1], pr[:, 1:2], hip * side_lr[:, 0:1], hip * side_lr[:, 1:2]], dim=1)
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
    return probs


def anatomy_prior_and_uncertainty(region_logits: torch.Tensor, side_logits: torch.Tensor):
    probs = assemble_anatomy_probs(region_logits, side_logits)
    prior = probs[:, 1:].sum(1, keepdim=True).clamp(0, 1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(1, keepdim=True)
    entropy = entropy / math.log(max(int(probs.shape[1]), 2))
    return prior, entropy.clamp(0, 1), probs


def assemble_semantic_logits(region_logits: torch.Tensor, side_logits: torch.Tensor, frac_logits: torch.Tensor) -> torch.Tensor:
    """
    Build differentiable 5-class semantic logits:
        bg, sacrum, left, right, fracture
    by overlaying binary fracture on assembled anatomy.
    """
    p_anat = assemble_anatomy_probs(region_logits, side_logits)
    p_frac = torch.sigmoid(frac_logits)
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
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
    return torch.log(probs.clamp_min(1e-6))


class UMAnatomyStructureFusion(nn.Module):
    """UM-Fusion: residual structure-feature injection gated by anatomy prior.

    Bug-fix vs original: the residual is now ``feat + gate * refine(feat)``
    (refining the main feature) rather than ``feat + gate * refine(struct_feat)``
    (which pushed structure info into background regions → gate_fg < gate_bg).
    Structure is still used as the gate condition so the gate fires in
    structured (fracture-likely) regions.
    """
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
        self.gate = nn.Sequential(
            conv_op(channels * 2 + 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
            norm_op(channels, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity(),
            nonlin(**(nonlin_kwargs or {"inplace": True})) if nonlin is not None else nn.Identity(),
            conv_op(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        # Refine the MAIN feature (not struct_feat) — fixes the gate-inversion issue.
        self.refine = _conv_norm_act(conv_op, channels, channels, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, 3)
        self.last_gate_mean = None
        self.last_gate_std = None
        self.last_gate_bg_mean = None
        self.last_gate_fg_mean = None

    def forward(self, feat: torch.Tensor, struct_feat: torch.Tensor, prior: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        if prior.shape[2:] != feat.shape[2:]:
            prior = F.interpolate(prior, size=feat.shape[2:], mode="trilinear", align_corners=False)
        if uncertainty.shape[2:] != feat.shape[2:]:
            uncertainty = F.interpolate(uncertainty, size=feat.shape[2:], mode="trilinear", align_corners=False)
        gate = self.gate(torch.cat([feat, struct_feat, prior, uncertainty], dim=1))
        gate_det = gate.detach()
        self.last_gate_mean = gate_det.mean()
        self.last_gate_std = gate_det.std()
        self.last_gate_bg_mean = None
        self.last_gate_fg_mean = None
        if not self.training:
            with torch.no_grad():
                gate_voxel = gate_det.mean(dim=1, keepdim=True)
                fg = prior > 0.5
                bg = ~fg
                if bool(fg.any().item()):
                    self.last_gate_fg_mean = gate_voxel[fg].mean()
                if bool(bg.any().item()):
                    self.last_gate_bg_mean = gate_voxel[bg].mean()
        # Residual: refine the main feature conditioned on structure-derived gate.
        # Using self.refine(feat) ensures fg gate → enhanced fracture features
        # (previously struct_feat was refined, pushing it into all regions).
        return feat + gate * self.refine(feat)


class AnatomyConditionedFractureAttention(nn.Module):
    """Anatomy-Conditioned Fracture Attention (ACFA).

    Paper contribution: uses the predicted anatomy probability map (from the
    region + side heads) as a spatial prior to guide fracture feature refinement.

    Clinical motivation: pelvic fractures only occur on bone surfaces.
    Conditioning on anatomy probabilities focuses fracture detection on bone
    regions and suppresses false positives in soft tissue — in a differentiable,
    end-to-end trainable way.

    Design principles
    -----------------
    * Anatomy probs are *detached* from the gradient graph.  Fracture
      gradients must not corrupt the anatomy heads.
    * The gate is highest where anatomy foreground probability is high,
      so the residual refinement is applied selectively on bone voxels.
    * A learnable residual scale starts near zero and grows only if the
      signal is useful.  This makes ACFA safe to add without LR tuning.

    Ablation table columns
    ----------------------
    Baseline → + Hierarchy → + ACFA → + Fracture-aware loss
    """

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        anat_classes: int = 4,      # bg / sacrum / left_hip / right_hip
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

        # Step 1: embed anatomy probability map into feature space.
        # anat_classes channels → channels channels.
        self.anat_embed = nn.Sequential(
            conv_op(anat_classes, mid, kernel_size=1, bias=True),
            _norm(mid),
            nonlin(**nonlin_kwargs),
            conv_op(mid, channels, kernel_size=1, bias=True),
        )

        # Step 2: anatomy-conditioned spatial gate.
        # Concatenate [decoder feature, anatomy context] → sigmoid attention.
        # Higher gate values in bone regions (prior high) → focused refinement.
        self.gate = nn.Sequential(
            conv_op(channels * 2, mid, kernel_size=3, padding=1, bias=True),
            _norm(mid),
            nonlin(**nonlin_kwargs),
            conv_op(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Step 3: feature refinement within attended (anatomy-positive) regions.
        self.refine = nn.Sequential(
            conv_op(channels, channels, kernel_size=3, padding=1, bias=True),
            _norm(channels),
            nonlin(**nonlin_kwargs),
        )

        # Learnable residual scale: sigmoid(-4) ≈ 0.018 at init → almost identity.
        # Gradually increases as training verifies the refinement is beneficial.
        self.res_scale_raw = nn.Parameter(torch.tensor(-4.0))

        # Diagnostics — read by trainer for validation logging
        self.last_gate_mean: Optional[torch.Tensor] = None
        self.last_gate_fg_mean: Optional[torch.Tensor] = None
        self.last_gate_bg_mean: Optional[torch.Tensor] = None
        self.last_res_scale: Optional[torch.Tensor] = None

    def reset_parameters_for_stable_start(self) -> None:
        """Called by trainer after weight init to ensure near-identity start."""
        nn.init.constant_(self.res_scale_raw, -4.0)

    def forward(self, feat: torch.Tensor, anatomy_probs: torch.Tensor) -> torch.Tensor:
        out_dtype = feat.dtype

        # Detach anatomy — no gradient from fracture stream to anatomy heads.
        anat_p = anatomy_probs.detach().float()
        if anat_p.shape[2:] != feat.shape[2:]:
            anat_p = F.interpolate(anat_p, size=feat.shape[2:], mode="trilinear", align_corners=False)

        anat_ctx = self.anat_embed(anat_p.to(dtype=out_dtype))
        gate = self.gate(torch.cat([feat, anat_ctx], dim=1))

        # max residual scale = 0.30; sigmoid(-4) ≈ 0.018 at init.
        scale = 0.30 * torch.sigmoid(self.res_scale_raw)
        refined = self.refine(feat)
        out = feat + scale * gate * refined

        with torch.no_grad():
            self.last_gate_mean = gate.detach().mean()
            self.last_res_scale = scale.detach()
            fg = anat_p[:, 1:].sum(dim=1, keepdim=True) > 0.5
            gate_scalar = gate.detach().mean(dim=1, keepdim=True)
            if bool(fg.any()):
                self.last_gate_fg_mean = gate_scalar[fg].mean()
            bg = ~fg
            if bool(bg.any()):
                self.last_gate_bg_mean = gate_scalar[bg].mean()

        return out.to(dtype=out_dtype)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        voxels = int(np.prod(list(input_size), dtype=np.int64))
        mid = max(self.channels // 4, 8)
        return np.int64((mid * 2 + self.channels * 2) * voxels)


class FeatureReturnResDecoder(nn.Module):
    def __init__(
        self,
        encoder: UNetResEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        use_skip_se: bool = True,
        aux_highres_only: bool = True,
        compute_raw_semantic: bool = True,
    ) -> None:
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.encoder = encoder
        self.num_classes = int(num_classes)
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []
        asfeb_modules = []
        se_modules = []
        feature_channels = []
        self.use_skip_se = bool(use_skip_se)
        self.compute_raw_semantic = bool(compute_raw_semantic)

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
            se_modules.append(ChannelSEBlock(input_features_skip, reduction=4) if self.use_skip_se else nn.Identity())
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
            feature_channels.append(input_features_skip)

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.asfeb_modules = nn.ModuleList(asfeb_modules)
        self.se_modules = nn.ModuleList(se_modules)
        self.feature_channels_low_to_high = feature_channels
        self.feature_channels_high_to_low = list(reversed(feature_channels))
        self.highres_channels = int(feature_channels[-1])

    def forward(self, skips: List[torch.Tensor], return_features: bool = False):
        low_res_input = skips[-1]
        seg_outputs = []
        decoder_features = []
        for stage_id in range(len(self.stages)):
            x = self.upsample_layers[stage_id](low_res_input)
            skip_features = self.asfeb_modules[stage_id](skips[-(stage_id + 2)])
            skip_features = self.se_modules[stage_id](skip_features)
            x = torch.cat((x, skip_features), dim=1)
            x = self.stages[stage_id](x)
            decoder_features.append(x)
            if self.compute_raw_semantic:
                if self.deep_supervision:
                    seg_outputs.append(self.seg_layers[stage_id](x))
                elif stage_id == (len(self.stages) - 1):
                    seg_outputs.append(self.seg_layers[-1](x))
            low_res_input = x

        seg_outputs = seg_outputs[::-1]
        decoder_features = decoder_features[::-1]
        seg = None
        if self.compute_raw_semantic:
            seg = seg_outputs if self.deep_supervision else seg_outputs[0]
        if return_features:
            return seg, decoder_features
        return seg

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
            if self.compute_raw_semantic and (self.deep_supervision or stage_id == (len(self.stages) - 1)):
                output += _feature_count(self.num_classes, skip_size)
        return output


class AGSSUXlstmBotArconv(nn.Module):
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
        arconv_stage_idxs: Union[Tuple[int, ...], List[int]] = (2,),
        arconv_reg_weight: float = 1e-4,
        return_dict_in_eval: bool = False,
        use_um_fusion: bool = True,
        use_skip_se: bool = True,
        aux_highres_only: bool = True,
        use_sem_aux: bool = False,
        use_xlstm: bool = True,
        use_acfa: bool = False,   # ACFA: Anatomy-Conditioned Fracture Attention
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
            arconv_stage_idxs=arconv_stage_idxs,
            arconv_reg_weight=arconv_reg_weight,
        )
        self.use_xlstm = bool(use_xlstm)
        self.xlstm = ViLLayer(dim=features_per_stage[-1]) if self.use_xlstm else nn.Identity()
        # decoder raw semantic output can be disabled entirely for fast training
        self.decoder = FeatureReturnResDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            use_skip_se=use_skip_se,
            compute_raw_semantic=use_sem_aux,
        )
        self.deep_supervision = bool(deep_supervision)
        self.num_classes = int(num_classes)
        self.return_dict_in_eval = bool(return_dict_in_eval)
        self.use_um_fusion = bool(use_um_fusion)
        self.use_acfa = bool(use_acfa)
        self.aux_highres_only = bool(aux_highres_only)
        self.use_sem_aux = bool(use_sem_aux)

        if self.use_um_fusion and self.use_acfa:
            raise ValueError("use_um_fusion and use_acfa are mutually exclusive — pick one.")

        feat_channels = self.decoder.feature_channels_high_to_low
        self.struct_stems = nn.ModuleList([
            _conv_norm_act(conv_op, ch, ch, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, 3) for ch in feat_channels
        ])
        self.struct_heads = nn.ModuleList([conv_op(ch, 2, kernel_size=1, stride=1, padding=0, bias=True) for ch in feat_channels])
        self.frac_heads = nn.ModuleList([conv_op(ch, 1, kernel_size=1, stride=1, padding=0, bias=True) for ch in feat_channels])
        self.region_heads = nn.ModuleList([conv_op(ch, 3, kernel_size=1, stride=1, padding=0, bias=True) for ch in feat_channels])
        self.side_heads = nn.ModuleList([conv_op(ch, 3, kernel_size=1, stride=1, padding=0, bias=True) for ch in feat_channels])
        self.sacfrac_heads = nn.ModuleList([conv_op(ch, 1, kernel_size=1, stride=1, padding=0, bias=True) for ch in feat_channels])

        high_ch = self.decoder.highres_channels

        # UM-Fusion path (original, kept for backward compatibility)
        if self.use_um_fusion:
            self.um_fusion = UMAnatomyStructureFusion(conv_op, high_ch, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
            self.final_struct_stem = _conv_norm_act(conv_op, high_ch, high_ch, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, 3)
            self.final_struct_head = conv_op(high_ch, 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.final_frac_head = conv_op(high_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.final_region_head = conv_op(high_ch, 3, kernel_size=1, stride=1, padding=0, bias=True)
            self.final_side_head = conv_op(high_ch, 3, kernel_size=1, stride=1, padding=0, bias=True)
            self.final_sacfrac_head = conv_op(high_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.final_sem_aux_head = conv_op(high_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True) if self.use_sem_aux else None
            self.acfa = None
        # ACFA path (paper contribution: cleaner anatomy-conditioned attention)
        elif self.use_acfa:
            self.acfa = AnatomyConditionedFractureAttention(
                conv_op=conv_op,
                channels=high_ch,
                anat_classes=4,  # bg / sacrum / left_hip / right_hip
                reduction=4,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
            self.um_fusion = None
            self.final_struct_stem = None
            self.final_struct_head = None
            self.final_frac_head = None
            self.final_region_head = None
            self.final_side_head = None
            self.final_sacfrac_head = None
            self.final_sem_aux_head = None
        else:
            # Plain hierarchical assembly — no attention module at all.
            self.um_fusion = None
            self.acfa = None
            self.final_struct_stem = None
            self.final_struct_head = None
            self.final_frac_head = None
            self.final_region_head = None
            self.final_side_head = None
            self.final_sacfrac_head = None
            self.final_sem_aux_head = None

    def set_return_dict_in_eval(self, enabled: bool = True) -> None:
        self.return_dict_in_eval = bool(enabled)

    def enable_agss_inference_outputs(self) -> None:
        self.set_return_dict_in_eval(True)

    def _collect_arconv_regularization_loss(self) -> torch.Tensor:
        regs = []
        device = next(self.parameters()).device
        for module in self.modules():
            if isinstance(module, ARConv3D) and getattr(module, "reg_loss", None) is not None:
                regs.append(module.reg_loss)
        if len(regs) == 0:
            return torch.zeros((), device=device)
        return torch.stack([r.to(device) for r in regs]).sum()

    def get_arconv_regularization_loss(self) -> torch.Tensor:
        return self._collect_arconv_regularization_loss()

    def _collect_arconv_diagnostics(self) -> dict:
        device = next(self.parameters()).device
        keys = [
            ("arconv_offset_abs", "last_offset_abs"),
            ("arconv_offset_max", "last_offset_max"),
            ("arconv_offset_tv", "last_offset_tv"),
            ("arconv_offset_sat_ratio", "last_offset_sat_ratio"),
            ("arconv_routing_entropy", "last_routing_entropy"),
            ("arconv_routing_max_prob", "last_routing_max_prob"),
            ("arconv_routing_collapse_ratio", "last_routing_collapse_ratio"),
        ]
        buckets = {k: [] for k, _ in keys}
        count = 0
        for module in self.modules():
            if isinstance(module, ARConv3D):
                count += 1
                for k, attr in keys:
                    v = getattr(module, attr, None)
                    if torch.is_tensor(v):
                        buckets[k].append(v.detach().to(device))
        out = {"arconv_module_count": torch.tensor(float(count), device=device)}
        for k, vals in buckets.items():
            if len(vals) > 0:
                out[k] = torch.stack(vals).mean()
        return out

    @staticmethod
    def _replace_highres(output, highres_logits):
        if isinstance(output, (list, tuple)):
            return [highres_logits] + list(output[1:])
        return highres_logits

    @staticmethod
    def _format_aux_outputs(outputs: List[torch.Tensor], deep_supervision: bool):
        return outputs if deep_supervision else outputs[0]

    @staticmethod
    def _assemble_seg_outputs(region_outputs, side_outputs, frac_outputs):
        if isinstance(region_outputs, (list, tuple)):
            return [assemble_semantic_logits(r, s, f) for r, s, f in zip(region_outputs, side_outputs, frac_outputs)]
        return assemble_semantic_logits(region_outputs, side_outputs, frac_outputs)

    def forward(self, x: torch.Tensor, return_dict: Optional[bool] = None):
        if return_dict is None:
            return_dict = self.training or self.return_dict_in_eval

        skips = self.encoder(x)
        # ViLLayer is nn.Identity when use_xlstm=False — no-op, no overhead.
        skips[-1] = self.xlstm(skips[-1])
        sem_aux_pre, features = self.decoder(skips, return_features=True)
        high_feat = features[0]

        if self.aux_highres_only:
            struct_high_feat = self.struct_stems[0](high_feat)
            high_struct = self.struct_heads[0](struct_high_feat)

            # Initial predictions from raw decoder feature
            high_frac    = self.frac_heads[0](high_feat)
            high_region  = self.region_heads[0](high_feat)
            high_side    = self.side_heads[0](high_feat)
            high_sacfrac = self.sacfrac_heads[0](high_feat)

            # ── ACFA path ──────────────────────────────────────────────────
            # Use anatomy probs (from region+side) to re-attend fracture features.
            # Anatomy gradient is detached inside ACFA; anatomy heads are unchanged.
            if self.use_acfa and self.acfa is not None:
                _, _, anat_probs = anatomy_prior_and_uncertainty(high_region, high_side)
                frac_feat_refined = self.acfa(high_feat, anat_probs)
                high_frac    = self.frac_heads[0](frac_feat_refined)
                high_sacfrac = self.sacfrac_heads[0](frac_feat_refined)
                # struct uses the same refined feature for consistency
                struct_high_feat = self.struct_stems[0](frac_feat_refined)
                high_struct = self.struct_heads[0](struct_high_feat)
                sem_aux = (sem_aux_pre[0] if isinstance(sem_aux_pre, (list, tuple))
                           else sem_aux_pre) if self.use_sem_aux else None

            # ── UM-Fusion path (kept for backward compat / ablation) ────────
            elif self.use_um_fusion and self.um_fusion is not None:
                prior, uncertainty, _ = anatomy_prior_and_uncertainty(high_region, high_side)
                fused_high = self.um_fusion(high_feat, struct_high_feat, prior, uncertainty)
                high_struct  = self.final_struct_head(self.final_struct_stem(fused_high))
                high_frac    = self.final_frac_head(fused_high)
                high_region  = self.final_region_head(fused_high)
                high_side    = self.final_side_head(fused_high)
                high_sacfrac = self.final_sacfrac_head(fused_high)
                if self.use_sem_aux and self.final_sem_aux_head is not None:
                    sem_aux = self.final_sem_aux_head(fused_high)
                else:
                    sem_aux = None

            # ── Plain hierarchical path ─────────────────────────────────────
            else:
                sem_aux = (sem_aux_pre[0] if isinstance(sem_aux_pre, (list, tuple))
                           else sem_aux_pre) if self.use_sem_aux else None

            high_seg = assemble_semantic_logits(high_region, high_side, high_frac)
            if self.deep_supervision:
                lowres_seg = [
                    F.interpolate(high_seg, size=f.shape[2:], mode='trilinear', align_corners=False)
                    for f in features[1:]
                ]
                seg = [high_seg] + lowres_seg
            else:
                seg = high_seg

            struct_outputs  = high_struct
            frac_outputs    = high_frac
            region_outputs  = high_region
            side_outputs    = high_side
            sacfrac_outputs = high_sacfrac

        else:
            # Full deep-supervision path: auxiliary heads at every decoder stage
            struct_feats    = [stem(f) for stem, f in zip(self.struct_stems, features)]
            struct_outputs  = [head(f) for head, f in zip(self.struct_heads, struct_feats)]
            frac_outputs    = [head(f) for head, f in zip(self.frac_heads, features)]
            region_outputs  = [head(f) for head, f in zip(self.region_heads, features)]
            side_outputs    = [head(f) for head, f in zip(self.side_heads, features)]
            sacfrac_outputs = [head(f) for head, f in zip(self.sacfrac_heads, features)]

            if self.use_acfa and self.acfa is not None:
                _, _, anat_probs = anatomy_prior_and_uncertainty(region_outputs[0], side_outputs[0])
                frac_feat_refined = self.acfa(high_feat, anat_probs)
                struct_feats[0]    = self.struct_stems[0](frac_feat_refined)
                struct_outputs[0]  = self.struct_heads[0](struct_feats[0])
                frac_outputs[0]    = self.frac_heads[0](frac_feat_refined)
                sacfrac_outputs[0] = self.sacfrac_heads[0](frac_feat_refined)
                sem_aux = sem_aux_pre if self.use_sem_aux else None

            elif self.use_um_fusion and self.um_fusion is not None:
                prior, uncertainty, _ = anatomy_prior_and_uncertainty(region_outputs[0], side_outputs[0])
                fused_high = self.um_fusion(high_feat, struct_feats[0], prior, uncertainty)
                struct_outputs[0]  = self.final_struct_head(self.final_struct_stem(fused_high))
                frac_outputs[0]    = self.final_frac_head(fused_high)
                region_outputs[0]  = self.final_region_head(fused_high)
                side_outputs[0]    = self.final_side_head(fused_high)
                sacfrac_outputs[0] = self.final_sacfrac_head(fused_high)
                if self.use_sem_aux and self.final_sem_aux_head is not None:
                    sem_aux = self._replace_highres(sem_aux_pre, self.final_sem_aux_head(fused_high))
                else:
                    sem_aux = None
            else:
                sem_aux = sem_aux_pre if self.use_sem_aux else None

            seg = self._assemble_seg_outputs(region_outputs, side_outputs, frac_outputs)

        if not return_dict:
            return seg

        out = {
            "seg":     seg,
            "frac":    self._format_aux_outputs(frac_outputs,    self.deep_supervision),
            "region":  self._format_aux_outputs(region_outputs,  self.deep_supervision),
            "side":    self._format_aux_outputs(side_outputs,    self.deep_supervision),
            "sacfrac": self._format_aux_outputs(sacfrac_outputs, self.deep_supervision),
            "struct":  self._format_aux_outputs(struct_outputs,  self.deep_supervision),
            "arconv_reg": self._collect_arconv_regularization_loss(),
        }
        if self.use_sem_aux and sem_aux is not None:
            out["sem_aux"] = sem_aux
        out.update(self._collect_arconv_diagnostics())

        # ACFA diagnostics
        if self.use_acfa and self.acfa is not None:
            for key in ["last_gate_mean", "last_gate_fg_mean", "last_gate_bg_mean", "last_res_scale"]:
                value = getattr(self.acfa, key, None)
                if torch.is_tensor(value):
                    out["acfa_" + key.replace("last_", "")] = value

        # UM-Fusion diagnostics (only if UM-Fusion is active)
        if self.use_um_fusion and self.um_fusion is not None:
            for key in ["last_gate_mean", "last_gate_std", "last_gate_bg_mean", "last_gate_fg_mean"]:
                value = getattr(self.um_fusion, key, None)
                if torch.is_tensor(value):
                    out[key.replace("last_", "um_")] = value
        return out

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        output = self.encoder.compute_conv_feature_map_size(input_size)
        skip_sizes = self.decoder._encoder_skip_sizes(input_size)
        if len(skip_sizes) > 0:
            output += self.xlstm.compute_conv_feature_map_size(skip_sizes[-1])
        output += self.decoder.compute_conv_feature_map_size(input_size)
        for ch, size in zip(self.decoder.feature_channels_high_to_low, skip_sizes[:-1]):
            output += _feature_count(ch, size)
            output += _feature_count(2 + 1 + 3 + 3 + 1, size)
        return output
