"""
SDAF-AGDSNet V2 network extension.

Compared with the first runnable SDAF version, this version adds:
1) multi-scale structural/affinity auxiliary heads for deep supervision;
2) a lightweight anatomy/foreground-uncertainty modulated structural fusion block;
3) forward-local ARConv regularization captured into output['arconv_reg'];
4) an explicit return_dict argument/helper for instance post-processing.
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
from nnunetv2.net.YuNet.sdaf_auxiliary import DEFAULT_AFFINITY_OFFSETS


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


def anatomy_prior_and_uncertainty_from_logits(
    logits: torch.Tensor,
    fracture_start_channel: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a soft anatomical/foreground prior A and uncertainty U from preliminary logits.

    If anatomy labels exist and channels are ordered as 0=background, 1..fracture_start-1=bone anatomy,
    A is the probability of anatomical bone classes. For binary segmentation, A becomes foreground
    probability and U becomes Bernoulli uncertainty. This keeps the module usable for both binary and
    anatomy+fragment label settings.
    """
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits)
        uncertainty = 4.0 * p * (1.0 - p)
        return p.clamp(0, 1), uncertainty.clamp(0, 1)

    probs = torch.softmax(logits, dim=1)
    c = probs.shape[1]
    fs = int(fracture_start_channel)
    if c > fs and fs > 1:
        prior = probs[:, 1:fs].sum(1, keepdim=True).clamp(0, 1)
    else:
        prior = probs[:, 1:].sum(1, keepdim=True).clamp(0, 1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(1, keepdim=True)
    entropy = entropy / math.log(max(c, 2))
    return prior, entropy.clamp(0, 1)


class UMStructuralFusion(nn.Module):
    """Uncertainty-modulated structural fusion for the highest-resolution decoder feature."""

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
        self.refine = _conv_norm_act(conv_op, channels, channels, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, 3)
        self.last_gate_mean = None
        self.last_gate_std = None

    def forward(
        self,
        frac_feat: torch.Tensor,
        struct_feat: torch.Tensor,
        anatomy_prior: torch.Tensor,
        anatomy_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        if anatomy_prior.shape[2:] != frac_feat.shape[2:]:
            anatomy_prior = F.interpolate(anatomy_prior, size=frac_feat.shape[2:], mode="trilinear", align_corners=False)
        if anatomy_uncertainty.shape[2:] != frac_feat.shape[2:]:
            anatomy_uncertainty = F.interpolate(anatomy_uncertainty, size=frac_feat.shape[2:], mode="trilinear", align_corners=False)
        g = self.gate(torch.cat([frac_feat, struct_feat, anatomy_prior, anatomy_uncertainty], dim=1))
        self.last_gate_mean = g.detach().mean()
        self.last_gate_std = g.detach().std()
        return frac_feat + g * self.refine(struct_feat)


class FeatureReturnResDecoder(nn.Module):
    """nnU-Net style residual decoder that also returns high-to-low decoder features."""

    def __init__(
        self,
        encoder: UNetResEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
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
        feature_channels = []

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
            feature_channels.append(input_features_skip)

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.asfeb_modules = nn.ModuleList(asfeb_modules)
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
            x = torch.cat((x, skip_features), dim=1)
            x = self.stages[stage_id](x)
            decoder_features.append(x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[stage_id](x))
            elif stage_id == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            low_res_input = x

        seg_outputs = seg_outputs[::-1]
        decoder_features = decoder_features[::-1]
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
            if self.deep_supervision or stage_id == (len(self.stages) - 1):
                output += _feature_count(self.num_classes, skip_size)
        return output


class SDAFUXlstmBotArconv(nn.Module):
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
        affinity_offsets: Sequence[Tuple[int, int, int]] = DEFAULT_AFFINITY_OFFSETS,
        return_dict_in_eval: bool = False,
        use_um_fusion: bool = True,
        fracture_start_channel: int = 4,
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
        self.xlstm = ViLLayer(dim=features_per_stage[-1])
        self.decoder = FeatureReturnResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        self.deep_supervision = bool(deep_supervision)
        self.num_classes = int(num_classes)
        self.return_dict_in_eval = bool(return_dict_in_eval)
        self.use_um_fusion = bool(use_um_fusion)
        self.fracture_start_channel = int(fracture_start_channel)
        self.affinity_offsets = tuple(tuple(int(i) for i in o) for o in affinity_offsets)

        feat_channels = self.decoder.feature_channels_high_to_low
        self.struct_stems = nn.ModuleList(
            [_conv_norm_act(conv_op, ch, ch, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, 3) for ch in feat_channels]
        )
        self.struct_heads = nn.ModuleList(
            [conv_op(ch, 3, kernel_size=1, stride=1, padding=0, bias=True) for ch in feat_channels]
        )
        self.affinity_heads = nn.ModuleList(
            [conv_op(ch, len(self.affinity_offsets), kernel_size=1, stride=1, padding=0, bias=True) for ch in feat_channels]
        )

        high_ch = self.decoder.highres_channels
        self.um_fusion = UMStructuralFusion(conv_op, high_ch, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
        self.final_seg_head = conv_op(high_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def set_return_dict_in_eval(self, enabled: bool = True) -> None:
        self.return_dict_in_eval = bool(enabled)

    def enable_sdaf_inference_outputs(self) -> None:
        self.set_return_dict_in_eval(True)

    def _collect_arconv_regularization_loss(self) -> torch.Tensor:
        regs = []
        device = next(self.parameters()).device
        for module in self.modules():
            if isinstance(module, ARConv3D) and module.reg_loss is not None:
                regs.append(module.reg_loss)
        if len(regs) == 0:
            return torch.zeros((), device=device)
        return torch.stack([r.to(device) for r in regs]).sum()

    def get_arconv_regularization_loss(self) -> torch.Tensor:
        # Legacy helper. Prefer output['arconv_reg'] in loss to avoid stale state.
        return self._collect_arconv_regularization_loss()


    def _collect_arconv_diagnostics(self) -> dict:
        """Collect detached ARConv diagnostics for validation logging.

        Note: arconv_reg is expected to be zero in eval mode because the
        regularization term is only computed during training. These diagnostics
        are computed in both train and eval mode, so they reveal whether ARConv
        offsets and routing are actually active.
        """
        device = next(self.parameters()).device
        buckets = {
            "arconv_offset_abs": [],
            "arconv_offset_max": [],
            "arconv_offset_tv": [],
            "arconv_routing_entropy": [],
            "arconv_routing_max_prob": [],
        }
        count = 0
        for module in self.modules():
            if isinstance(module, ARConv3D):
                count += 1
                for key, attr in [
                    ("arconv_offset_abs", "last_offset_abs"),
                    ("arconv_offset_max", "last_offset_max"),
                    ("arconv_offset_tv", "last_offset_tv"),
                    ("arconv_routing_entropy", "last_routing_entropy"),
                    ("arconv_routing_max_prob", "last_routing_max_prob"),
                ]:
                    value = getattr(module, attr, None)
                    if torch.is_tensor(value):
                        buckets[key].append(value.detach().to(device))
        out = {"arconv_module_count": torch.tensor(float(count), device=device)}
        for key, values in buckets.items():
            if len(values) > 0:
                out[key] = torch.stack(values).mean()
        return out

    @staticmethod
    def _replace_highres_seg(seg_output, highres_logits):
        if isinstance(seg_output, (list, tuple)):
            return [highres_logits] + list(seg_output[1:])
        return highres_logits

    def forward(self, x: torch.Tensor, return_dict: Optional[bool] = None):
        if return_dict is None:
            return_dict = self.training or self.return_dict_in_eval

        skips = self.encoder(x)
        skips[-1] = self.xlstm(skips[-1])
        seg_pre, features = self.decoder(skips, return_features=True)

        struct_feats = [stem(f) for stem, f in zip(self.struct_stems, features)]
        struct_outputs = [head(f) for head, f in zip(self.struct_heads, struct_feats)]

        high_pre = seg_pre[0] if isinstance(seg_pre, (list, tuple)) else seg_pre
        if self.use_um_fusion:
            anatomy_prior, anatomy_uncertainty = anatomy_prior_and_uncertainty_from_logits(
                high_pre, fracture_start_channel=self.fracture_start_channel
            )
            fused_high = self.um_fusion(features[0], struct_feats[0], anatomy_prior, anatomy_uncertainty)
            high_logits = self.final_seg_head(fused_high)
            affinity_high = self.affinity_heads[0](fused_high)
        else:
            high_logits = high_pre
            affinity_high = self.affinity_heads[0](struct_feats[0])

        seg = self._replace_highres_seg(seg_pre, high_logits)
        affinity_outputs = [affinity_high] + [head(f) for head, f in zip(self.affinity_heads[1:], struct_feats[1:])]

        # In normal eval/predict mode we return pure segmentation logits for nnU-Net compatibility.
        if not return_dict:
            return seg

        if not self.deep_supervision:
            struct_out = struct_outputs[0]
            affinity_out = affinity_outputs[0]
        else:
            struct_out = struct_outputs
            affinity_out = affinity_outputs

        out = {
            "seg": seg,
            "struct": struct_out,
            "affinity": affinity_out,
            "arconv_reg": self._collect_arconv_regularization_loss(),
        }
        out.update(self._collect_arconv_diagnostics())
        if self.use_um_fusion:
            if getattr(self.um_fusion, "last_gate_mean", None) is not None:
                out["um_gate_mean"] = self.um_fusion.last_gate_mean
            if getattr(self.um_fusion, "last_gate_std", None) is not None:
                out["um_gate_std"] = self.um_fusion.last_gate_std
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
            output += _feature_count(3 + len(self.affinity_offsets), size)
        return output
