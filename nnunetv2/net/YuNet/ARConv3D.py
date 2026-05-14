
"""
Geometry-adaptive anisotropic 3D convolution with lightweight diagnostics.

This file is backward compatible with the AGSS/SDAF trainers but adds:
- offset saturation ratio
- routing collapse ratio
which are useful to verify whether ARConv has degenerated into always-max-offset
sampling or single-branch routing.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Size3 = Union[int, Sequence[int]]


def _to_3tuple(value: Size3) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    value = tuple(value)
    if len(value) != 3:
        raise ValueError(f"Expected int or 3-tuple/list, got {value}")
    return int(value[0]), int(value[1]), int(value[2])


def _tv_loss_3d(x: torch.Tensor) -> torch.Tensor:
    loss = x.new_tensor(0.0)
    terms = 0
    if x.shape[2] > 1:
        loss = loss + (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
        terms += 1
    if x.shape[3] > 1:
        loss = loss + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        terms += 1
    if x.shape[4] > 1:
        loss = loss + (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()
        terms += 1
    return loss / max(terms, 1)


class ARConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3 = 3,
        stride: Size3 = 1,
        padding: Size3 = 1,
        d_max: float = 0.5,   # halved: prevents the Tanh-saturated regime
        h_max: float = 0.5,
        w_max: float = 0.5,
        kernel_list: Optional[Iterable[Tuple[int, int, int]]] = None,
        reg_weight: float = 1e-3,
        tv_weight: float = 0.5,
        sat_weight: float = 1.0,   # 4× stronger saturation penalty
        conv_bias: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _to_3tuple(kernel_size)
        self.stride = _to_3tuple(stride)
        self.padding = _to_3tuple(padding)
        self.d_max = float(d_max)
        self.h_max = float(h_max)
        self.w_max = float(w_max)
        self.reg_weight = float(reg_weight)
        self.tv_weight = float(tv_weight)
        self.sat_weight = float(sat_weight)

        if kernel_list is None:
            kernel_list = ((3, 3, 3), (3, 5, 5), (5, 3, 5), (5, 5, 3))
        self.kernel_list = [tuple(int(i) for i in k) for k in kernel_list]

        self.input_proj = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=conv_bias,
            ),
            nn.LeakyReLU(inplace=True),
        )
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()

        self.offset_head = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(self.out_channels, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )
        self.modulation_head = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.bias_head = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, bias=True)
        self.routing_head = nn.Conv3d(self.out_channels, len(self.kernel_list), kernel_size=1, bias=True)

        self.branches = nn.ModuleList(
            [
                nn.Conv3d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=k,
                    stride=1,
                    padding=tuple(ki // 2 for ki in k),
                    bias=conv_bias,
                )
                for k in self.kernel_list
            ]
        )

        self.reg_loss: Optional[torch.Tensor] = None

        # Diagnostics read by trainer
        self.last_offset_abs: Optional[torch.Tensor] = None
        self.last_offset_max: Optional[torch.Tensor] = None
        self.last_offset_tv: Optional[torch.Tensor] = None
        self.last_offset_sat_ratio: Optional[torch.Tensor] = None
        self.last_routing_entropy: Optional[torch.Tensor] = None
        self.last_routing_max_prob: Optional[torch.Tensor] = None
        self.last_routing_collapse_ratio: Optional[torch.Tensor] = None

        self.reset_parameters_for_stable_start()

    def reset_parameters_for_stable_start(self) -> None:
        final_offset_conv = self.offset_head[-2]
        if isinstance(final_offset_conv, nn.Conv3d):
            nn.init.zeros_(final_offset_conv.weight)
            nn.init.zeros_(final_offset_conv.bias)
        nn.init.zeros_(self.routing_head.weight)
        nn.init.zeros_(self.routing_head.bias)
        final_mod_conv = self.modulation_head[0]
        nn.init.zeros_(final_mod_conv.weight)
        nn.init.zeros_(final_mod_conv.bias)

    @staticmethod
    def _make_base_grid(
        batch_size: int,
        depth: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        z = torch.linspace(-1.0, 1.0, depth, device=device, dtype=dtype) if depth > 1 else torch.zeros(1, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype) if height > 1 else torch.zeros(1, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype) if width > 1 else torch.zeros(1, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack((xx, yy, zz), dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

    @staticmethod
    def _normalize_offsets(offset: torch.Tensor) -> torch.Tensor:
        _, _, depth, height, width = offset.shape
        d_scale = 0.0 if depth <= 1 else 2.0 / (depth - 1)
        h_scale = 0.0 if height <= 1 else 2.0 / (height - 1)
        w_scale = 0.0 if width <= 1 else 2.0 / (width - 1)
        d_norm = offset[:, 0] * d_scale
        h_norm = offset[:, 1] * h_scale
        w_norm = offset[:, 2] * w_scale
        return torch.stack((w_norm, h_norm, d_norm), dim=-1)

    def _compute_regularization(self, offset: torch.Tensor) -> torch.Tensor:
        magnitude = offset.abs().mean()
        smoothness = _tv_loss_3d(offset)
        # softly penalize offsets close to the maximum range to avoid full-map saturation
        sat_thresh = offset.new_tensor([self.d_max, self.h_max, self.w_max]).view(1, 3, 1, 1, 1) * 0.70
        sat_penalty = torch.relu(offset.abs() - sat_thresh).mean()
        return self.reg_weight * (magnitude + self.tv_weight * smoothness + self.sat_weight * sat_penalty)

    def _update_diagnostics(self, offset: torch.Tensor, routing: torch.Tensor) -> None:
        with torch.no_grad():
            self.last_offset_abs = offset.abs().mean()
            self.last_offset_max = offset.abs().amax()
            self.last_offset_tv = _tv_loss_3d(offset)

            max_vals = offset.new_tensor([self.d_max, self.h_max, self.w_max]).view(1, 3, 1, 1, 1).clamp_min(1e-6)
            self.last_offset_sat_ratio = (offset.abs() > (0.95 * max_vals)).float().mean()

            routing_prob = routing.clamp_min(1e-8)
            entropy = -(routing_prob * torch.log(routing_prob)).sum(dim=1)
            max_prob = routing_prob.max(dim=1).values
            self.last_routing_entropy = entropy.mean()
            self.last_routing_max_prob = max_prob.mean()
            self.last_routing_collapse_ratio = (max_prob > 0.9).float().mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.dropout(self.input_proj(x))
        b, _, d, h, w = base.shape

        raw_offset = self.offset_head(base)
        scale = raw_offset.new_tensor([self.d_max, self.h_max, self.w_max]).view(1, 3, 1, 1, 1)
        offset = raw_offset * scale

        if self.training and self.reg_weight > 0:
            self.reg_loss = self._compute_regularization(offset)
        else:
            self.reg_loss = None

        grid = self._make_base_grid(b, d, h, w, base.device, base.dtype)
        grid = grid + self._normalize_offsets(offset)

        sampled = F.grid_sample(
            base,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        routing = F.softmax(self.routing_head(base), dim=1)
        self._update_diagnostics(offset, routing)

        routed = sampled.new_tensor(0.0)
        for branch_id, branch in enumerate(self.branches):
            routed = routed + routing[:, branch_id:branch_id + 1] * branch(sampled)

        modulation = self.modulation_head(base)
        bias = self.bias_head(base)
        return routed * modulation + bias

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> int:
        input_size = tuple(int(i) for i in input_size)
        out_size = []
        for i, k, p, s in zip(input_size, self.kernel_size, self.padding, self.stride):
            out_size.append((i + 2 * p - k) // s + 1)
        voxels = int(torch.tensor(out_size).prod().item())
        n_branch = len(self.branches)
        return int(voxels * self.out_channels * (4 + n_branch))
