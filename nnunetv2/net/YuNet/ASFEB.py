"""
Adaptive Spatial Feature Enhancement Block (ASFEB).

This version removes the file-level test code and replaces hard-coded BatchNorm3d
with a configurable normalization operator, so it is safer for small-batch 3D nnU-Net
training. It is intended for decoder skip-feature refinement rather than as the main
claimed innovation.
"""
from __future__ import annotations

from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFEB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        out_channels = int(out_channels or in_channels)
        norm_op_kwargs = dict(norm_op_kwargs or {"eps": 1e-5, "affine": True})
        nonlin_kwargs = dict(nonlin_kwargs or {"inplace": True})
        hidden_channels = max(out_channels // reduction, 8)

        def norm(c: int) -> nn.Module:
            return nn.Identity() if norm_op is None else norm_op(c, **norm_op_kwargs)

        self.local_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            norm(out_channels),
            nonlin(**nonlin_kwargs),
        )

        self.pool_fusion = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            norm(out_channels),
            nonlin(**nonlin_kwargs),
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, hidden_channels, kernel_size=1, bias=True),
            nonlin(**nonlin_kwargs),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x_local = self.local_conv(x)

        pooled = torch.cat(
            (
                F.max_pool3d(x_local, kernel_size=3, stride=1, padding=1),
                F.avg_pool3d(x_local, kernel_size=3, stride=1, padding=1),
            ),
            dim=1,
        )
        x_fused = self.pool_fusion(pooled)

        channel_weight = self.channel_attention(x_fused)
        avg_map = torch.mean(x_fused, dim=1, keepdim=True)
        max_map = torch.max(x_fused, dim=1, keepdim=True)[0]
        spatial_weight = self.spatial_attention(torch.cat((avg_map, max_map), dim=1))

        return residual + x_fused * channel_weight * spatial_weight

    def compute_conv_feature_map_size(self, input_size) -> int:
        voxels = int(np.prod(input_size, dtype=np.int64))
        # local conv + pool fusion + spatial attention + shortcut/attention approx.
        return int(voxels * 4)
