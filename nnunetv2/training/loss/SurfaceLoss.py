from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class SurfaceLoss(nn.Module):
    def __init__(self, idc):
        super(SurfaceLoss, self).__init__()
        self.idc = idc

    def forward(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        计算 Surface Loss
        :param probs: 模型输出的概率图，形状为 (B, C, D, H, W)
        :param dist_maps: 距离图，形状为 (B, C, D, H, W)
        :return: Surface Loss 值
        """
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        loss_over_idc = torch.einsum("bcdhw,bcdhw->bcdhw", pc, dc)  # 点乘
        surface_loss = torch.mean(loss_over_idc)
        return surface_loss
