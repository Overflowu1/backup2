import torch
import numpy as np
from sympy.abc import alpha

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
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
        dc = torch.nan_to_num(dc,nan=0.0,posinf=0.0,neginf=0.0)
        dc = dc / (dc.max()+1e-8)
        loss_over_idc = torch.einsum("bcdhw,bcdhw->bcdhw", pc, dc)  # 点乘
        surface_loss = torch.mean(loss_over_idc)
        return surface_loss

class DC_and_CE_and_Surface_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, surface_kwargs, weight_ce=1, weight_dice=1, weight_surface=1,
                 ignore_label=None, dice_class=SoftDiceLoss):
        """
        Weights for CE, Dice, and Surface loss do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param surface_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_surface:
        :param ignore_label:
        :param dice_class:
        """
        super(DC_and_CE_and_Surface_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_surface = weight_surface
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.surface_loss = SurfaceLoss(**surface_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor,epoch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_and_Surface_loss)'
            mask = target != self.ignore_label
            target_surface = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_surface = target
            mask = None

        # Compute Dice Loss
        dc_loss = self.dc(net_output, target_surface, loss_mask=mask) if self.weight_dice != 0 else 0

        # Compute CE Loss
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 and (
                    self.ignore_label is None or num_fg > 0) else 0

        # Compute Surface Loss

        surface_loss = self.surface_loss(net_output, target) if self.weight_surface != 0 else 0
        # alpha = 1.0 * (epoch + 1) / 500
        alpha = 0.05
        # print(f"Epoch {epoch + 1}, alpha: {alpha}")

        # Total loss
        result = self.weight_ce*(1-alpha) * ce_loss + self.weight_dice*(1-alpha) * dc_loss + self.weight_surface*alpha * surface_loss
        # result = self.weight_ce* ce_loss + self.weight_dice * dc_loss + self.weight_surface * surface_loss

        return result

    # def compute_distance_map(self, target: torch.Tensor, label_index: int = 1) -> torch.Tensor:
    #     """
    #     计算目标张量的距离图。
    #     :param target: 目标张量，形状为 (B, C, D, H, W)
    #     :param label_index: 目标标签的索引（默认为1）
    #     :return: 距离图张量，形状为 (B, C, D, H, W)
    #     """
    #     B, C, D, H, W = target.shape
    #     distance_maps = torch.zeros_like(target)
    #
    #     for b in range(B):
    #         for c in range(C):
    #             # 提取单个样本和通道
    #             sample = target[b, c]
    #
    #             # 计算所有非零位置（前景）
    #             coords = torch.nonzero(sample == label_index)
    #
    #             # 构造网格
    #             grid = torch.stack(torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W)), dim=-1)
    #             grid = grid.view(-1, 3).float()
    #
    #             # 确保 grid 和 coords 在同一个设备上
    #             grid = grid.to(coords.device)
    #
    #             # 计算距离
    #             if coords.numel() > 0:
    #                 coords = coords.float()
    #                 distances = torch.cdist(grid, coords, p=2)  # 使用欧几里得距离
    #                 min_distances = distances.min(dim=1)[0]
    #                 distance_maps[b, c] = min_distances.view(D, H, W)
    #             else:
    #                 distance_maps[b, c] = torch.zeros_like(sample)
    #
    #     return distance_maps

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
