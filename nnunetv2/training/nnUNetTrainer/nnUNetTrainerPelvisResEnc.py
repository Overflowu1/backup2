# nnUNetTrainerPelvisBasic.py
# -*- coding: utf-8 -*-
"""
Pelvis basic trainer for your local nnU-Net v2 fork.

Goal:
    Use the most basic nnU-Net architecture, not a hand-written network.

Main changes:
    1) Do NOT override build_network_architecture.
       The default nnU-Net network from get_network_from_plans is used.
    2) Disable custom AGSS/coordinate-map auxiliary input channels.
       Your local dataloader adds 3 coordinate channels by default, causing:
           network input channels = 1
           actual data channels   = 4
       This trainer forces those auxiliary channels off.
    3) Increase foreground oversampling for sacrum/hip_bone segmentation.

Recommended first run:
    -tr nnUNetTrainerPelvisBasic

Optional:
    -tr nnUNetTrainerPelvisBasicBoundary
"""

import os
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


def _force_basic_nnunet_input():
    """
    Your local fork has a modified 3D dataloader that can append auxiliary
    channels such as z/y/x coordinate maps. For a basic nnU-Net run we disable
    them so that the dataloader channel count matches dataset.json/plans.

    These env vars are read at dataloader runtime, so setting them in trainer
    __init__ and on_train_start is sufficient.
    """
    os.environ["NNUNET_AGSS_COORD_MAP"] = "0"
    os.environ["NNUNET_AGSS_USE_PRECOMPUTED"] = "0"


class nnUNetTrainerPelvisBasic(nnUNetTrainer):
    """
    Basic and stable pelvis trainer.

    Labels expected:
        0 background
        1 sacrum
        2 hip_bone

    Network:
        default nnU-Net network from the original nnUNetTrainer.

    Loss:
        default nnU-Net Dice + CE loss.

    Sampling:
        foreground oversampling increased from 0.33 to 0.50.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        _force_basic_nnunet_input()

        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )

        # More patches centered on foreground. This helps when sacrum/hip_bone
        # occupy a relatively small part of the intraoperative CT volume.
        self.oversample_foreground_percent = 0.50

        # Re-apply oversampling logic for DDP compatibility. For non-DDP this
        # only keeps batch_size consistent with the plans.
        self._set_batch_size_and_oversample()

        self.print_to_log_file(
            "[PelvisBasic] Using default nnU-Net architecture; "
            "disabled NNUNET_AGSS_COORD_MAP and NNUNET_AGSS_USE_PRECOMPUTED; "
            f"oversample_foreground_percent={self.oversample_foreground_percent}",
            also_print_to_console=True,
        )

    def on_train_start(self):
        # Set again immediately before dataloader creation.
        _force_basic_nnunet_input()
        return super().on_train_start()

    def train_step(self, batch: dict) -> dict:
        """
        Same as nnUNetTrainer.train_step, but with a clearer channel mismatch error.
        """
        data = batch["data"]
        if self.num_input_channels is not None and data.shape[1] != self.num_input_channels:
            raise RuntimeError(
                "[PelvisBasic] Channel mismatch before network forward: "
                f"network was built for {self.num_input_channels} input channel(s), "
                f"but dataloader produced {data.shape[1]} channel(s). "
                "For the most basic nnU-Net run, make sure these are disabled before launching Python:\n"
                "  export NNUNET_AGSS_COORD_MAP=0\n"
                "  export NNUNET_AGSS_USE_PRECOMPUTED=0\n"
                "If you intentionally want coordinate-map channels, the network must also be built with +3 input channels."
            )
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        if self.num_input_channels is not None and data.shape[1] != self.num_input_channels:
            raise RuntimeError(
                "[PelvisBasic] Validation channel mismatch: "
                f"network expects {self.num_input_channels}, dataloader produced {data.shape[1]}."
            )
        return super().validation_step(batch)


# -------------------------------------------------------------------------
# Optional boundary-aware variant
# -------------------------------------------------------------------------

class BoundaryDiceLoss(nn.Module):
    """
    Lightweight boundary Dice loss.
    Use this only after nnUNetTrainerPelvisBasic can train normally.
    """

    def __init__(self, smooth: float = 1e-5, include_background: bool = False, ignore_label=None):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.ignore_label = ignore_label

    @staticmethod
    def _morph_gradient(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            pool = F.max_pool3d
        elif x.ndim == 4:
            pool = F.max_pool2d
        else:
            raise RuntimeError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

        dilated = pool(x, kernel_size=3, stride=1, padding=1)
        eroded = -pool(-x, kernel_size=3, stride=1, padding=1)
        return torch.clamp(dilated - eroded, 0.0, 1.0)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()

        valid_mask = None
        if self.ignore_label is not None:
            valid_mask = target != int(self.ignore_label)
            target = torch.where(valid_mask, target, torch.zeros_like(target))

        target = torch.clamp(target, 0, num_classes - 1)

        target_oh = F.one_hot(target, num_classes=num_classes)
        target_oh = target_oh.movedim(-1, 1).float()

        if not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        pred_b = self._morph_gradient(probs)
        target_b = self._morph_gradient(target_oh)

        if valid_mask is not None:
            valid_mask = valid_mask[:, None].float()
            pred_b = pred_b * valid_mask
            target_b = target_b * valid_mask

        reduce_axes = tuple(range(2, pred_b.ndim))
        intersection = (pred_b * target_b).sum(dim=reduce_axes)
        denominator = pred_b.sum(dim=reduce_axes) + target_b.sum(dim=reduce_axes)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class PelvisBoundaryCompoundLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, boundary_weight: float = 0.05, ignore_label=None):
        super().__init__()
        self.base_loss = base_loss
        self.boundary_loss = BoundaryDiceLoss(
            include_background=False,
            ignore_label=ignore_label,
        )
        self.boundary_weight = float(boundary_weight)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base_loss(net_output, target) + self.boundary_weight * self.boundary_loss(net_output, target)


class nnUNetTrainerPelvisBasicBoundary(nnUNetTrainerPelvisBasic):
    """
    Default nnU-Net architecture + DiceCE + small boundary Dice.

    Try this only after nnUNetTrainerPelvisBasic has successfully trained.
    """

    def _build_loss(self):
        if self.label_manager.has_regions:
            # Keep default behavior for region-based training.
            # Your current sacrum/hip_bone task should not enter this branch.
            return super()._build_loss()

        base_loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        loss = PelvisBoundaryCompoundLoss(
            base_loss=base_loss,
            boundary_weight=0.05,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
