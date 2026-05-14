"""
AGSS-Clean loss for hierarchical pelvic bone and fracture segmentation.

This version is strict with respect to structural information:
- D_core / D_surface are supervision targets only.
- small_weight and D_surface can weight the fracture loss.
- The network may predict a structure field, supervised by D_core/D_surface.
- No ground-truth structure field is used as network input.

Expected target layout after dataloader concatenation:
    0 semantic label: 0 bg, 1 sacrum, 2 left hip, 3 right hip, 4 fracture
    1 y_frac
    2 y_anat
    3 y_region
    4 y_side
    5 y_sacfrac
    6 D_core
    7 D_surface
    8 small_weight
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _highest_resolution(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _seg_only_target(target):
    if isinstance(target, (list, tuple)):
        return [t[:, 0:1] if torch.is_tensor(t) and t.ndim >= 5 and t.shape[1] > 1 else t for t in target]
    if torch.is_tensor(target) and target.ndim >= 5 and target.shape[1] > 1:
        return target[:, 0:1]
    return target


def _resize_target_like(t: torch.Tensor, pred: torch.Tensor, mode: str) -> torch.Tensor:
    if t.shape[2:] == pred.shape[2:]:
        return t
    if mode == "nearest":
        return F.interpolate(t.float(), size=pred.shape[2:], mode="nearest")
    return F.interpolate(t.float(), size=pred.shape[2:], mode="trilinear", align_corners=False)


def _finite_logits(logits: torch.Tensor, clamp: float = 30.0) -> torch.Tensor:
    return torch.nan_to_num(
        logits.float(),
        nan=0.0,
        posinf=float(clamp),
        neginf=-float(clamp),
    ).clamp(-float(clamp), float(clamp))


def _finite_loss(loss: torch.Tensor, nan: float = 0.0, posinf: float = 20.0, neginf: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(loss, nan=float(nan), posinf=float(posinf), neginf=float(neginf))


def _loss_downsample_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    factor: int,
    target_mode: str = "trilinear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample dense loss tensors to cut high-resolution auxiliary cost."""
    factor = int(factor)
    if factor <= 1 or min(pred.shape[2:]) < factor * 2:
        return pred, target
    spatial = tuple(max(1, int(i) // factor) for i in pred.shape[2:])
    dims = pred.ndim - 2
    pred_mode = "trilinear" if dims == 3 else "bilinear"
    pred_small = F.interpolate(pred, size=spatial, mode=pred_mode, align_corners=False)
    if target_mode == "nearest":
        target_small = F.interpolate(target.float(), size=spatial, mode="nearest")
    else:
        interp_mode = "trilinear" if dims == 3 else "bilinear"
        target_small = F.interpolate(target.float(), size=spatial, mode=interp_mode, align_corners=False)
    return pred_small, target_small


# ---------------------------------------------------------------------------
# Split AGSS target
# ---------------------------------------------------------------------------

def _split_agss_target(target, fracture_label: int = 4):
    """
    Return:
        sem, frac, anat, region, side, sacfrac, struct_fields, small_weight

    New target format has 9 channels after concatenating semantic + 8 AGSS aux.
    The function remains backward-compatible with semantic-only and 8-channel
    older aux layouts.
    """
    if isinstance(target, (list, tuple)):
        sems, fracs, anats, regions, sides, sacfracs, structs, small_weights = [], [], [], [], [], [], [], []
        for t in target:
            s, f, a, r, sd, sf, st, sw = _split_agss_target(t, fracture_label)
            sems.append(s)
            fracs.append(f)
            anats.append(a)
            regions.append(r)
            sides.append(sd)
            sacfracs.append(sf)
            structs.append(st)
            small_weights.append(sw)
        return sems, fracs, anats, regions, sides, sacfracs, structs, small_weights

    t = target
    sem = t[:, 0:1]

    if t.shape[1] >= 9:
        frac = t[:, 1:2]
        anat = t[:, 2:3]
        region = t[:, 3:4]
        side = t[:, 4:5]
        sacfrac = t[:, 5:6]
        struct_fields = t[:, 6:8]
        small_weight = t[:, 8:9]
    elif t.shape[1] >= 8:
        # semantic + old 7-channel AGSS aux: no small_weight channel
        frac = t[:, 1:2]
        anat = t[:, 2:3]
        region = t[:, 3:4]
        side = t[:, 4:5]
        sacfrac = t[:, 5:6]
        struct_fields = t[:, 6:8]
        small_weight = torch.ones_like(frac, dtype=torch.float32)
    else:
        # semantic only fallback
        frac = (sem == int(fracture_label)).float()
        anat = sem.clone()
        anat[anat == int(fracture_label)] = 0
        anat[(anat < 0) | (anat > 3)] = 0
        region = torch.zeros_like(sem)
        region[anat == 1] = 1
        region[(anat == 2) | (anat == 3)] = 2
        side = torch.zeros_like(sem)
        side[anat == 2] = 1
        side[anat == 3] = 2
        sacfrac = ((frac > 0.5) & (anat == 1)).float()
        struct_fields = torch.zeros((sem.shape[0], 2, *sem.shape[2:]), device=sem.device, dtype=torch.float32)
        small_weight = torch.ones_like(frac, dtype=torch.float32)

    return sem, frac, anat, region, side, sacfrac, struct_fields, small_weight


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def multiclass_ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    logits_f = _finite_logits(logits, clamp=30.0)
    target = _resize_target_like(target.float(), logits_f, "nearest")[:, 0].long()
    safe_target = target.clamp(0, num_classes - 1)

    ce = F.cross_entropy(logits_f, safe_target, reduction="mean")
    if float(dice_weight) <= 0:
        return _finite_loss(ce, posinf=20.0)

    prob = torch.softmax(logits_f, dim=1)
    onehot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 4, 1, 2, 3).to(prob.dtype)
    axes = (0,) + tuple(range(2, logits_f.ndim))
    inter = (prob * onehot).sum(axes)
    denom = prob.sum(axes) + onehot.sum(axes)
    dice_per_class = (2.0 * inter + 1e-6) / (denom + 1e-6)
    dice = 1.0 - dice_per_class[1:].mean()
    return _finite_loss(ce + float(dice_weight) * dice, posinf=20.0)


def focal_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    small_weight_target: torch.Tensor | None = None,
    small_component_weight: float = 0.0,
    struct_fields: torch.Tensor | None = None,
    geometry_weight: float = 0.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """
    Geometry-aware focal BCE + weighted Dice for fracture head.

    Voxel weight combines:
        (1 + geometry_weight * D_surface)
        (1 + small_component_weight * (small_weight - 1))
    and is clamped to [1, 8].
    """
    logits_f = _finite_logits(logits, clamp=30.0)
    target = _resize_target_like(target.float(), logits_f, "nearest")
    target = (target > 0.5).float()

    prob = torch.sigmoid(logits_f).clamp(1e-7, 1.0 - 1e-7)

    voxel_weight = torch.ones_like(target, dtype=logits_f.dtype)

    if struct_fields is not None and float(geometry_weight) > 0:
        sf = _resize_target_like(struct_fields.float(), logits_f, "trilinear")
        sf = torch.nan_to_num(sf, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        # channel 1 is D_surface if both core/surface are present
        surface = sf[:, 1:2] if sf.shape[1] > 1 else sf[:, 0:1]
        voxel_weight = voxel_weight * (1.0 + float(geometry_weight) * surface)

    if small_weight_target is not None and float(small_component_weight) > 0:
        sw = _resize_target_like(small_weight_target.float(), logits_f, "nearest")
        sw = torch.nan_to_num(sw, nan=1.0, posinf=4.0, neginf=1.0).clamp_min(1.0)
        voxel_weight = voxel_weight * (1.0 + float(small_component_weight) * (sw - 1.0))

    voxel_weight = voxel_weight.to(dtype=logits_f.dtype).clamp(1.0, 8.0)

    bce = F.binary_cross_entropy_with_logits(logits_f, target, reduction="none")
    p_t = target * prob + (1.0 - target) * (1.0 - prob)
    focal_weight = (1.0 - p_t).pow(float(gamma))
    alpha_t = target * float(alpha) + (1.0 - target) * (1.0 - float(alpha))
    focal_bce = (alpha_t * focal_weight * bce * voxel_weight).mean()

    if float(dice_weight) <= 0:
        return _finite_loss(focal_bce, posinf=20.0)

    axes = (0,) + tuple(range(2, logits_f.ndim))
    inter = (prob * target * voxel_weight).sum(axes)
    denom = (prob * voxel_weight).sum(axes) + (target * voxel_weight).sum(axes)
    dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()

    return _finite_loss(focal_bce + float(dice_weight) * dice, posinf=20.0)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits_f = _finite_logits(logits, clamp=30.0)
    target = _resize_target_like(target.float(), logits_f, "nearest")
    target = (target > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(logits_f, target)
    prob = torch.sigmoid(logits_f)
    axes = (0,) + tuple(range(2, logits_f.ndim))
    inter = (prob * target).sum(axes)
    denom = prob.sum(axes) + target.sum(axes)
    dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    return _finite_loss(bce + dice, posinf=20.0)


def structure_field_loss(
    struct_logits: torch.Tensor,
    struct_target: torch.Tensor,
    frac_target: torch.Tensor | None = None,
    downsample_factor: int = 1,
) -> torch.Tensor:
    """
    Supervise predicted structure fields:
        channel 0: D_core
        channel 1: D_surface
    Targets are continuous in [0, 1].
    """
    logits_f = _finite_logits(struct_logits, clamp=30.0)
    struct_target = _resize_target_like(struct_target.float(), logits_f, "trilinear")
    logits_f, struct_target = _loss_downsample_pair(
        logits_f, struct_target, int(downsample_factor), target_mode="trilinear"
    )
    struct_target = torch.nan_to_num(struct_target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    pred = torch.sigmoid(logits_f)
    roi = (struct_target.sum(1, keepdim=True) > 0.05).float()
    if frac_target is not None:
        frac_target = _resize_target_like(frac_target.float(), logits_f, "nearest")
        roi = torch.maximum(roi, (frac_target > 0.5).float())
    weight = 1.0 + 4.0 * roi

    bce = F.binary_cross_entropy_with_logits(logits_f, struct_target, reduction="none")
    bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    l1 = (torch.abs(pred - struct_target) * weight).sum() / weight.sum().clamp_min(1.0)
    return _finite_loss(bce + l1, posinf=20.0)


def region_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits_f = _finite_logits(logits, clamp=30.0)
    target = _resize_target_like(target.float(), logits_f, "nearest")[:, 0].long()
    safe_target = target.clamp(0, 2)
    return _finite_loss(F.cross_entropy(logits_f, safe_target, reduction="mean"), posinf=20.0)


def masked_side_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits_f = _finite_logits(logits, clamp=30.0)
    target = _resize_target_like(target.float(), logits_f, "nearest")[:, 0].long()
    safe_target = target.clamp(0, 2)
    mask = (target > 0).float()
    if mask.sum() < 8:
        return logits_f.new_tensor(0.0)
    ce_all = F.cross_entropy(logits_f, safe_target, reduction="none")
    return _finite_loss((ce_all * mask).sum() / mask.sum().clamp_min(1.0), posinf=20.0)


def anatomy_prior_penalty(
    frac_logits: torch.Tensor,
    region_logits: torch.Tensor,
    side_logits: torch.Tensor,
) -> torch.Tensor:
    region_logits_f = _finite_logits(region_logits, clamp=30.0)
    side_logits_f = _finite_logits(side_logits, clamp=30.0)
    frac_logits_f = _finite_logits(frac_logits, clamp=30.0)

    eps = 1e-7
    pr = torch.softmax(region_logits_f, dim=1)
    ps = torch.softmax(side_logits_f, dim=1)
    hip = pr[:, 2:3]
    side_lr = ps[:, 1:3]
    side_lr = side_lr / side_lr.sum(1, keepdim=True).clamp_min(eps)
    p_anat_fg = pr[:, 1:2] + hip * side_lr[:, 0:1] + hip * side_lr[:, 1:2]

    if p_anat_fg.shape[2:] != frac_logits_f.shape[2:]:
        p_anat_fg = F.interpolate(p_anat_fg, size=frac_logits_f.shape[2:], mode="trilinear", align_corners=False)

    p_frac = torch.sigmoid(frac_logits_f)
    return _finite_loss((p_frac * (1.0 - p_anat_fg).clamp_min(0)).mean(), posinf=20.0)


# ---------------------------------------------------------------------------
# Clean loss class
# ---------------------------------------------------------------------------

@dataclass
class CleanLossWeights:
    semantic: float = 1.0
    frac: float = 1.0
    anatomy: float = 0.30
    anatomy_late: float = 0.10
    sacfrac: float = 0.50
    prior: float = 0.02
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    semantic_dice: float = 1.0
    frac_dice: float = 1.0
    small_component: float = 0.25
    geometry: float = 0.50
    struct: float = 0.10
    geometry_start_epoch: int = 20
    small_component_start_epoch: int = 40
    struct_downsample_factor: int = 2


class AGSSCleanLoss(torch.nn.Module):
    """Clean loss for strict predicted-structure AGSS-Clean."""

    def __init__(
        self,
        weights: CleanLossWeights = CleanLossWeights(),
        fracture_label: int = 4,
        anatomy_schedule_start_epoch: int = 40,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.fracture_label = int(fracture_label)
        self.anatomy_schedule_start_epoch = int(anatomy_schedule_start_epoch)
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _anatomy_weight(self) -> float:
        if self.current_epoch < self.anatomy_schedule_start_epoch:
            return float(self.weights.anatomy)
        return float(self.weights.anatomy_late)

    def forward(self, output, target):
        if not isinstance(output, dict):
            target_seg = _highest_resolution(_seg_only_target(target))
            output_seg = _highest_resolution(output)
            return multiclass_ce_dice_loss(output_seg, target_seg, num_classes=output_seg.shape[1])

        sem_t, frac_t, anat_t, region_t, side_t, sacfrac_t, struct_t, small_weight_t = _split_agss_target(
            target, self.fracture_label
        )

        sem_t = _highest_resolution(sem_t)
        frac_t = _highest_resolution(frac_t)
        region_t = _highest_resolution(region_t)
        side_t = _highest_resolution(side_t)
        sacfrac_t = _highest_resolution(sacfrac_t)
        struct_t = _highest_resolution(struct_t)
        small_weight_t = _highest_resolution(small_weight_t)

        seg_pred = _highest_resolution(output["seg"])

        loss = self.weights.semantic * multiclass_ce_dice_loss(
            seg_pred,
            sem_t,
            num_classes=seg_pred.shape[1],
            dice_weight=self.weights.semantic_dice,
        )

        if "struct" in output and self.weights.struct > 0:
            loss = loss + self.weights.struct * structure_field_loss(
                _highest_resolution(output["struct"]),
                struct_t,
                frac_target=frac_t,
                downsample_factor=self.weights.struct_downsample_factor,
            )

        if "frac" in output and self.weights.frac > 0:
            geometry_w = self.weights.geometry if self.current_epoch >= self.weights.geometry_start_epoch else 0.0
            small_w = (
                self.weights.small_component
                if self.current_epoch >= self.weights.small_component_start_epoch
                else 0.0
            )
            loss = loss + self.weights.frac * focal_bce_dice_loss(
                _highest_resolution(output["frac"]),
                frac_t,
                alpha=self.weights.focal_alpha,
                gamma=self.weights.focal_gamma,
                small_weight_target=small_weight_t,
                small_component_weight=small_w,
                struct_fields=struct_t,
                geometry_weight=geometry_w,
                dice_weight=self.weights.frac_dice,
            )

        anat_w = self._anatomy_weight()
        if anat_w > 0:
            if "region" in output:
                loss = loss + anat_w * region_ce_loss(_highest_resolution(output["region"]), region_t)
            if "side" in output:
                loss = loss + anat_w * masked_side_ce_loss(_highest_resolution(output["side"]), side_t)

        if "sacfrac" in output and self.weights.sacfrac > 0:
            loss = loss + self.weights.sacfrac * bce_dice_loss(_highest_resolution(output["sacfrac"]), sacfrac_t)

        if self.weights.prior > 0 and "frac" in output and "region" in output and "side" in output:
            loss = loss + self.weights.prior * anatomy_prior_penalty(
                _highest_resolution(output["frac"]),
                _highest_resolution(output["region"]),
                _highest_resolution(output["side"]),
            )

        return _finite_loss(loss, posinf=20.0)
