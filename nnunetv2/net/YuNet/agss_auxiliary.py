
"""
AGSS v3 auxiliary targets and losses.

Semantic labels:
    0 background
    1 sacrum
    2 left_hip
    3 right_hip
    4 fracture

Auxiliary channels produced by precompute_agss_auxiliary.py:
    0 y_frac      : binary fracture
    1 y_anat      : filled anatomy label (0/1/2/3), with fracture voxels assigned to nearest anatomy
    2 y_region    : 0 background, 1 sacrum, 2 hip-union
    3 y_side      : 0 background/non-hip, 1 left, 2 right
    4 y_sacfrac   : binary sacrum-fracture mask
    5 D_core      : connected-component normalized fracture core field
    6 D_surface   : fracture surface field
    7 small_weight: per-voxel weight emphasising small fracture fragments
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt, label as cc_label
except Exception:  # pragma: no cover
    binary_erosion = None
    distance_transform_edt = None
    cc_label = None

AGSS_NUM_AUX_CHANNELS = 8
AGSS_AUX_CHANNEL_NAMES = ("frac", "anat", "region", "side", "sacfrac", "core", "surface", "small_weight")


def _as_label_3d(seg: np.ndarray) -> np.ndarray:
    arr = np.asarray(seg)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D label or 1x3D label, got {arr.shape}")
    return arr.astype(np.int16, copy=False)


def infer_anatomy_for_fracture_voxels(
    label: np.ndarray,
    fracture_label: int = 4,
    anatomy_labels: Tuple[int, int, int] = (1, 2, 3),
) -> np.ndarray:
    label = _as_label_3d(label)
    anat = np.zeros_like(label, dtype=np.uint8)
    for a in anatomy_labels:
        anat[label == int(a)] = int(a)

    frac = label == int(fracture_label)
    if not np.any(frac):
        return anat
    anatomy_mask = np.isin(label, np.asarray(anatomy_labels, dtype=label.dtype))
    if not np.any(anatomy_mask) or distance_transform_edt is None:
        return anat
    _, indices = distance_transform_edt(~anatomy_mask, return_indices=True)
    nearest = label[tuple(indices)].astype(np.uint8, copy=False)
    nearest[~np.isin(nearest, anatomy_labels)] = 0
    anat[frac] = nearest[frac]
    return anat


def make_region_side_targets(
    label: np.ndarray,
    fracture_label: int = 4,
    anatomy_labels: Tuple[int, int, int] = (1, 2, 3),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        anat_full : 0 bg, 1 sacrum, 2 left, 3 right
        region    : 0 bg, 1 sacrum, 2 hip_union
        side      : 0 bg/non-hip, 1 left, 2 right
    """
    anat = infer_anatomy_for_fracture_voxels(label, fracture_label, anatomy_labels).astype(np.uint8)
    region = np.zeros_like(anat, dtype=np.uint8)
    region[anat == int(anatomy_labels[0])] = 1
    region[(anat == int(anatomy_labels[1])) | (anat == int(anatomy_labels[2]))] = 2

    side = np.zeros_like(anat, dtype=np.uint8)
    side[anat == int(anatomy_labels[1])] = 1
    side[anat == int(anatomy_labels[2])] = 2
    return anat, region, side


def make_sacrum_fracture_mask(
    label: np.ndarray,
    fracture_label: int = 4,
    anatomy_labels: Tuple[int, int, int] = (1, 2, 3),
) -> np.ndarray:
    label = _as_label_3d(label)
    frac = label == int(fracture_label)
    anat = infer_anatomy_for_fracture_voxels(label, fracture_label, anatomy_labels)
    return (frac & (anat == int(anatomy_labels[0]))).astype(np.float32)


def make_core_surface_fields(
    label: np.ndarray,
    fracture_label: int = 4,
    surface_sigma: float = 2.0,
    surface_radius: float = 8.0,
    per_component_core: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    label = _as_label_3d(label)
    fracture = label == int(fracture_label)
    core = np.zeros(label.shape, dtype=np.float32)
    surface_field = np.zeros(label.shape, dtype=np.float32)
    if not np.any(fracture) or distance_transform_edt is None:
        return core, surface_field

    if per_component_core and cc_label is not None:
        comp, n = cc_label(fracture)
        for idx in range(1, n + 1):
            m = comp == idx
            if not np.any(m):
                continue
            d = distance_transform_edt(m).astype(np.float32)
            mx = float(d.max())
            if mx > 0:
                d /= mx
            core[m] = d[m]
    else:
        d = distance_transform_edt(fracture).astype(np.float32)
        mx = float(d.max())
        if mx > 0:
            d /= mx
        core[fracture] = d[fracture]

    if binary_erosion is not None:
        eroded = binary_erosion(fracture, iterations=1)
        surface = fracture & (~eroded)
    else:
        surface = fracture

    if np.any(surface):
        ds = distance_transform_edt(~surface).astype(np.float32)
        surface_field = np.exp(-(ds ** 2) / (2.0 * float(surface_sigma) ** 2)).astype(np.float32)
        near = distance_transform_edt(~fracture) < float(surface_radius)
        surface_field *= near.astype(np.float32)
    return core.astype(np.float32), surface_field.astype(np.float32)


def make_small_component_weight(
    label: np.ndarray,
    fracture_label: int = 4,
    s_ref: float = 2000.0,
    gamma: float = 0.5,
    w_max: float = 4.0,
) -> np.ndarray:
    """Per-voxel weight emphasising small fracture fragments.

    weight = 1 + (w_max - 1) * (s_ref / (component_size + s_ref)) ^ gamma
    Non-fracture voxels get weight 1.
    """
    label = _as_label_3d(label)
    fracture = label == int(fracture_label)
    weight = np.ones(label.shape, dtype=np.float32)
    if not np.any(fracture) or cc_label is None:
        return weight

    comp, n = cc_label(fracture)
    for idx in range(1, n + 1):
        m = comp == idx
        size = float(np.count_nonzero(m))
        w = 1.0 + (float(w_max) - 1.0) * (float(s_ref) / (size + float(s_ref))) ** float(gamma)
        weight[m] = float(w)
    return weight.astype(np.float32, copy=False)


def build_agss_auxiliary_from_label(
    label: np.ndarray,
    fracture_label: int = 4,
    anatomy_labels: Tuple[int, int, int] = (1, 2, 3),
    surface_sigma: float = 2.0,
    surface_radius: float = 8.0,
    small_component_s_ref: float = 2000.0,
    small_component_gamma: float = 0.5,
    small_component_w_max: float = 4.0,
) -> np.ndarray:
    label = _as_label_3d(label)
    frac = (label == int(fracture_label)).astype(np.float32)
    anat, region, side = make_region_side_targets(label, fracture_label, anatomy_labels)
    sacfrac = make_sacrum_fracture_mask(label, fracture_label, anatomy_labels)
    core, surface = make_core_surface_fields(label, fracture_label, surface_sigma, surface_radius)
    small_weight = make_small_component_weight(
        label, fracture_label,
        s_ref=float(small_component_s_ref),
        gamma=float(small_component_gamma),
        w_max=float(small_component_w_max),
    )
    aux = np.stack(
        [
            frac.astype(np.float32),
            anat.astype(np.float32),
            region.astype(np.float32),
            side.astype(np.float32),
            sacfrac.astype(np.float32),
            core.astype(np.float32),
            surface.astype(np.float32),
            small_weight.astype(np.float32),
        ],
        axis=0,
    )
    return aux.astype(np.float32, copy=False)


@dataclass
class AGSSLossWeights:
    sem_aux: float = 0.15
    frac: float = 1.0
    region: float = 0.25
    side: float = 0.20
    sacfrac: float = 0.35
    struct: float = 0.20
    prior: float = 0.10
    consistency: float = 0.05
    arconv: float = 0.1    # was 1.0: regularisation must not dominate fracture loss


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _highest_resolution(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _seg_only_target(target):
    if isinstance(target, (list, tuple)):
        return [t[:, 0:1] if torch.is_tensor(t) and t.ndim >= 5 and t.shape[1] > 1 else t for t in target]
    if torch.is_tensor(target) and target.ndim >= 5 and target.shape[1] > 1:
        return target[:, 0:1]
    return target


def _split_agss_target(target, fracture_label: int = 4):
    if isinstance(target, (list, tuple)):
        sems, fracs, anats, regions, sides, sacfracs, structs, small_weights = [], [], [], [], [], [], [], []
        for t in target:
            s, f, a, r, sd, sf, st, sw = _split_agss_target(t, fracture_label)
            sems.append(s); fracs.append(f); anats.append(a); regions.append(r); sides.append(sd); sacfracs.append(sf); structs.append(st); small_weights.append(sw)
        return sems, fracs, anats, regions, sides, sacfracs, structs, small_weights

    t = target
    sem = t[:, 0:1]
    if t.shape[1] >= 9:
        frac = t[:, 1:2]
        anat = t[:, 2:3]
        region = t[:, 3:4]
        side = t[:, 4:5]
        sacfrac = t[:, 5:6]
        struct = t[:, 6:8]
        small_weight = t[:, 8:9]
    elif t.shape[1] >= 8:
        frac = t[:, 1:2]
        anat = t[:, 2:3]
        region = t[:, 3:4]
        side = t[:, 4:5]
        sacfrac = t[:, 5:6]
        struct = t[:, 6:8]
        small_weight = torch.ones_like(frac, dtype=torch.float32)
    else:
        frac = (sem == int(fracture_label)).float()
        anat = sem.clone(); anat[anat == int(fracture_label)] = 0; anat[(anat < 0) | (anat > 3)] = 0
        region = torch.zeros_like(sem)
        region[anat == 1] = 1
        region[(anat == 2) | (anat == 3)] = 2
        side = torch.zeros_like(sem)
        side[anat == 2] = 1
        side[anat == 3] = 2
        sacfrac = ((frac > 0.5) & (anat == 1)).float()
        struct = torch.zeros((sem.shape[0], 2, *sem.shape[2:]), device=sem.device, dtype=torch.float32)
        small_weight = torch.ones_like(frac, dtype=torch.float32)
    return sem, frac, anat, region, side, sacfrac, struct, small_weight


def _resize_target_like(t: torch.Tensor, pred: torch.Tensor, mode: str) -> torch.Tensor:
    if t.shape[2:] == pred.shape[2:]:
        return t
    if mode == "nearest":
        return F.interpolate(t.float(), size=pred.shape[2:], mode="nearest")
    return F.interpolate(t.float(), size=pred.shape[2:], mode="trilinear", align_corners=False)


def _soft_dice_binary_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = _resize_target_like(target.float(), logits, "nearest")
    target = (target > 0.5).float()
    prob = torch.sigmoid(logits)
    axes = (0,) + tuple(range(2, logits.ndim))
    inter = (prob * target).sum(axes)
    denom = prob.sum(axes) + target.sum(axes)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def _bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, voxel_weight: torch.Tensor | None = None) -> torch.Tensor:
    target = _resize_target_like(target.float(), logits, "nearest")
    target = (target > 0.5).float()
    if voxel_weight is not None:
        voxel_weight = _resize_target_like(voxel_weight.float(), logits, "nearest")
        voxel_weight = torch.nan_to_num(voxel_weight, nan=1.0, posinf=4.0, neginf=1.0).clamp(1.0, 4.0)
        bce_all = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce_all * voxel_weight).sum() / voxel_weight.sum().clamp_min(1.0)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = _soft_dice_binary_from_logits(logits, target)
    return bce + dice

def _finite_loss(
    loss: torch.Tensor,
    nan: float = 0.0,
    posinf: float = 10.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    """
    Replace NaN/Inf loss values with finite constants to avoid training crash.
    For normal finite losses, this does nothing.
    """
    return torch.nan_to_num(
        loss,
        nan=float(nan),
        posinf=float(posinf),
        neginf=float(neginf),
    )

def _multiclass_ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_label: int = -1) -> torch.Tensor:
    target = _resize_target_like(target.float(), logits, "nearest")[:, 0].long()
    valid = target != int(ignore_label)
    safe_target = target.clone()
    safe_target[~valid] = 0
    safe_target = safe_target.clamp(0, num_classes - 1)
    ce_all = F.cross_entropy(logits, safe_target, reduction="none")
    ce = (ce_all * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    prob = torch.softmax(logits, dim=1)
    onehot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 4, 1, 2, 3).to(prob.dtype)
    valid_f = valid[:, None].to(prob.dtype)
    prob = prob * valid_f
    onehot = onehot * valid_f
    axes = (0,) + tuple(range(2, logits.ndim))
    inter = (prob * onehot).sum(axes)
    denom = prob.sum(axes) + onehot.sum(axes)
    dice_per_class = (2.0 * inter + 1e-6) / (denom + 1e-6)
    dice = 1.0 - dice_per_class[1:].mean()
    return ce + dice


def _masked_multiclass_ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    target = _resize_target_like(target.float(), logits, "nearest")[:, 0].long()
    mask = (_resize_target_like(mask.float(), logits, "nearest") > 0.5)[:, 0]
    mask_count = mask.float().sum()
    # Skip when masked voxels too few — Dice would be epsilon-dominated.
    if not bool((mask_count > 8).item()):
        return logits.new_tensor(0.0)

    safe_target = target.clamp(0, num_classes - 1)
    ce_all = F.cross_entropy(logits, safe_target, reduction="none")
    ce = (ce_all * mask.float()).sum() / mask_count.clamp_min(1.0)

    prob = torch.softmax(logits, dim=1)
    onehot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 4, 1, 2, 3).to(prob.dtype)
    mask_f = mask[:, None].to(prob.dtype)
    prob = prob * mask_f
    onehot = onehot * mask_f
    axes = (0,) + tuple(range(2, logits.ndim))
    inter = (prob * onehot).sum(axes)
    denom = prob.sum(axes) + onehot.sum(axes)
    # Count-scaled epsilon avoids spurious Dice≈1 when mask is small.
    eps = max(float(mask_count.item()) * 1e-3, 1e-2)
    dice_per_class = (2.0 * inter + eps) / (denom + eps)
    if num_classes > 1:
        dice = 1.0 - dice_per_class[1:].mean()
    else:
        dice = 1.0 - dice_per_class.mean()
    return _finite_loss(ce + dice, posinf=10.0)


def _weighted_deep_supervision_loss(preds, targets, loss_fn) -> torch.Tensor:
    pred_list = _as_list(preds)
    target_list = _as_list(targets)
    if len(target_list) == 1 and len(pred_list) > 1:
        target_list = target_list * len(pred_list)
    n = min(len(pred_list), len(target_list))
    weights = torch.tensor([1.0 / (2 ** i) for i in range(n)], device=pred_list[0].device, dtype=pred_list[0].dtype)
    weights = weights / weights.sum()
    total = pred_list[0].new_tensor(0.0)
    for i in range(n):
        total = total + weights[i] * loss_fn(pred_list[i], target_list[i])
    return total


def assemble_anatomy_probs_from_logits(region_logits: torch.Tensor, side_logits: torch.Tensor) -> torch.Tensor:
    """Assemble 4-class anatomy probabilities: bg, sacrum, left, right."""
    pr = torch.softmax(region_logits, dim=1)
    ps = torch.softmax(side_logits, dim=1)
    hip = pr[:, 2:3]
    side_lr = ps[:, 1:3]
    side_lr = side_lr / side_lr.sum(1, keepdim=True).clamp_min(1e-6)
    probs = torch.cat([pr[:, 0:1], pr[:, 1:2], hip * side_lr[:, 0:1], hip * side_lr[:, 1:2]], dim=1)
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
    return probs


def anatomy_prior_uncertainty_from_logits(region_logits: torch.Tensor, side_logits: torch.Tensor):
    probs = assemble_anatomy_probs_from_logits(region_logits, side_logits)
    prior = probs[:, 1:].sum(1, keepdim=True).clamp(0, 1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(1, keepdim=True)
    entropy = entropy / np.log(max(int(probs.shape[1]), 2))
    return prior, entropy.clamp(0, 1), probs


def fracture_probability_from_semantic(seg_logits: torch.Tensor, fracture_label: int = 4):
    if seg_logits.shape[1] == 1:
        return torch.sigmoid(seg_logits)
    probs = torch.softmax(seg_logits, dim=1)
    return probs[:, int(fracture_label):int(fracture_label)+1]


class AGSSAuxiliaryLoss(torch.nn.Module):
    def __init__(
        self,
        base_loss,
        weights: AGSSLossWeights = AGSSLossWeights(),
        fracture_label: int = 4,
        enable_auxiliary: bool = True,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
        self.fracture_label = int(fracture_label)
        self.enable_auxiliary = bool(enable_auxiliary)

    def forward(self, output, target):
        if not isinstance(output, dict):
            return self.base_loss(output, _seg_only_target(target))

        sem_t, frac_t, anat_t, region_t, side_t, sacfrac_t, struct_t, small_weight_t = _split_agss_target(target, self.fracture_label)
        loss = self.base_loss(output["seg"], sem_t)

        if self.enable_auxiliary:
            if "sem_aux" in output and self.weights.sem_aux > 0:
                loss = loss + self.weights.sem_aux * _weighted_deep_supervision_loss(
                    output["sem_aux"], sem_t, lambda p, y: _multiclass_ce_dice_loss(p, y, 5)
                )
            if "frac" in output and self.weights.frac > 0:
                sw_t = _highest_resolution(small_weight_t)
                loss = loss + self.weights.frac * _weighted_deep_supervision_loss(
                    output["frac"], frac_t, lambda p, y: _bce_dice_loss(p, y, voxel_weight=sw_t)
                )
            if "region" in output and self.weights.region > 0:
                loss = loss + self.weights.region * _weighted_deep_supervision_loss(
                    output["region"], region_t, lambda p, y: _multiclass_ce_dice_loss(p, y, 3)
                )
            if "side" in output and self.weights.side > 0:
                def side_loss(p, y):
                    # only supervise where target belongs to hip union or to left/right side
                    y = _resize_target_like(y.float(), p, "nearest")
                    valid = (y > 0.5).float()
                    return _masked_multiclass_ce_dice_loss(p, y, valid, 3)
                loss = loss + self.weights.side * _weighted_deep_supervision_loss(
                    output["side"], side_t, side_loss
                )
            if "sacfrac" in output and self.weights.sacfrac > 0:
                loss = loss + self.weights.sacfrac * _weighted_deep_supervision_loss(
                    output["sacfrac"], sacfrac_t, _bce_dice_loss
                )
            if "struct" in output and self.weights.struct > 0:
                def struct_loss(p, y):
                    y = _resize_target_like(y.float(), p, "trilinear").clamp(0, 1)
                    return F.smooth_l1_loss(torch.sigmoid(p), y)
                loss = loss + self.weights.struct * _weighted_deep_supervision_loss(output["struct"], struct_t, struct_loss)

            if self.weights.prior > 0 or self.weights.consistency > 0:
                frac_h = _highest_resolution(output.get("frac"))
                p_frac = torch.sigmoid(frac_h)
                if "region" in output and "side" in output:
                    region_h = _highest_resolution(output["region"])
                    side_h = _highest_resolution(output["side"])
                    p_anat_fg, _, _ = anatomy_prior_uncertainty_from_logits(region_h, side_h)
                    if p_anat_fg.shape[2:] != p_frac.shape[2:]:
                        p_anat_fg = F.interpolate(p_anat_fg, size=p_frac.shape[2:], mode="trilinear", align_corners=False)
                    if self.weights.prior > 0:
                        loss = loss + self.weights.prior * (p_frac * (1.0 - p_anat_fg).clamp_min(0)).mean()
                if self.weights.consistency > 0 and "sem_aux" in output:
                    sem_aux_h = _highest_resolution(output["sem_aux"])
                    p_sem_frac = fracture_probability_from_semantic(sem_aux_h, self.fracture_label)
                    if p_sem_frac.shape[2:] != p_frac.shape[2:]:
                        p_sem_frac = F.interpolate(p_sem_frac, size=p_frac.shape[2:], mode="trilinear", align_corners=False)
                    loss = loss + self.weights.consistency * F.l1_loss(p_sem_frac, p_frac.detach())

        arconv_reg = output.get("arconv_reg", None)
        if torch.is_tensor(arconv_reg) and self.weights.arconv > 0:
            loss = loss + self.weights.arconv * arconv_reg
        return loss
