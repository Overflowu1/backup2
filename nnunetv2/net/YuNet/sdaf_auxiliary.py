"""
SDAF auxiliary target generation and loss, compatible with older nnU-Net v2.

V2 changes
----------
1. Binary labels with --fracture-start-label 1 are converted to connected components
   before generating core distance and affinity labels.
2. Multi-scale auxiliary supervision is supported when deep supervision transforms
   downsample all appended target channels.
3. ARConv regularization can be read from output['arconv_reg'], avoiding Trainer-side
   reads of mutable module state.
4. Contact supervision is safe for binary/non-instance datasets: D_contact may be all
   zeros, and the loss automatically avoids contact-weight explosions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import (
        binary_erosion,
        distance_transform_edt,
        label as cc_label,
        maximum_filter,
        minimum_filter,
    )
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    binary_erosion = distance_transform_edt = cc_label = maximum_filter = minimum_filter = None


DEFAULT_AFFINITY_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def _as_label_volume(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    if label.ndim == 4:
        label = label[0]
    if label.ndim != 3:
        raise ValueError(f"Expected 3D or Cx3D label, got shape {label.shape}")
    return label.astype(np.int32, copy=False)


def _connected_components_or_binary(fg: np.ndarray) -> np.ndarray:
    fg = fg.astype(bool)
    if not fg.any():
        return np.zeros_like(fg, dtype=np.int32)
    if SCIPY_AVAILABLE:
        inst, _ = cc_label(fg.astype(np.uint8))
        return inst.astype(np.int32, copy=False)
    return fg.astype(np.int32)


def _instance_label_from_target(label: np.ndarray, fracture_start_label: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns instance label, foreground mask, has_real_instance_ids.

    Binary fracture datasets commonly use 0=background, 1=fracture. If the user sets
    fracture_start_label=1 and there is only one foreground value, we treat it as binary
    and derive pseudo instances by connected components instead of assuming that all
    foreground voxels are one real fragment instance.
    """
    label = _as_label_volume(label)
    fs = int(fracture_start_label)
    ids = [int(i) for i in np.unique(label) if int(i) >= fs]

    # Binary case: 0/1 label map. Use connected components as pseudo instances.
    if fs <= 1 and set(ids).issubset({1}):
        fg = label > 0
        return _connected_components_or_binary(fg), fg, False

    if len(ids) > 0:
        inst = np.zeros_like(label, dtype=np.int32)
        for i in ids:
            inst[label == i] = i
        return inst, inst > 0, True

    # Wrong fracture_start_label or no explicit instances: fallback to all foreground.
    fg = label > 0
    return _connected_components_or_binary(fg), fg, False


def has_real_instance_ids(label: np.ndarray, fracture_start_label: int = 4) -> bool:
    _, _, real = _instance_label_from_target(label, fracture_start_label)
    return bool(real)


def _surface_from_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if SCIPY_AVAILABLE:
        eroded = binary_erosion(mask, iterations=1, border_value=0)
        return mask & (~eroded)
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    c = padded[1:-1, 1:-1, 1:-1]
    neigh = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return (c > 0) & (neigh == 0)


def _gaussian_distance_to_binary(binary: np.ndarray, sigma: float, limit_to: Optional[np.ndarray] = None) -> np.ndarray:
    binary = binary.astype(bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.float32)
    if not SCIPY_AVAILABLE:
        return binary.astype(np.float32)
    dist = distance_transform_edt(~binary)
    field = np.exp(-(dist ** 2) / (2.0 * float(sigma) ** 2)).astype(np.float32)
    if limit_to is not None and limit_to.any():
        near = distance_transform_edt(~limit_to.astype(bool)) < max(4, int(round(4 * float(sigma))))
        field *= near.astype(np.float32)
    return field.astype(np.float32)


def make_core_distance_field(label: np.ndarray, fracture_start_label: int = 4) -> np.ndarray:
    inst, fg, _ = _instance_label_from_target(label, fracture_start_label)
    out = np.zeros_like(inst, dtype=np.float32)
    ids = [int(i) for i in np.unique(inst) if int(i) > 0]
    if len(ids) == 0:
        return out
    if not SCIPY_AVAILABLE:
        out[fg] = 1.0
        return out
    for fid in ids:
        m = inst == fid
        d = distance_transform_edt(m).astype(np.float32)
        if d.max() > 0:
            d /= (d.max() + 1e-6)
        d = np.clip(d * (1.0 + 0.03 * np.log1p(float(m.sum()))), 0, 1)
        out[m] = d[m]
    return out.astype(np.float32)


def make_surface_field(label: np.ndarray, fracture_start_label: int = 4, sigma: float = 2.0) -> np.ndarray:
    _, fg, _ = _instance_label_from_target(label, fracture_start_label)
    surface = _surface_from_mask(fg)
    return _gaussian_distance_to_binary(surface, sigma=sigma, limit_to=fg)


def make_contact_field(
    label: np.ndarray,
    fracture_start_label: int = 4,
    window: int = 7,
    sigma: float = 2.0,
) -> np.ndarray:
    inst, fg, real_instances = _instance_label_from_target(label, fracture_start_label)
    ids = [int(i) for i in np.unique(inst) if int(i) > 0]
    # Contact surface is only meaningful with real fragment instance ids. For binary labels
    # or pseudo CC instances, returning zero is intentional and is reported by precompute.
    if (not real_instances) or len(ids) <= 1 or (not fg.any()) or (not SCIPY_AVAILABLE):
        return np.zeros_like(inst, dtype=np.float32)

    large = max(ids) + 100000
    min_src = inst.copy()
    min_src[min_src <= 0] = large
    local_max = maximum_filter(inst, size=int(window))
    local_min = minimum_filter(min_src, size=int(window))
    contact = fg & (
        ((local_max != inst) & (local_max > 0))
        | ((local_min != inst) & (local_min < large))
    )
    return _gaussian_distance_to_binary(contact, sigma=sigma, limit_to=fg)


def make_structural_fields(
    label: np.ndarray,
    fracture_start_label: int = 4,
    surface_sigma: float = 2.0,
    contact_window: int = 7,
    contact_sigma: float = 2.0,
) -> np.ndarray:
    label = _as_label_volume(label)
    return np.stack(
        [
            make_core_distance_field(label, fracture_start_label),
            make_surface_field(label, fracture_start_label, surface_sigma),
            make_contact_field(label, fracture_start_label, contact_window, contact_sigma),
        ],
        axis=0,
    ).astype(np.float32)


def make_affinity_labels(
    label: np.ndarray,
    offsets: Sequence[Tuple[int, int, int]] = DEFAULT_AFFINITY_OFFSETS,
    fracture_start_label: int = 4,
) -> np.ndarray:
    inst, _, _ = _instance_label_from_target(label, fracture_start_label)
    labels = inst.astype(np.int32, copy=False)
    affs: List[np.ndarray] = []
    for dz, dy, dx in offsets:
        shifted = np.zeros_like(labels, dtype=np.int32)
        z_src = slice(max(0, -dz), labels.shape[0] - max(0, dz))
        y_src = slice(max(0, -dy), labels.shape[1] - max(0, dy))
        x_src = slice(max(0, -dx), labels.shape[2] - max(0, dx))
        z_dst = slice(max(0, dz), labels.shape[0] - max(0, -dz))
        y_dst = slice(max(0, dy), labels.shape[1] - max(0, -dy))
        x_dst = slice(max(0, dx), labels.shape[2] - max(0, -dx))
        shifted[z_dst, y_dst, x_dst] = labels[z_src, y_src, x_src]
        valid = (labels > 0) | (shifted > 0)
        same = (labels == shifted) & (labels > 0)
        aff = np.zeros_like(labels, dtype=np.float32)
        aff[valid] = same[valid].astype(np.float32)
        affs.append(aff)
    return np.stack(affs, axis=0).astype(np.float32)


def foreground_probability_from_logits(seg_logits: torch.Tensor, fracture_start_channel: int = 4) -> torch.Tensor:
    if isinstance(seg_logits, (list, tuple)):
        seg_logits = seg_logits[0]
    if seg_logits.shape[1] == 1:
        return torch.sigmoid(seg_logits)
    probs = torch.softmax(seg_logits, dim=1)
    if probs.shape[1] > int(fracture_start_channel):
        return probs[:, int(fracture_start_channel):].sum(1, keepdim=True).clamp(0, 1)
    return probs[:, 1:].sum(1, keepdim=True).clamp(0, 1)


@dataclass
class SDAFLossWeights:
    struct: float = 0.30
    affinity: float = 0.20
    consistency: float = 0.10
    contact_weight: float = 2.00
    arconv: float = 1.00


def _extract_first_channel_target(target):
    if isinstance(target, (list, tuple)):
        return [t[:, 0:1] if torch.is_tensor(t) and t.ndim >= 5 and t.shape[1] > 1 else t for t in target]
    if torch.is_tensor(target) and target.ndim >= 5 and target.shape[1] > 1:
        return target[:, 0:1]
    return target


def _split_one_target(t: torch.Tensor, num_struct_channels: int, num_affinity_channels: int):
    seg = t[:, 0:1] if t.ndim >= 5 and t.shape[1] > 1 else t
    struct = None
    aff = None
    if torch.is_tensor(t) and t.ndim >= 5:
        c = int(t.shape[1])
        s0 = 1
        s1 = s0 + int(num_struct_channels)
        a1 = s1 + int(num_affinity_channels)
        if c >= s1:
            struct = t[:, s0:s1].float().clamp(0, 1)
        if c >= a1:
            aff = t[:, s1:a1].float().clamp(0, 1)
    return seg, struct, aff


def _split_target_pack(
    target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, Any]],
    num_struct_channels: int,
    num_affinity_channels: int,
):
    if isinstance(target, dict):
        return target.get("seg"), target.get("struct"), target.get("affinity")
    if isinstance(target, (list, tuple)):
        segs, structs, affs = [], [], []
        any_struct, any_aff = False, False
        for t in target:
            seg, struct, aff = _split_one_target(t, num_struct_channels, num_affinity_channels)
            segs.append(seg)
            structs.append(struct)
            affs.append(aff)
            any_struct = any_struct or struct is not None
            any_aff = any_aff or aff is not None
        return segs, (structs if any_struct else None), (affs if any_aff else None)
    return _split_one_target(target, num_struct_channels, num_affinity_channels)


def _online_make_aux_from_target(
    seg_target,
    fracture_start_label: int,
    affinity_offsets: Sequence[Tuple[int, int, int]],
    surface_sigma: float,
    contact_window: int,
    contact_sigma: float,
):
    high = seg_target[0] if isinstance(seg_target, (list, tuple)) else seg_target
    if not torch.is_tensor(high):
        return None, None
    labs = high.detach().cpu().numpy()
    struct_list, aff_list = [], []
    for b in range(labs.shape[0]):
        lab = labs[b, 0]
        struct_list.append(
            make_structural_fields(
                lab,
                fracture_start_label=fracture_start_label,
                surface_sigma=surface_sigma,
                contact_window=contact_window,
                contact_sigma=contact_sigma,
            )
        )
        aff_list.append(make_affinity_labels(lab, offsets=affinity_offsets, fracture_start_label=fracture_start_label))
    struct = torch.from_numpy(np.stack(struct_list, axis=0)).to(device=high.device, dtype=torch.float32)
    aff = torch.from_numpy(np.stack(aff_list, axis=0)).to(device=high.device, dtype=torch.float32)
    return struct, aff


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _deep_supervision_weights(n: int, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    vals = torch.tensor([1.0 / (2 ** i) for i in range(n)], device=device, dtype=dtype)
    vals = vals / vals.sum().clamp_min(1e-6)
    return [v for v in vals]


class SDAFAuxiliaryLoss:
    def __init__(
        self,
        base_loss,
        weights: SDAFLossWeights = SDAFLossWeights(),
        fracture_start_label: int = 4,
        affinity_offsets: Sequence[Tuple[int, int, int]] = DEFAULT_AFFINITY_OFFSETS,
        surface_sigma: float = 2.0,
        contact_window: int = 7,
        contact_sigma: float = 2.0,
        enable_auxiliary: bool = True,
        online_fallback: bool = False,
    ):
        self.base_loss = base_loss
        self.weights = weights
        self.fracture_start_label = int(fracture_start_label)
        self.affinity_offsets = tuple(tuple(int(i) for i in o) for o in affinity_offsets)
        self.surface_sigma = float(surface_sigma)
        self.contact_window = int(contact_window)
        self.contact_sigma = float(contact_sigma)
        self.enable_auxiliary = bool(enable_auxiliary)
        self.online_fallback = bool(online_fallback)
        self.num_struct_channels = 3
        self.num_affinity_channels = len(self.affinity_offsets)

    def _struct_loss_one(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = gt.to(pred.device, dtype=pred.dtype).clamp(0, 1)
        if gt.shape[2:] != pred.shape[2:]:
            gt = F.interpolate(gt, size=pred.shape[2:], mode="trilinear", align_corners=False)
        sp = torch.sigmoid(pred)
        contact_gt_nonempty = bool((gt[:, 2:3].detach().sum() > 0).item())
        contact_w = self.weights.contact_weight if contact_gt_nonempty else 1.0
        return (
            F.smooth_l1_loss(sp[:, 0:1], gt[:, 0:1])
            + F.smooth_l1_loss(sp[:, 1:2], gt[:, 1:2])
            + contact_w * F.smooth_l1_loss(sp[:, 2:3], gt[:, 2:3])
        )

    def _affinity_loss_one(self, pred: torch.Tensor, gt: torch.Tensor, struct_gt: Optional[torch.Tensor]) -> torch.Tensor:
        gt = gt.to(pred.device, dtype=pred.dtype).clamp(0, 1)
        if gt.shape[2:] != pred.shape[2:]:
            gt = F.interpolate(gt, size=pred.shape[2:], mode="nearest")
        weight = None
        if struct_gt is not None:
            cg = struct_gt[:, 2:3].to(pred.device, dtype=pred.dtype).clamp(0, 1)
            if cg.shape[2:] != pred.shape[2:]:
                cg = F.interpolate(cg, size=pred.shape[2:], mode="trilinear", align_corners=False)
            # Use GT contact field as the affinity emphasis. It is stable and automatically
            # becomes neutral for binary datasets where contact is all zero.
            weight = 1.0 + 3.0 * cg
        return F.binary_cross_entropy_with_logits(pred, gt, weight=weight)

    def _consistency_loss_one(self, seg_logits: torch.Tensor, struct_pred: torch.Tensor, aff_pred: torch.Tensor) -> torch.Tensor:
        p_frac = foreground_probability_from_logits(seg_logits, self.fracture_start_label)
        if p_frac.shape[2:] != struct_pred.shape[2:]:
            p_frac = F.interpolate(p_frac, size=struct_pred.shape[2:], mode="trilinear", align_corners=False)
        sp = torch.sigmoid(struct_pred)
        union = torch.amax(sp, dim=1, keepdim=True)
        aff = torch.sigmoid(aff_pred)
        return (
            F.l1_loss(p_frac, union)
            + 0.5 * (sp[:, 2:3].detach() * aff).mean()
            + 0.5 * (sp[:, 0:1].detach() * (1.0 - aff)).mean()
        )

    def __call__(self, output, target):
        seg_target, struct_gt, aff_gt = _split_target_pack(
            target,
            num_struct_channels=self.num_struct_channels,
            num_affinity_channels=self.num_affinity_channels,
        )

        if not isinstance(output, dict):
            return self.base_loss(output, seg_target)

        seg_output = output["seg"]
        loss = self.base_loss(seg_output, seg_target)

        arconv_reg = output.get("arconv_reg", None)
        if torch.is_tensor(arconv_reg) and self.weights.arconv > 0:
            loss = loss + self.weights.arconv * arconv_reg.to(device=loss.device, dtype=loss.dtype)

        if not self.enable_auxiliary:
            return loss

        struct_pred = output.get("struct")
        aff_pred = output.get("affinity")

        if (struct_gt is None or aff_gt is None) and self.online_fallback:
            online_struct, online_aff = _online_make_aux_from_target(
                seg_target,
                fracture_start_label=self.fracture_start_label,
                affinity_offsets=self.affinity_offsets,
                surface_sigma=self.surface_sigma,
                contact_window=self.contact_window,
                contact_sigma=self.contact_sigma,
            )
            if struct_gt is None:
                struct_gt = online_struct
            if aff_gt is None:
                aff_gt = online_aff

        struct_pred_list = _as_list(struct_pred)
        aff_pred_list = _as_list(aff_pred)
        struct_gt_list = _as_list(struct_gt)
        aff_gt_list = _as_list(aff_gt)
        seg_output_list = _as_list(seg_output)

        if struct_pred_list is not None and struct_gt_list is not None:
            n = min(len(struct_pred_list), len(struct_gt_list))
            if n > 0:
                weights = _deep_supervision_weights(n, struct_pred_list[0].device, struct_pred_list[0].dtype)
                l_struct = struct_pred_list[0].new_tensor(0.0)
                for i in range(n):
                    if struct_gt_list[i] is not None:
                        l_struct = l_struct + weights[i] * self._struct_loss_one(struct_pred_list[i], struct_gt_list[i])
                loss = loss + self.weights.struct * l_struct

        if aff_pred_list is not None and aff_gt_list is not None:
            n = min(len(aff_pred_list), len(aff_gt_list))
            if n > 0:
                weights = _deep_supervision_weights(n, aff_pred_list[0].device, aff_pred_list[0].dtype)
                l_aff = aff_pred_list[0].new_tensor(0.0)
                for i in range(n):
                    sgt = struct_gt_list[i] if (struct_gt_list is not None and i < len(struct_gt_list)) else None
                    if aff_gt_list[i] is not None:
                        l_aff = l_aff + weights[i] * self._affinity_loss_one(aff_pred_list[i], aff_gt_list[i], sgt)
                loss = loss + self.weights.affinity * l_aff

        if struct_pred_list is not None and aff_pred_list is not None and self.weights.consistency > 0:
            n = min(len(struct_pred_list), len(aff_pred_list), len(seg_output_list))
            if n > 0:
                weights = _deep_supervision_weights(n, struct_pred_list[0].device, struct_pred_list[0].dtype)
                l_cons = struct_pred_list[0].new_tensor(0.0)
                for i in range(n):
                    l_cons = l_cons + weights[i] * self._consistency_loss_one(seg_output_list[i], struct_pred_list[i], aff_pred_list[i])
                loss = loss + self.weights.consistency * l_cons

        return loss
