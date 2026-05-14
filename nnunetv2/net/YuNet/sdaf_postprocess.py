"""
Affinity/contact guided instance recovery for SDAF-AGDSNet outputs.

For post-processing, run the network with return_dict=True or call
network.enable_sdaf_inference_outputs() before prediction to obtain struct/affinity.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    binary_dilation = None
    distance_transform_edt = None
    cc_label = None

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    peak_local_max = None
    watershed = None


def _to_numpy(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_sdaf_probabilities(output, fracture_start_channel: int = 4) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert network output dict/tensor to numpy probabilities.

    Returns:
        fracture_prob: (D,H,W)
        struct_prob:   (3,D,H,W) or None
        affinity_prob: (K,D,H,W) or None
    """
    if isinstance(output, dict):
        seg = output['seg']
        struct = output.get('struct')
        affinity = output.get('affinity')
    else:
        seg = output
        struct = None
        affinity = None
    if isinstance(seg, (list, tuple)):
        seg = seg[0]
    if struct is not None and isinstance(struct, (list, tuple)):
        struct = struct[0]
    if affinity is not None and isinstance(affinity, (list, tuple)):
        affinity = affinity[0]

    if torch is not None and torch.is_tensor(seg):
        if seg.shape[0] == 1:
            seg = seg[0]
        if seg.shape[0] == 1:
            frac = torch.sigmoid(seg[0]).detach().cpu().numpy()
        else:
            probs = torch.softmax(seg, dim=0)
            if probs.shape[0] > int(fracture_start_channel):
                frac = probs[int(fracture_start_channel):].sum(0).detach().cpu().numpy()
            else:
                frac = probs[1:].sum(0).detach().cpu().numpy()
    else:
        seg_np = _to_numpy(seg)
        if seg_np.ndim == 5:
            seg_np = seg_np[0]
        # Assume already probabilities if not torch logits.
        if seg_np.shape[0] == 1:
            frac = 1.0 / (1.0 + np.exp(-seg_np[0]))
        else:
            e = np.exp(seg_np - seg_np.max(axis=0, keepdims=True))
            probs = e / e.sum(axis=0, keepdims=True)
            frac = probs[int(fracture_start_channel):].sum(0) if probs.shape[0] > int(fracture_start_channel) else probs[1:].sum(0)

    struct_prob = None
    if struct is not None:
        s = _to_numpy(struct)
        if s.ndim == 5:
            s = s[0]
        struct_prob = 1.0 / (1.0 + np.exp(-s))

    affinity_prob = None
    if affinity is not None:
        a = _to_numpy(affinity)
        if a.ndim == 5:
            a = a[0]
        affinity_prob = 1.0 / (1.0 + np.exp(-a))

    return frac.astype(np.float32), struct_prob, affinity_prob


def _remove_small_components(instances: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 0:
        return instances.astype(np.int32, copy=False)
    out = instances.copy().astype(np.int32)
    ids, counts = np.unique(out, return_counts=True)
    for i, c in zip(ids, counts):
        if i != 0 and int(c) < int(min_voxels):
            out[out == i] = 0
    new = np.zeros_like(out, dtype=np.int32)
    next_id = 1
    for i in np.unique(out):
        if i == 0:
            continue
        new[out == i] = next_id
        next_id += 1
    return new


def assign_barrier_to_nearest_component(fracture_mask: np.ndarray, components: np.ndarray) -> np.ndarray:
    if not SCIPY_AVAILABLE or components.max() == 0:
        return components.astype(np.int32)
    out = components.copy().astype(np.int32)
    missing = fracture_mask & (out == 0)
    if not missing.any():
        return out
    _, indices = distance_transform_edt(out == 0, return_indices=True)
    nearest_labels = out[tuple(indices)]
    out[missing] = nearest_labels[missing]
    return out.astype(np.int32)


def connected_component_contact_split(
    fracture_prob: np.ndarray,
    struct_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    contact_threshold: float = 0.45,
    contact_dilation_iter: int = 1,
    min_voxels: int = 20,
) -> np.ndarray:
    fracture_prob = np.asarray(fracture_prob, dtype=np.float32)
    fracture_mask = fracture_prob > float(threshold)
    if not fracture_mask.any():
        return np.zeros_like(fracture_prob, dtype=np.int32)
    if not SCIPY_AVAILABLE:
        return fracture_mask.astype(np.int32)

    barrier = np.zeros_like(fracture_mask, dtype=bool)
    if struct_prob is not None:
        struct_prob = np.asarray(struct_prob, dtype=np.float32)
        if struct_prob.ndim == 4 and struct_prob.shape[0] >= 3:
            barrier = (struct_prob[2] > float(contact_threshold)) & fracture_mask
            if contact_dilation_iter > 0:
                barrier = binary_dilation(barrier, iterations=int(contact_dilation_iter)) & fracture_mask

    seed_region = fracture_mask & (~barrier)
    components, _ = cc_label(seed_region.astype(np.uint8))
    components = assign_barrier_to_nearest_component(fracture_mask, components)
    return _remove_small_components(components, min_voxels=min_voxels)


def watershed_core_contact_split(
    fracture_prob: np.ndarray,
    struct_prob: np.ndarray,
    threshold: float = 0.5,
    core_threshold: float = 0.45,
    contact_weight: float = 1.0,
    min_voxels: int = 20,
) -> np.ndarray:
    if not (SCIPY_AVAILABLE and SKIMAGE_AVAILABLE):
        return connected_component_contact_split(fracture_prob, struct_prob, threshold=threshold, min_voxels=min_voxels)
    fracture_prob = np.asarray(fracture_prob, dtype=np.float32)
    struct_prob = np.asarray(struct_prob, dtype=np.float32)
    fracture_mask = fracture_prob > float(threshold)
    if not fracture_mask.any():
        return np.zeros_like(fracture_prob, dtype=np.int32)

    core = struct_prob[0]
    contact = struct_prob[2] if struct_prob.shape[0] >= 3 else np.zeros_like(core)
    coordinates = peak_local_max(core, labels=fracture_mask.astype(np.uint8), min_distance=2, threshold_abs=core_threshold)
    markers = np.zeros_like(fracture_prob, dtype=np.int32)
    for idx, (z, y, x) in enumerate(coordinates, start=1):
        markers[z, y, x] = idx
    if markers.max() == 0:
        seed_region = fracture_mask & (core > core_threshold)
        markers, _ = cc_label(seed_region.astype(np.uint8))
    if markers.max() == 0:
        return connected_component_contact_split(fracture_prob, struct_prob, threshold=threshold, min_voxels=min_voxels)

    elevation = -core + float(contact_weight) * contact
    instances = watershed(elevation, markers=markers, mask=fracture_mask)
    return _remove_small_components(instances.astype(np.int32), min_voxels=min_voxels)


def recover_instances(
    fracture_prob: np.ndarray,
    struct_prob: Optional[np.ndarray] = None,
    affinity_prob: Optional[np.ndarray] = None,
    method: str = "contact_cc",
    **kwargs,
) -> np.ndarray:
    if method == "watershed":
        if struct_prob is None:
            raise ValueError("struct_prob is required for watershed instance recovery")
        return watershed_core_contact_split(fracture_prob, struct_prob, **kwargs)
    if method == "contact_cc":
        return connected_component_contact_split(fracture_prob, struct_prob, **kwargs)
    raise ValueError(f"Unknown SDAF postprocess method: {method}")
