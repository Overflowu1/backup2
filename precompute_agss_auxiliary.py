#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute AGSS v3 auxiliary targets.

For labels:
    0 background, 1 sacrum, 2 left_hip, 3 right_hip, 4 fracture

Creates:
    <preprocessed-folder>/agss_aux/<case>.npy or .npz
with aux shape (8, D, H, W):
    0 y_frac
    1 y_anat
    2 y_region
    3 y_side
    4 y_sacfrac
    5 D_core
    6 D_surface
    7 small_weight

Also infers the most likely left-right axis from the centroid separation between
label 2 and label 3 across the dataset. This is used to optionally disable
left-right mirroring during training.

Coordinate maps are not stored here; they are generated on-the-fly in the hybrid
data loader to avoid storage overhead.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from nnunetv2.net.YuNet.agss_auxiliary import build_agss_auxiliary_from_label, AGSS_AUX_CHANNEL_NAMES
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from nnunetv2.net.YuNet.agss_auxiliary import build_agss_auxiliary_from_label, AGSS_AUX_CHANNEL_NAMES


def _load_seg(preprocessed_folder: Path, case_id: str) -> np.ndarray:
    npy = preprocessed_folder / f"{case_id}_seg.npy"
    npz = preprocessed_folder / f"{case_id}.npz"
    if npy.is_file():
        seg = np.load(npy, mmap_mode=None)
    elif npz.is_file():
        seg = np.load(npz)["seg"]
    else:
        raise FileNotFoundError(f"Cannot find segmentation for {case_id}: {npy} or {npz}")
    if seg.ndim == 4:
        seg = seg[0]
    return seg.astype(np.int16, copy=False)


def _centroid(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def _process_one(args):
    preprocessed_folder, out_folder, case_id, fracture_label, anatomy_labels, surface_sigma, surface_radius, small_component_s_ref, small_component_gamma, small_component_w_max, save_npy, overwrite = args
    preprocessed_folder = Path(preprocessed_folder)
    out_folder = Path(out_folder)
    out_npy = out_folder / f"{case_id}.npy"
    out_npz = out_folder / f"{case_id}.npz"
    if not overwrite and (out_npy.is_file() or out_npz.is_file()):
        seg = _load_seg(preprocessed_folder, case_id)
        c_left = _centroid(seg == int(anatomy_labels[1]))
        c_right = _centroid(seg == int(anatomy_labels[2]))
        return case_id, "skipped", {
            "shape": list(seg.shape),
            "fracture_voxels": int((seg == int(fracture_label)).sum()),
            "left_centroid": c_left.tolist() if c_left is not None else None,
            "right_centroid": c_right.tolist() if c_right is not None else None,
        }

    seg = _load_seg(preprocessed_folder, case_id)
    aux = build_agss_auxiliary_from_label(
        seg,
        fracture_label=int(fracture_label),
        anatomy_labels=tuple(int(i) for i in anatomy_labels),
        surface_sigma=float(surface_sigma),
        surface_radius=float(surface_radius),
        small_component_s_ref=float(small_component_s_ref),
        small_component_gamma=float(small_component_gamma),
        small_component_w_max=float(small_component_w_max),
    )
    out_folder.mkdir(parents=True, exist_ok=True)
    if save_npy:
        np.save(out_npy, aux.astype(np.float16, copy=False))
    np.savez_compressed(out_npz, aux=aux.astype(np.float16, copy=False), channel_names=np.asarray(AGSS_AUX_CHANNEL_NAMES))

    c_left = _centroid(seg == int(anatomy_labels[1]))
    c_right = _centroid(seg == int(anatomy_labels[2]))
    stats = {
        "shape": list(seg.shape),
        "fracture_voxels": int((seg == int(fracture_label)).sum()),
        "sacrum_voxels": int((seg == int(anatomy_labels[0])).sum()),
        "left_voxels": int((seg == int(anatomy_labels[1])).sum()),
        "right_voxels": int((seg == int(anatomy_labels[2])).sum()),
        "region_sacrum_voxels": int((aux[2] == 1).sum()),
        "region_hip_voxels": int((aux[2] == 2).sum()),
        "side_left_voxels": int((aux[3] == 1).sum()),
        "side_right_voxels": int((aux[3] == 2).sum()),
        "sacfrac_voxels": int((aux[4] > 0.5).sum()),
        "core_nonzero": int((aux[5] > 0).sum()),
        "surface_nonzero": int((aux[6] > 0.05).sum()),
        "small_weight_max": float(aux[7].max()) if aux.shape[0] > 7 else 1.0,
        "small_weight_mean_on_fracture": (float(aux[7][seg == int(fracture_label)].mean()) if aux.shape[0] > 7 and int((seg == int(fracture_label)).sum()) > 0 else 1.0),
        "left_centroid": c_left.tolist() if c_left is not None else None,
        "right_centroid": c_right.tolist() if c_right is not None else None,
        "centroid_abs_delta": (np.abs(c_left - c_right).tolist() if c_left is not None and c_right is not None else None),
    }
    return case_id, "done", stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-folder", required=True, type=Path)
    parser.add_argument("--out-folder", default=None, type=Path)
    parser.add_argument("--fracture-label", default=4, type=int)
    parser.add_argument("--anatomy-labels", default="1,2,3", type=str, help="comma-separated sacrum,left,right labels")
    parser.add_argument("--surface-sigma", default=2.0, type=float)
    parser.add_argument("--surface-radius", default=8.0, type=float)
    parser.add_argument("--small-component-s-ref", default=2000.0, type=float, help="reference component size for small-fragment weighting")
    parser.add_argument("--small-component-gamma", default=0.5, type=float, help="exponent for small-fragment weighting")
    parser.add_argument("--small-component-w-max", default=4.0, type=float, help="maximum small-fragment voxel weight")
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    preprocessed_folder = args.preprocessed_folder
    out_folder = args.out_folder if args.out_folder is not None else preprocessed_folder / "agss_aux"
    anatomy_labels = tuple(int(i) for i in args.anatomy_labels.split(","))
    if len(anatomy_labels) != 3:
        raise ValueError("--anatomy-labels must contain exactly three labels, e.g. 1,2,3")

    case_ids = sorted([p.name[:-4] for p in preprocessed_folder.glob("*.npz") if "segFromPrevStage" not in p.name])
    if len(case_ids) == 0:
        raise RuntimeError(f"No .npz cases found in {preprocessed_folder}")

    jobs = [
        (str(preprocessed_folder), str(out_folder), cid, args.fracture_label, anatomy_labels,
         args.surface_sigma, args.surface_radius, args.small_component_s_ref, args.small_component_gamma,
         args.small_component_w_max, args.save_npy, args.overwrite)
        for cid in case_ids
    ]

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = pool.map(_process_one, jobs)
    else:
        results = [_process_one(j) for j in jobs]

    deltas = []
    report: Dict[str, object] = {
        "preprocessed_folder": str(preprocessed_folder),
        "out_folder": str(out_folder),
        "num_cases": len(case_ids),
        "fracture_label": int(args.fracture_label),
        "anatomy_labels": list(anatomy_labels),
        "surface_sigma": float(args.surface_sigma),
        "surface_radius": float(args.surface_radius),
        "small_component_s_ref": float(args.small_component_s_ref),
        "small_component_gamma": float(args.small_component_gamma),
        "small_component_w_max": float(args.small_component_w_max),
        "channel_names": list(AGSS_AUX_CHANNEL_NAMES),
        "cases_with_fracture": 0,
        "cases_without_fracture": 0,
        "case_stats": {},
    }

    for cid, status, stats in results:
        report["case_stats"][cid] = stats
        if stats.get("fracture_voxels", 0) > 0:
            report["cases_with_fracture"] += 1
        else:
            report["cases_without_fracture"] += 1
        if stats.get("centroid_abs_delta", None) is not None:
            deltas.append(np.asarray(stats["centroid_abs_delta"], dtype=np.float64))

    inferred_lr_axis = None
    mean_abs_delta = None
    if len(deltas) > 0:
        mean_abs_delta = np.stack(deltas, axis=0).mean(axis=0)
        inferred_lr_axis = int(np.argmax(mean_abs_delta))
        report["mean_left_right_centroid_abs_delta"] = mean_abs_delta.tolist()
        report["inferred_lr_axis"] = inferred_lr_axis

    out_folder.mkdir(parents=True, exist_ok=True)
    with open(out_folder / "agss_aux_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote AGSS aux to: {out_folder}")
    print(f"Cases: {len(case_ids)}, with fracture: {report['cases_with_fracture']}, without fracture: {report['cases_without_fracture']}")
    if inferred_lr_axis is not None:
        print(f"Inferred left-right axis: {inferred_lr_axis}, mean abs centroid delta: {mean_abs_delta}")
    print(f"Report: {out_folder / 'agss_aux_report.json'}")


if __name__ == "__main__":
    main()
