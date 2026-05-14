#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate_agss_full_volume_v31.py

Evaluate full-volume 5-class pelvis+fracture predictions against GT for AGSS v3.1 hybrid.
Predictions are standard fused 5-class labels, so the evaluation is compatible
with both flat and hierarchical training as long as the saved predictions are
0/1/2/3/4 semantic labels.

Labels: 0 background, 1 sacrum, 2 left_hip, 3 right_hip, 4 fracture

Output metrics per case (and mean/std summary):

Semantic segmentation
    dice / iou : sacrum, left_hip, right_hip, fracture

Left / right hip discrimination  (v2 new)
    left_to_right_confusion  : % of gt-left-hip voxels predicted as right-hip
    right_to_left_confusion  : % of gt-right-hip voxels predicted as left-hip

Fracture
    precision / recall / dice / iou : fracture
    fracture_outside_pelvis_ratio

Sacrum fracture  (v2 new)
    sacrum_frac_dice / precision / recall
    sacrum_frac_gt_ratio

Surface / boundary metrics  (v2 new, requires scipy)
    hd95_fracture, assd_fracture
    hd95_sacrum,   assd_sacrum
    surface_dice_fracture          (soft surface Dice at tolerance 2 mm)

Usage
-----
python evaluate_agss_full_volume_v2.py \\
    --pred-folder /path/to/predictions \\
    --gt-folder   /path/to/ground_truth \\
    --out-csv     results/metrics.csv \\
    --summary-json results/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    _SCIPY_OK = True
except ImportError:
    binary_erosion = None
    distance_transform_edt = None
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_arr(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = np.asarray(img.GetSpacing()[::-1], dtype=np.float32)  # z, y, x
    return arr.astype(np.int16), spacing


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def dice_iou_prec_rec(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou  = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    prec = (tp + 1e-6) / (tp + fp + 1e-6)
    rec  = (tp + 1e-6) / (tp + fn + 1e-6)
    return float(dice), float(iou), float(prec), float(rec)


# ---------------------------------------------------------------------------
# Surface / boundary metrics
# ---------------------------------------------------------------------------

def surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[float, float]:
    """Return (HD95, ASSD) in mm; NaN when either mask is empty."""
    if not _SCIPY_OK:
        return float('nan'), float('nan')
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return float('nan'), float('nan')
    pred_s = pred ^ binary_erosion(pred)
    gt_s   = gt   ^ binary_erosion(gt)
    dt_gt   = distance_transform_edt(~gt_s,   sampling=spacing)
    dt_pred = distance_transform_edt(~pred_s, sampling=spacing)
    d1 = dt_gt[pred_s]
    d2 = dt_pred[gt_s]
    all_d = np.concatenate([d1, d2]) if len(d1) + len(d2) > 0 else np.asarray([0.0])
    return float(np.percentile(all_d, 95)), float(np.mean(all_d))


def surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: np.ndarray,
    tolerance_mm: float = 2.0,
) -> float:
    """Surface Dice at a given distance tolerance (mm).

    Returns NaN if scipy unavailable or either mask empty.
    """
    if not _SCIPY_OK:
        return float('nan')
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if not pred.any() or not gt.any():
        return float('nan')
    pred_s = pred ^ binary_erosion(pred)
    gt_s   = gt   ^ binary_erosion(gt)
    dt_gt   = distance_transform_edt(~gt_s,   sampling=spacing)
    dt_pred = distance_transform_edt(~pred_s, sampling=spacing)
    # Boundary voxels within tolerance of the other surface
    pred_within = pred_s & (dt_gt   <= tolerance_mm)
    gt_within   = gt_s   & (dt_pred <= tolerance_mm)
    num = pred_within.sum() + gt_within.sum()
    den = pred_s.sum() + gt_s.sum()
    return float(num) / float(den) if den > 0 else float('nan')


# ---------------------------------------------------------------------------
# Per-case computation
# ---------------------------------------------------------------------------

def compute_case_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: np.ndarray,
    surface_dice_tol_mm: float = 2.0,
) -> dict:
    row = {}

    # ---- Semantic Dice / IoU per class -----------------------------------
    for cls, name in [(1, 'sacrum'), (2, 'left_hip'), (3, 'right_hip'), (4, 'fracture')]:
        d, i, p, r = dice_iou_prec_rec(pred == cls, gt == cls)
        row[f'dice_{name}']     = d
        row[f'iou_{name}']      = i
        row[f'prec_{name}']     = p
        row[f'rec_{name}']      = r

    # ---- Left / right hip confusion (v2) ---------------------------------
    gt_left  = gt == 2
    gt_right = gt == 3
    pred_arr = pred   # alias for clarity
    left_total  = gt_left.sum()
    right_total = gt_right.sum()

    row['left_to_right_confusion'] = (
        float(np.logical_and(gt_left, pred_arr == 3).sum()) / max(left_total, 1)
    )
    row['right_to_left_confusion'] = (
        float(np.logical_and(gt_right, pred_arr == 2).sum()) / max(right_total, 1)
    )

    # ---- Fracture outside pelvis -----------------------------------------
    pred_frac   = pred == 4
    pred_pelvis = (pred > 0) & (pred != 4)   # anatomy classes
    gt_pelvis   = (gt > 0) & (gt != 4)
    # Using GT pelvis as reference (more stable than pred anatomy)
    outside_gt_pelvis = float(
        np.logical_and(pred_frac, ~gt_pelvis).sum()
    ) / max(pred_frac.sum(), 1)
    row['fracture_outside_gt_pelvis_ratio'] = outside_gt_pelvis

    # ---- Sacrum fracture: voxels of class 4 that overlap GT sacrum -------
    # Defined as: ground truth = fracture (4) AND located within GT sacrum bounding region.
    # Proxy: GT sacrum dilated by 1 cm (approx) vs GT fracture intersection.
    gt_sacrum = gt == 1
    gt_sacrum_frac = np.logical_and(gt == 4, _dilate_mask(gt_sacrum, radius_voxels=10))
    pred_sacrum_frac = np.logical_and(pred == 4, _dilate_mask(gt_sacrum, radius_voxels=10))
    sf_d, sf_i, sf_p, sf_r = dice_iou_prec_rec(pred_sacrum_frac, gt_sacrum_frac)
    row['sacrum_frac_dice']     = sf_d
    row['sacrum_frac_iou']      = sf_i
    row['sacrum_frac_precision'] = sf_p
    row['sacrum_frac_recall']   = sf_r
    row['sacrum_frac_gt_voxels'] = int(gt_sacrum_frac.sum())

    # ---- Surface / boundary metrics (fracture and sacrum) ----------------
    hd95_f, assd_f = surface_distances(pred == 4, gt == 4, spacing)
    row['hd95_fracture']  = hd95_f
    row['assd_fracture']  = assd_f

    hd95_s, assd_s = surface_distances(pred == 1, gt == 1, spacing)
    row['hd95_sacrum']  = hd95_s
    row['assd_sacrum']  = assd_s

    row['surface_dice_fracture'] = surface_dice(pred == 4, gt == 4, spacing, tolerance_mm=surface_dice_tol_mm)
    row['surface_dice_sacrum']   = surface_dice(pred == 1, gt == 1, spacing, tolerance_mm=surface_dice_tol_mm)
    row['surface_dice_left_hip'] = surface_dice(pred == 2, gt == 2, spacing, tolerance_mm=surface_dice_tol_mm)
    row['surface_dice_right_hip']= surface_dice(pred == 3, gt == 3, spacing, tolerance_mm=surface_dice_tol_mm)

    return row


def _dilate_mask(mask: np.ndarray, radius_voxels: int) -> np.ndarray:
    """Simple binary dilation using distance transform."""
    if not _SCIPY_OK:
        return mask
    from scipy.ndimage import distance_transform_edt as _dt
    inv = ~mask.astype(bool)
    dist = _dt(inv)
    return dist <= radius_voxels


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def aggregate_summary(rows: list) -> dict:
    if not rows:
        return {}
    summary = {}
    float_keys = [k for k in rows[0] if k != 'case']
    for k in float_keys:
        vals = [r[k] for r in rows if not np.isnan(float(r[k]))]
        summary[f'{k}_mean'] = float(np.mean(vals)) if vals else float('nan')
        summary[f'{k}_std']  = float(np.std(vals))  if vals else float('nan')
        summary[f'{k}_median'] = float(np.median(vals)) if vals else float('nan')
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="AGSS full-volume evaluation v2")
    ap.add_argument('--pred-folder',    required=True, type=Path)
    ap.add_argument('--gt-folder',      required=True, type=Path)
    ap.add_argument('--file-ending',    default='.nii.gz')
    ap.add_argument('--out-csv',        required=True, type=Path)
    ap.add_argument('--summary-json',   required=True, type=Path)
    ap.add_argument('--surface-dice-tol-mm', default=2.0, type=float,
                    help='Tolerance in mm for surface Dice (default 2.0)')
    args = ap.parse_args()

    pred_files = sorted(args.pred_folder.glob(f'*{args.file_ending}'))
    if not pred_files:
        print(f'[WARN] No prediction files found in {args.pred_folder}')

    rows = []
    for pf in pred_files:
        case = pf.name[:-len(args.file_ending)]
        gf   = args.gt_folder / pf.name
        if not gf.is_file():
            print(f'[skip] missing GT for {case}: {gf}')
            continue

        pred, spacing = read_arr(pf)
        gt,   _       = read_arr(gf)

        if pred.shape != gt.shape:
            raise RuntimeError(f'Shape mismatch for {case}: pred {pred.shape}, gt {gt.shape}')

        row = {'case': case}
        row.update(compute_case_metrics(pred, gt, spacing, args.surface_dice_tol_mm))
        rows.append(row)
        print(
            f'[{case}]  frac_dice={row["dice_fracture"]:.3f}  '
            f'sacrum={row["dice_sacrum"]:.3f}  '
            f'L/R conf={row["left_to_right_confusion"]:.3f}/{row["right_to_left_confusion"]:.3f}  '
            f'sacrum_frac={row["sacrum_frac_dice"]:.3f}  '
            f'surf_dice_frac={row["surface_dice_fracture"]:.3f}'
        )

    # Write per-case CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    # Write summary JSON
    summary = aggregate_summary(rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\nWrote {len(rows)} cases to {args.out_csv}')
    print(f'Summary: {args.summary_json}')

    # Quick console summary of key metrics
    if summary:
        print('\n=== Summary (mean ± std) ===')
        key_pairs = [
            ('dice_sacrum',           'Sacrum Dice'),
            ('dice_left_hip',         'Left hip Dice'),
            ('dice_right_hip',        'Right hip Dice'),
            ('dice_fracture',         'Fracture Dice'),
            ('left_to_right_confusion',  'L→R confusion'),
            ('right_to_left_confusion',  'R→L confusion'),
            ('sacrum_frac_dice',      'Sacrum frac Dice'),
            ('hd95_fracture',         'HD95 fracture (mm)'),
            ('surface_dice_fracture', 'Surface Dice fracture'),
        ]
        for k, label in key_pairs:
            m_key, s_key = f'{k}_mean', f'{k}_std'
            if m_key in summary:
                print(f'  {label:30s}  {summary[m_key]:.4f} ± {summary.get(s_key, float("nan")):.4f}')


if __name__ == '__main__':
    main()
