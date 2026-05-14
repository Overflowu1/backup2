#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute AGSS hard-fragment sampling coordinates.

Input:
    agss_aux/CASE.npy or CASE.npz, shape = (8, D, H, W)

Output:
    agss_hard_locs/CASE.npz with keys:
        small   : coords of small fracture components, N x 3
        sacfrac : coords of sacrum-fracture voxels, N x 3
        surface : coords of fracture surface voxels, N x 3
        frac    : coords of all fracture voxels, N x 3

Aux channel convention:
    aux[0] = y_frac
    aux[4] = y_sacfrac
    aux[6] = D_surface
    aux[7] = small_weight
"""

from __future__ import annotations

import argparse
import os
import glob
from multiprocessing import Pool
from typing import Dict

import numpy as np


def _load_aux(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    data = np.load(path)
    if "aux" in data:
        return data["aux"]
    # fallback: first array in npz
    key = list(data.keys())[0]
    return data[key]


def _downsample_coords(coords: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.int32)

    if coords.shape[0] == 0:
        return coords.astype(np.int32, copy=False)

    if max_points > 0 and coords.shape[0] > max_points:
        idx = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[idx]

    return coords.astype(np.int32, copy=False)


def build_hard_locs_from_aux(
    aux: np.ndarray,
    small_thr: float = 1.2,
    surface_thr: float = 0.30,
    max_points_per_mode: int = 50000,
    seed: int = 12345,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    frac = aux[0] > 0.5

    if aux.shape[0] > 7:
        small_mask = frac & (aux[7] > float(small_thr))
    else:
        small_mask = np.zeros_like(frac, dtype=bool)

    if aux.shape[0] > 4:
        sacfrac_mask = aux[4] > 0.5
    else:
        sacfrac_mask = np.zeros_like(frac, dtype=bool)

    if aux.shape[0] > 6:
        # Restrict surface hard samples to fracture voxels.
        # Otherwise surface field may include a broad neighborhood.
        surface_mask = frac & (aux[6] > float(surface_thr))
    else:
        surface_mask = np.zeros_like(frac, dtype=bool)

    out = {
        "small": _downsample_coords(np.argwhere(small_mask), max_points_per_mode, rng),
        "sacfrac": _downsample_coords(np.argwhere(sacfrac_mask), max_points_per_mode, rng),
        "surface": _downsample_coords(np.argwhere(surface_mask), max_points_per_mode, rng),
        "frac": _downsample_coords(np.argwhere(frac), max_points_per_mode, rng),
    }

    return out


def process_one(args):
    aux_path, out_folder, small_thr, surface_thr, max_points = args
    case_id = os.path.basename(aux_path)
    case_id = case_id[:-4] if case_id.endswith(".npy") else case_id[:-4]

    aux = _load_aux(aux_path)
    hard = build_hard_locs_from_aux(
        aux,
        small_thr=small_thr,
        surface_thr=surface_thr,
        max_points_per_mode=max_points,
        seed=abs(hash(case_id)) % (2**32),
    )

    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, case_id + ".npz")
    np.savez_compressed(out_path, **hard)

    return case_id, {k: int(v.shape[0]) for k, v in hard.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux-folder", required=True)
    parser.add_argument("--out-folder", required=True)
    parser.add_argument("--small-thr", type=float, default=1.2)
    parser.add_argument("--surface-thr", type=float, default=0.30)
    parser.add_argument("--max-points-per-mode", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    aux_files = sorted(glob.glob(os.path.join(args.aux_folder, "*.npy")))
    aux_files += sorted(glob.glob(os.path.join(args.aux_folder, "*.npz")))
    aux_files = [
        p for p in aux_files
        if not os.path.basename(p).startswith("agss_aux_report")
    ]

    jobs = []
    for p in aux_files:
        case_id = os.path.basename(p)[:-4]
        out_path = os.path.join(args.out_folder, case_id + ".npz")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        jobs.append((p, args.out_folder, args.small_thr, args.surface_thr, args.max_points_per_mode))

    print(f"Found aux files: {len(aux_files)}")
    print(f"Jobs to write: {len(jobs)}")
    print(f"Output folder: {args.out_folder}")

    if len(jobs) == 0:
        return

    if args.num_workers <= 1:
        results = [process_one(j) for j in jobs]
    else:
        with Pool(args.num_workers) as pool:
            results = list(pool.imap_unordered(process_one, jobs))

    total = {"small": 0, "sacfrac": 0, "surface": 0, "frac": 0}
    for _, counts in results:
        for k in total:
            total[k] += counts.get(k, 0)

    print("Done.")
    print("Total sampled coords:", total)


if __name__ == "__main__":
    main()