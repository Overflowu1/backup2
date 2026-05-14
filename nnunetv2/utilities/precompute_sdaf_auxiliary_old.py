#!/usr/bin/env python3
"""
Precompute SDAF auxiliary targets for older nnU-Net v2 preprocessed folders.

Example for your current run:
python tools/precompute_sdaf_auxiliary_old.py \
  --preprocessed-folder /mnt/data/DATA/nnUNet_preprocessed/Dataset102_Frc3/nnUNetPlans_3d_lowres \
  --fracture-start-label 4 \
  --num-workers 8 \
  --overwrite

For binary labels 0=background, 1=fracture, use --fracture-start-label 1.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np

from nnunetv2.net.YuNet.sdaf_auxiliary import (
    DEFAULT_AFFINITY_OFFSETS,
    make_affinity_labels,
    make_structural_fields,
)


def _case_ids(folder: Path) -> List[str]:
    ids = []
    for f in os.listdir(folder):
        if f.endswith('.npz') and ('segFromPrevStage' not in f):
            ids.append(f[:-4])
    ids.sort()
    return ids


def _load_seg(folder: Path, case_id: str) -> np.ndarray:
    seg_npy = folder / f'{case_id}_seg.npy'
    npz_file = folder / f'{case_id}.npz'
    if seg_npy.is_file():
        seg = np.load(seg_npy, mmap_mode='r')
    elif npz_file.is_file():
        seg = np.load(npz_file)['seg']
    else:
        raise FileNotFoundError(f'Cannot find segmentation for case {case_id} in {folder}')
    # nnU-Net seg shape is usually (1, D, H, W). Use first channel.
    if seg.ndim == 4:
        seg = np.asarray(seg[0])
    elif seg.ndim != 3:
        raise RuntimeError(f'Unexpected seg shape for {case_id}: {seg.shape}')
    return np.asarray(seg).astype(np.int32, copy=False)


def _process_one(
    case_id: str,
    folder: str,
    out_folder: str,
    fracture_start_label: int,
    surface_sigma: float,
    contact_window: int,
    contact_sigma: float,
    overwrite: bool,
) -> Tuple[str, str]:
    folder_p = Path(folder)
    out_p = Path(out_folder)
    out_file = out_p / f'{case_id}.npz'
    if out_file.is_file() and not overwrite:
        return case_id, 'skip'

    seg = _load_seg(folder_p, case_id)
    struct = make_structural_fields(
        seg,
        fracture_start_label=fracture_start_label,
        surface_sigma=surface_sigma,
        contact_window=contact_window,
        contact_sigma=contact_sigma,
    )
    affinity = make_affinity_labels(
        seg,
        offsets=DEFAULT_AFFINITY_OFFSETS,
        fracture_start_label=fracture_start_label,
    )

    # Save uncompressed npz for faster dataloader reading. Values are compact anyway.
    # struct float16 is enough as supervision heatmap; affinity uint8 saves disk/RAM.
    np.savez(
        out_file,
        struct=struct.astype(np.float16),
        affinity=(affinity > 0.5).astype(np.uint8),
    )
    return case_id, 'done'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-folder', required=True, type=str)
    parser.add_argument('--fracture-start-label', default=4, type=int)
    parser.add_argument('--surface-sigma', default=2.0, type=float)
    parser.add_argument('--contact-window', default=7, type=int)
    parser.add_argument('--contact-sigma', default=2.0, type=float)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    folder = Path(args.preprocessed_folder)
    if not folder.is_dir():
        raise RuntimeError(f'Preprocessed folder not found: {folder}')

    out_folder = folder / 'sdaf_aux'
    out_folder.mkdir(exist_ok=True)

    ids = _case_ids(folder)
    if len(ids) == 0:
        raise RuntimeError(f'No .npz cases found in {folder}')

    print(f'Precomputing SDAF auxiliary targets')
    print(f'  folder: {folder}')
    print(f'  output: {out_folder}')
    print(f'  cases: {len(ids)}')
    print(f'  fracture_start_label: {args.fracture_start_label}')

    fn = partial(
        _process_one,
        folder=str(folder),
        out_folder=str(out_folder),
        fracture_start_label=args.fracture_start_label,
        surface_sigma=args.surface_sigma,
        contact_window=args.contact_window,
        contact_sigma=args.contact_sigma,
        overwrite=args.overwrite,
    )

    if args.num_workers <= 1:
        results = [fn(i) for i in ids]
    else:
        with mp.get_context('spawn').Pool(args.num_workers) as pool:
            results = pool.map(fn, ids)

    done = sum(1 for _, s in results if s == 'done')
    skip = sum(1 for _, s in results if s == 'skip')
    print(f'Finished. done={done}, skip={skip}')


if __name__ == '__main__':
    main()
