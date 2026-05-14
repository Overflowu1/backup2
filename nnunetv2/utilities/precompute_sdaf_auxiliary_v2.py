#!/usr/bin/env python3
"""
Precompute SDAF auxiliary targets for older nnU-Net v2 preprocessed folders.

Examples:
  # instance labels: 0 background, 1-3 anatomy, >=4 fragments
  python tools/precompute_sdaf_auxiliary_v2.py \
    --preprocessed-folder /mnt/data/DATA/nnUNet_preprocessed/Dataset102_Frc3/nnUNetPlans_3d_lowres \
    --fracture-start-label 4 --num-workers 8 --overwrite

  # binary labels: 0 background, 1 fracture
  python tools/precompute_sdaf_auxiliary_v2.py \
    --preprocessed-folder /mnt/data/DATA/nnUNet_preprocessed/Dataset102_Frc3/nnUNetPlans_3d_lowres \
    --fracture-start-label 1 --num-workers 8 --overwrite
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from nnunetv2.net.YuNet.sdaf_auxiliary import (
    DEFAULT_AFFINITY_OFFSETS,
    has_real_instance_ids,
    make_affinity_labels,
    make_contact_field,
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
    save_npy: bool,
) -> Dict[str, object]:
    folder_p = Path(folder)
    out_p = Path(out_folder)
    out_npz = out_p / f'{case_id}.npz'
    out_npy = out_p / f'{case_id}.npy'
    if out_npz.is_file() and ((not save_npy) or out_npy.is_file()) and not overwrite:
        return {'case_id': case_id, 'status': 'skip'}

    seg = _load_seg(folder_p, case_id)
    labels = [int(i) for i in np.unique(seg)]
    real_inst = has_real_instance_ids(seg, fracture_start_label=fracture_start_label)
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
    contact_nonzero = int((struct[2] > 1e-5).sum())
    aff_positive = int((affinity > 0.5).sum())

    np.savez(
        out_npz,
        struct=struct.astype(np.float16),
        affinity=(affinity > 0.5).astype(np.uint8),
    )
    if save_npy:
        combined = np.vstack((struct.astype(np.float16), (affinity > 0.5).astype(np.float16)))
        np.save(out_npy, combined)

    return {
        'case_id': case_id,
        'status': 'done',
        'labels': labels,
        'real_instance_ids': bool(real_inst),
        'contact_voxels': contact_nonzero,
        'affinity_positive_voxels': aff_positive,
        'shape': list(seg.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-folder', required=True, type=str)
    parser.add_argument('--fracture-start-label', default=4, type=int)
    parser.add_argument('--surface-sigma', default=2.0, type=float)
    parser.add_argument('--contact-window', default=7, type=int)
    parser.add_argument('--contact-sigma', default=2.0, type=float)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save-npy', action='store_true', help='Also save combined aux as .npy for faster loading at the cost of disk space.')
    args = parser.parse_args()

    folder = Path(args.preprocessed_folder)
    if not folder.is_dir():
        raise RuntimeError(f'Preprocessed folder not found: {folder}')

    out_folder = folder / 'sdaf_aux'
    out_folder.mkdir(exist_ok=True)
    ids = _case_ids(folder)
    if len(ids) == 0:
        raise RuntimeError(f'No .npz cases found in {folder}')

    print('Precomputing SDAF auxiliary targets')
    print(f'  folder: {folder}')
    print(f'  output: {out_folder}')
    print(f'  cases: {len(ids)}')
    print(f'  fracture_start_label: {args.fracture_start_label}')
    print(f'  save_npy: {args.save_npy}')

    fn = partial(
        _process_one,
        folder=str(folder),
        out_folder=str(out_folder),
        fracture_start_label=args.fracture_start_label,
        surface_sigma=args.surface_sigma,
        contact_window=args.contact_window,
        contact_sigma=args.contact_sigma,
        overwrite=args.overwrite,
        save_npy=args.save_npy,
    )

    if args.num_workers <= 1:
        results = [fn(i) for i in ids]
    else:
        with mp.get_context('spawn').Pool(args.num_workers) as pool:
            results = pool.map(fn, ids)

    done = sum(1 for r in results if r.get('status') == 'done')
    skip = sum(1 for r in results if r.get('status') == 'skip')
    real_inst = sum(1 for r in results if r.get('real_instance_ids') is True)
    contact_cases = sum(1 for r in results if int(r.get('contact_voxels', 0)) > 0)
    no_contact_cases = [r.get('case_id') for r in results if r.get('status') == 'done' and int(r.get('contact_voxels', 0)) == 0]

    report = {
        'preprocessed_folder': str(folder),
        'output_folder': str(out_folder),
        'fracture_start_label': args.fracture_start_label,
        'num_cases': len(ids),
        'done': done,
        'skip': skip,
        'cases_with_real_instance_ids': real_inst,
        'cases_with_nonzero_contact_field': contact_cases,
        'no_contact_cases': no_contact_cases[:50],
        'results': results,
    }
    report_file = out_folder / 'sdaf_aux_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'Finished. done={done}, skip={skip}')
    print(f'Cases with real instance ids: {real_inst}/{len(ids)}')
    print(f'Cases with nonzero contact field: {contact_cases}/{len(ids)}')
    if contact_cases == 0:
        print('WARNING: all D_contact maps are empty. This usually means binary labels or no adjacent instance labels.')
        print('         D_core/D_surface/affinity can still train, but contact-surface claims need instance labels.')
    print(f'Report saved to: {report_file}')


if __name__ == '__main__':
    main()
