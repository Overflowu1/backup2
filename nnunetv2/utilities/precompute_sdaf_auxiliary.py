#!/usr/bin/env python3
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.net.YuNet.sdaf_auxiliary import DEFAULT_AFFINITY_OFFSETS, make_affinity_labels, make_structural_fields


def parse_offsets(s: str):
    if s.lower() in ("default", "6", "6n"):
        return DEFAULT_AFFINITY_OFFSETS
    return tuple(tuple(int(v) for v in item.split(',')) for item in s.split(';') if item.strip())


def worker(args):
    folder, case_id, out_folder, fracture_start_label, surface_sigma, contact_window, contact_sigma, offsets, overwrite = args
    out_file = Path(out_folder) / f"{case_id}.npz"
    if out_file.is_file() and not overwrite:
        return case_id, 'skip'
    ds_cls = infer_dataset_class(folder)
    ds = ds_cls(folder, identifiers=[case_id])
    _, seg, _, _ = ds.load_case(case_id)
    seg = np.asarray(seg)
    label = seg[0] if seg.ndim == 4 and seg.shape[0] == 1 else (np.argmax(seg, axis=0) if seg.ndim == 4 else seg)
    label = label.astype(np.int32, copy=False)
    struct = make_structural_fields(label, fracture_start_label, surface_sigma, contact_window, contact_sigma).astype(np.float16)
    affinity = (make_affinity_labels(label, offsets, fracture_start_label) > 0.5).astype(np.uint8)
    np.savez_compressed(out_file, struct=struct, affinity=affinity)
    return case_id, 'ok'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preprocessed-folder', required=True)
    p.add_argument('--fracture-start-label', type=int, default=4)
    p.add_argument('--surface-sigma', type=float, default=2.0)
    p.add_argument('--contact-window', type=int, default=7)
    p.add_argument('--contact-sigma', type=float, default=2.0)
    p.add_argument('--offsets', default='default')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    folder = Path(a.preprocessed_folder)
    out_folder = folder / 'sdaf_aux'
    out_folder.mkdir(exist_ok=True)
    ds_cls = infer_dataset_class(str(folder))
    ds = ds_cls(str(folder), identifiers=None)
    case_ids = list(ds.identifiers)
    offsets = parse_offsets(a.offsets)
    jobs = [(str(folder), c, str(out_folder), a.fracture_start_label, a.surface_sigma, a.contact_window, a.contact_sigma, offsets, a.overwrite) for c in case_ids]
    print(f'[SDAF] {len(jobs)} cases -> {out_folder}')
    if a.num_workers <= 1:
        for j in jobs:
            print(worker(j))
    else:
        with Pool(a.num_workers) as pool:
            for cid, status in pool.imap_unordered(worker, jobs):
                print(f'{cid}: {status}')
    print('[SDAF] Done')


if __name__ == '__main__':
    main()
