"""
AGSS-aware dataloader for older nnU-Net v2 (v3.1 hybrid / lite-safe version).

Key fixes:
- Do NOT load full-volume AGSS aux as float32.
- Use npy mmap for AGSS aux.
- Convert aux to float32 only after patch cropping.
- Avoid loading aux twice per case.
- Coordinate maps are patch-wise and OFF by default.
- Add try/except around generate_train_batch to print real worker errors.
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, Tuple, List

import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "t", "yes", "y")


def _agss_enabled() -> bool:
    return _env_flag("NNUNET_AGSS_USE_PRECOMPUTED", "0")


def _agss_num_aux_channels() -> int:
    return int(os.environ.get("NNUNET_AGSS_NUM_AUX_CHANNELS", "8"))


def _coord_map_enabled() -> bool:
    # Lite 版建议默认关闭。需要坐标图时，在 trainer/env 里显式设为 1。
    return _env_flag("NNUNET_AGSS_COORD_MAP", "0")


def _sacrum_frac_oversample_ratio() -> float:
    return float(os.environ.get("NNUNET_AGSS_SACRUM_FRAC_OVERSAMPLE_RATIO", "0.0"))


def _build_patch_coord_map(
    full_shape: Tuple[int, int, int],
    valid_bbox_lbs: List[int],
    valid_bbox_ubs: List[int],
) -> np.ndarray:
    """
    Return patch-wise (3, d, h, w) float32 normalized coordinates in [-1, 1].

    Only builds coordinates for the valid in-volume patch, then the caller pads it.
    This avoids materializing full-volume coordinate maps.
    """
    D, H, W = full_shape

    z = np.linspace(-1.0, 1.0, D, dtype=np.float32)[valid_bbox_lbs[0]:valid_bbox_ubs[0]].reshape(-1, 1, 1)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)[valid_bbox_lbs[1]:valid_bbox_ubs[1]].reshape(1, -1, 1)
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)[valid_bbox_lbs[2]:valid_bbox_ubs[2]].reshape(1, 1, -1)

    p_d = int(valid_bbox_ubs[0] - valid_bbox_lbs[0])
    p_h = int(valid_bbox_ubs[1] - valid_bbox_lbs[1])
    p_w = int(valid_bbox_ubs[2] - valid_bbox_lbs[2])

    coord = np.broadcast_arrays(
        np.broadcast_to(z, (p_d, p_h, p_w)),
        np.broadcast_to(y, (p_d, p_h, p_w)),
        np.broadcast_to(x, (p_d, p_h, p_w)),
    )
    return np.stack(coord, axis=0).astype(np.float32, copy=False)


def _parse_weighted_fg_sampling():
    classes_str = os.environ.get("NNUNET_AGSS_FG_CLASSES", "").strip()
    weights_str = os.environ.get("NNUNET_AGSS_FG_WEIGHTS", "").strip()

    if len(classes_str) == 0:
        return None, None

    classes = [int(i.strip()) for i in classes_str.split(",") if len(i.strip()) > 0]
    if len(classes) == 0:
        return None, None

    if len(weights_str) == 0:
        weights = np.ones(len(classes), dtype=np.float64)
    else:
        weights = np.asarray(
            [float(i.strip()) for i in weights_str.split(",") if len(i.strip()) > 0],
            dtype=np.float64,
        )
        if len(weights) != len(classes):
            weights = np.ones(len(classes), dtype=np.float64)

    weights = np.clip(weights, 1e-8, None)
    weights /= weights.sum()
    return classes, weights


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def determine_shapes(self):
        data, seg, properties = self._data.load_case(self.indices[0])

        num_color_channels = data.shape[0]
        if _coord_map_enabled():
            num_color_channels += 3

        seg_channels = seg.shape[0]
        if _agss_enabled():
            if seg_channels != 1:
                raise RuntimeError(
                    "AGSS precomputed mode supports only one original segmentation channel. "
                    f"Got seg.shape[0]={seg_channels}. Do not use this trainer for cascade training."
                )
            seg_channels = seg_channels + _agss_num_aux_channels()

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg_channels, *self.patch_size)
        return data_shape, seg_shape

    def _get_aux_folder(self, case_id: str) -> str:
        env_folder = os.environ.get("NNUNET_AGSS_AUX_FOLDER", None)
        if env_folder is not None and len(env_folder) > 0:
            return env_folder

        entry = self._data[case_id]
        return os.path.join(os.path.dirname(entry["data_file"]), "agss_aux")

    def _cache_enabled(self) -> bool:
        # 不建议开启。完整 aux volume 很大，多 worker 下容易爆内存。
        return _env_flag("NNUNET_AGSS_CACHE_AUX", "0")

    def _load_agss_aux(self, case_id: str, expected_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Load AGSS auxiliary target.

        Important:
        - npy uses mmap_mode='r', so we do not load the full aux volume into RAM.
        - Do NOT astype(float32) here.
        - Convert to float32 only after patch cropping in generate_train_batch().
        """
        if self._cache_enabled():
            if not hasattr(self, "_agss_aux_cache"):
                self._agss_aux_cache: Dict[str, np.ndarray] = {}
            if case_id in self._agss_aux_cache:
                return self._agss_aux_cache[case_id]

        aux_folder = self._get_aux_folder(case_id)
        npy_file = os.path.join(aux_folder, f"{case_id}.npy")
        npz_file = os.path.join(aux_folder, f"{case_id}.npz")

        if os.path.isfile(npy_file):
            aux = np.load(npy_file, mmap_mode="r")
        elif os.path.isfile(npz_file):
            # npz cannot be truly memory-mapped. Prefer .npy for AGSS aux.
            arr = np.load(npz_file)
            if "aux" not in arr:
                raise RuntimeError(f"AGSS npz file must contain key 'aux': {npz_file}")
            aux = arr["aux"]
        else:
            raise RuntimeError(
                f"Missing AGSS auxiliary file: {npz_file} or {npy_file}\n"
                "Please run precompute_agss_auxiliary.py first."
            )

        if aux.shape[0] < _agss_num_aux_channels():
            raise RuntimeError(
                f"AGSS aux has too few channels: case={case_id}, "
                f"shape={aux.shape}, required={_agss_num_aux_channels()}"
            )

        aux = aux[:_agss_num_aux_channels()]

        if tuple(aux.shape[1:]) != tuple(expected_shape):
            raise RuntimeError(
                f"AGSS aux shape mismatch for {case_id}: got aux.shape={aux.shape}, "
                f"expected spatial shape={expected_shape}. "
                "Re-run precompute_agss_auxiliary.py for this exact nnU-Net configuration."
            )

        if self._cache_enabled():
            self._agss_aux_cache[case_id] = aux

        return aux

    def _choose_fg_class(self, class_locations: dict):
        classes, weights = _parse_weighted_fg_sampling()
        if classes is None:
            return None

        eligible = [c for c in classes if c in class_locations and len(class_locations[c]) > 0]
        if len(eligible) == 0:
            return None

        idx_map = {c: i for i, c in enumerate(classes)}
        w = np.asarray([weights[idx_map[c]] for c in eligible], dtype=np.float64)
        w /= w.sum()

        return int(np.random.choice(np.asarray(eligible, dtype=np.int64), p=w))

    def _get_bbox_sacrum_frac_biased(
        self,
        shape: Tuple[int, ...],
        properties: dict,
        aux: np.ndarray,
    ):
        """
        Bias patch selection toward sacrum-fracture voxels, aux channel 4.

        This path should be OFF for Lite by default:
            NNUNET_AGSS_SACRUM_FRAC_OVERSAMPLE_RATIO=0.0

        If enabled, this samples one positive voxel without materializing coords.tolist().
        """
        if aux is not None and aux.shape[0] > 4:
            sf_mask = aux[4] > 0.5
            flat = np.flatnonzero(sf_mask)

            if flat.size > 0:
                dim = len(shape)
                chosen_flat = flat[np.random.randint(0, flat.size)]
                chosen = np.asarray(np.unravel_index(chosen_flat, shape), dtype=np.int64)

                lbs, ubs = [], []
                for d in range(dim):
                    ps = int(self.patch_size[d])

                    if shape[d] <= ps:
                        lo = 0
                    else:
                        lo = int(chosen[d]) - ps // 2
                        lo = max(0, min(lo, int(shape[d]) - ps))

                    hi = lo + ps
                    lbs.append(lo)
                    ubs.append(hi)

                return lbs, ubs

        return self.get_bbox(
            shape,
            force_fg=True,
            class_locations=properties["class_locations"],
        )

    def generate_train_batch(self):
        try:
            selected_keys = self.get_indices()

            data_all = np.zeros(self.data_shape, dtype=np.float32)
            seg_all = np.zeros(
                self.seg_shape,
                dtype=np.float32 if _agss_enabled() else np.int16,
            )

            case_properties = []
            sacrum_ratio = _sacrum_frac_oversample_ratio()

            for j, i in enumerate(selected_keys):
                force_fg = self.get_do_oversample(j)

                data, seg, properties = self._data.load_case(i)
                case_properties.append(properties)

                if _agss_enabled() and seg.shape[0] != 1:
                    raise RuntimeError(
                        "AGSS precomputed mode does not support cascaded previous-stage segmentation. "
                        f"Case {i} has seg.shape={seg.shape}."
                    )

                shape = data.shape[1:]
                dim = len(shape)

                aux = None
                overwrite_class = None

                if force_fg:
                    use_sacrum_bias = (
                        _agss_enabled()
                        and sacrum_ratio > 0.0
                        and np.random.random() < sacrum_ratio
                    )

                    if use_sacrum_bias:
                        aux = self._load_agss_aux(i, tuple(shape))
                        bbox_lbs, bbox_ubs = self._get_bbox_sacrum_frac_biased(
                            shape,
                            properties,
                            aux,
                        )
                    else:
                        overwrite_class = self._choose_fg_class(properties["class_locations"])

                        if overwrite_class is None:
                            force_class_env = os.environ.get("NNUNET_FORCE_FG_CLASS", "").strip()
                            if force_class_env != "":
                                try:
                                    force_class = int(force_class_env)
                                    locs = properties["class_locations"].get(force_class, [])
                                    if len(locs) > 0:
                                        overwrite_class = force_class
                                except Exception:
                                    overwrite_class = None

                        bbox_lbs, bbox_ubs = self.get_bbox(
                            shape,
                            force_fg,
                            properties["class_locations"],
                            overwrite_class=overwrite_class,
                        )
                else:
                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape,
                        force_fg,
                        properties["class_locations"],
                        overwrite_class=overwrite_class,
                    )

                valid_bbox_lbs = [max(0, bbox_lbs[d]) for d in range(dim)]
                valid_bbox_ubs = [min(shape[d], bbox_ubs[d]) for d in range(dim)]

                padding = [
                    (-min(0, bbox_lbs[d]), max(bbox_ubs[d] - shape[d], 0))
                    for d in range(dim)
                ]

                # -------------------------
                # image patch
                # -------------------------
                data_slice = tuple(
                    [slice(0, data.shape[0])]
                    + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)]
                )
                data_patch = data[data_slice]

                data_padded = np.pad(
                    data_patch,
                    ((0, 0), *padding),
                    "constant",
                    constant_values=0,
                ).astype(np.float32, copy=False)

                if _coord_map_enabled():
                    coord_patch = _build_patch_coord_map(
                        tuple(shape),
                        valid_bbox_lbs,
                        valid_bbox_ubs,
                    )
                    coord_padded = np.pad(
                        coord_patch,
                        ((0, 0), *padding),
                        "constant",
                        constant_values=0,
                    ).astype(np.float32, copy=False)

                    data_all[j] = np.concatenate([data_padded, coord_padded], axis=0)
                else:
                    data_all[j] = data_padded

                # -------------------------
                # semantic seg patch
                # -------------------------
                seg_slice = tuple(
                    [slice(0, seg.shape[0])]
                    + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)]
                )
                seg_patch = seg[seg_slice]

                seg_padded = np.pad(
                    seg_patch,
                    ((0, 0), *padding),
                    "constant",
                    constant_values=-1,
                )

                # -------------------------
                # AGSS aux patch
                # -------------------------
                if _agss_enabled():
                    if aux is None:
                        aux = self._load_agss_aux(i, tuple(shape))

                    aux_slice = tuple(
                        [slice(0, aux.shape[0])]
                        + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)]
                    )

                    aux_patch = aux[aux_slice]

                    # Critical:
                    # only convert the cropped patch to float32,
                    # never the full aux volume.
                    aux_patch = np.asarray(aux_patch, dtype=np.float32)

                    aux_padded = np.pad(
                        aux_patch,
                        ((0, 0), *padding),
                        "constant",
                        constant_values=0,
                    ).astype(np.float32, copy=False)

                    seg_all[j] = np.vstack(
                        (
                            seg_padded.astype(np.float32, copy=False),
                            aux_padded,
                        )
                    )
                else:
                    seg_all[j] = seg_padded

            return {
                "data": data_all,
                "seg": seg_all,
                "properties": case_properties,
                "keys": selected_keys,
            }

        except Exception as e:
            print("\n[AGSS DataLoader ERROR]", repr(e), flush=True)
            traceback.print_exc()
            raise