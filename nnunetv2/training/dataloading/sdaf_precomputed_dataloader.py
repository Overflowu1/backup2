from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetSDAFPrecomputedDataLoader(nnUNetDataLoader):
    """nnU-Net dataloader that loads precomputed SDAF aux maps and crops them with the same bbox."""

    def __init__(
        self,
        data: nnUNetBaseDataset,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        label_manager: LabelManager,
        oversample_foreground_percent: float = 0.0,
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        pad_sides: Union[List[int], Tuple[int, ...]] = None,
        probabilistic_oversampling: bool = False,
        transforms=None,
        aux_folder: Union[str, Path, None] = None,
        num_struct_channels: int = 3,
        num_affinity_channels: int = 6,
    ):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)
        self.aux_folder = Path(aux_folder) if aux_folder is not None else Path(getattr(data, 'source_folder')) / 'sdaf_aux'
        self.num_struct_channels = int(num_struct_channels)
        self.num_affinity_channels = int(num_affinity_channels)
        if not self.aux_folder.is_dir():
            raise RuntimeError(f'SDAF aux folder not found: {self.aux_folder}. Run precompute_sdaf_auxiliary.py first.')

    def _load_aux(self, case_id: str):
        fn = self.aux_folder / f'{case_id}.npz'
        if not fn.is_file():
            raise RuntimeError(f'Missing SDAF aux file: {fn}')
        arr = np.load(fn)
        struct = arr['struct'].astype(np.float32, copy=False)
        affinity = arr['affinity'].astype(np.float32, copy=False)
        return struct, affinity

    def _split(self, seg_sample):
        # Deep supervision transform returns a list. Use high-res aux from seg_sample[0].
        if isinstance(seg_sample, list):
            high = seg_sample[0]
            target = [s[0:1].round().to(torch.int16) for s in seg_sample]
        else:
            high = seg_sample
            target = high[0:1].round().to(torch.int16)
        s0 = 1
        s1 = s0 + self.num_struct_channels
        a1 = s1 + self.num_affinity_channels
        if high.shape[0] < a1:
            raise RuntimeError(f'Combined segmentation has {high.shape[0]} channels, expected >= {a1}')
        struct = high[s0:s1].float().clamp_(0, 1)
        affinity = high[s1:a1].float().clamp_(0, 1)
        return target, struct, affinity

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = target_all = struct_all = affinity_all = None
        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    if seg_prev is not None:
                        raise RuntimeError('SDAF precomputed dataloader does not support cascaded previous-stage seg yet.')
                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                    struct, affinity = self._load_aux(i)
                    struct_cropped = torch.from_numpy(crop_and_pad_nd(struct, bbox, 0)).float()
                    affinity_cropped = torch.from_numpy(crop_and_pad_nd(affinity, bbox, 0)).float()

                    # Keep alignment through existing spatial/mirror transforms by temporarily appending aux to seg.
                    combined_seg = torch.cat([seg_cropped.float(), struct_cropped, affinity_cropped], dim=0)
                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        combined_seg = combined_seg[:, 0]
                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': combined_seg})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = combined_seg
                    target_sample, struct_sample, affinity_sample = self._split(seg_sample)

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample
                    if isinstance(target_sample, list):
                        if target_all is None:
                            target_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in target_sample]
                        for s_idx, s in enumerate(target_sample):
                            target_all[s_idx][j] = s
                    else:
                        if target_all is None:
                            target_all = torch.empty((self.batch_size, *target_sample.shape), dtype=target_sample.dtype)
                        target_all[j] = target_sample
                    if struct_all is None:
                        struct_all = torch.empty((self.batch_size, *struct_sample.shape), dtype=torch.float32)
                    if affinity_all is None:
                        affinity_all = torch.empty((self.batch_size, *affinity_sample.shape), dtype=torch.float32)
                    struct_all[j] = struct_sample
                    affinity_all[j] = affinity_sample
        return {'data': data_all, 'target': target_all, 'sdaf_struct': struct_all, 'sdaf_affinity': affinity_all, 'keys': selected_keys}
