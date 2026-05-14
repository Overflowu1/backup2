"""
AGSS trainer v3.1 hybrid.

File organisation:
  L1-36    Imports
  L37-132  Helpers & metric keys
  L133-354 Validation metric functions (_split_highres_combined_target,
           _binary_metrics_from_masks, _compute_agss_validation_metrics)
  L356-821 AGSSArconvLstmASFEBTrainer (parent class)
  L822-1316  Ablation / variant trainers (NoAux, ACFA, Lite, FracOptimized, …)
  L1318-1541 AGSSArconvLstmASFEBTrainer_Clean + Clean helpers
  L1694-end  Clean ablation variants (Flat, NoACFA, CEFrac, …)

Adds over v3:
- coordinate-map support (z/y/x channels appended in the dataloader)
- optional ChannelSE skip-attention in the network
while keeping:
- hierarchical region/side/sacrum-fracture supervision
- balanced foreground oversampling across fracture + anatomy classes
- automatic disabling of mirroring along inferred left-right axis
- extended validation metrics for raw semantic, fused hierarchy, confusion,
  ARConv saturation/collapse, and UM-Fusion behavior
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from nnunetv2.net.YuNet.AGSSNet import AGSSUXlstmBotArconv
from nnunetv2.net.YuNet.AGSSNetClean import AGSSNetClean, assemble_anatomy_probs, get_agss_clean_from_plans
from nnunetv2.net.YuNet.agss_auxiliary import AGSSAuxiliaryLoss, AGSSLossWeights, AGSS_NUM_AUX_CHANNELS
from nnunetv2.net.YuNet.agss_loss_clean import AGSSCleanLoss, CleanLossWeights


@contextmanager
def _dummy_context():
    yield


def _autocast_context(device):
    if getattr(device, 'type', str(device)) == 'cuda':
        try:
            return torch.autocast(device_type='cuda', enabled=True)
        except Exception:
            return torch.cuda.amp.autocast(enabled=True)
    return _dummy_context()


def _move_to_device(obj, device):
    if isinstance(obj, (list, tuple)):
        return [i.to(device, non_blocking=True) for i in obj]
    return obj.to(device, non_blocking=True)


def _seg_only_target(target):
    if isinstance(target, (list, tuple)):
        out = []
        for t in target:
            if torch.is_tensor(t) and t.ndim >= 5 and t.shape[1] > 1:
                out.append(t[:, 0:1])
            else:
                out.append(t)
        return out
    if torch.is_tensor(target) and target.ndim >= 5 and target.shape[1] > 1:
        return target[:, 0:1]
    return target


def _reduce_agss_target_for_gpu(target):
    """Keep full AGSS channels only at highest resolution; lower deep-supervision
    targets keep the semantic channel only. This cuts GPU memory substantially.
    """
    if isinstance(target, (list, tuple)):
        out = []
        for i, t in enumerate(target):
            if torch.is_tensor(t) and t.ndim >= 5 and t.shape[1] > 1:
                out.append(t if i == 0 else t[:, 0:1].contiguous())
            else:
                out.append(t)
        return out
    return target


def _highest_resolution(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _safe_mean_numpy(values):
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(np.asarray(v))
            if not np.isnan(f):
                vals.append(f)
        except Exception:
            pass
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


# Validation metric keys must exist in every batch output because nnU-Net
# collates dictionaries key-by-key. On epochs where we intentionally skip
# expensive diagnostics, or on batches without sacrum-fracture voxels, these
# keys are filled with NaN so collate_outputs remains stable.
AGSS_VAL_METRIC_KEYS = [
    'agss_val_dice_sacrum', 'agss_val_dice_left_hip', 'agss_val_dice_right_hip', 'agss_val_dice_fracture',
    'agss_val_iou_sacrum', 'agss_val_iou_left_hip', 'agss_val_iou_right_hip', 'agss_val_iou_fracture',
    'agss_val_rawsem_dice_sacrum', 'agss_val_rawsem_dice_left_hip', 'agss_val_rawsem_dice_right_hip', 'agss_val_rawsem_dice_fracture',
    'agss_val_fracture_dice', 'agss_val_fracture_iou', 'agss_val_fracture_precision', 'agss_val_fracture_recall',
    'agss_val_pred_fracture_ratio', 'agss_val_gt_fracture_ratio',
    'agss_val_binfrac_dice', 'agss_val_binfrac_precision', 'agss_val_binfrac_recall',
    'agss_val_fracture_outside_pelvis_ratio',
    'agss_val_anat_dice_sacrum', 'agss_val_anat_dice_left_hip', 'agss_val_anat_dice_right_hip',
    'agss_val_region_acc', 'agss_val_side_acc_on_hip', 'agss_val_frac_anat_acc', 'agss_val_sacfrac_dice',
    'agss_val_gt_left_to_right', 'agss_val_gt_right_to_left', 'agss_val_gt_left_to_frac',
    'agss_val_gt_right_to_frac', 'agss_val_gt_sacrum_to_frac', 'agss_val_gt_frac_to_bg',
    'agss_val_gt_frac_to_left', 'agss_val_gt_frac_to_right', 'agss_val_gt_frac_to_sacrum',
    'agss_val_core_mae', 'agss_val_surface_mae', 'agss_val_struct_roi_mae',
    'agss_val_core_gt_ratio', 'agss_val_surface_gt_ratio',
    'agss_val_arconv_reg', 'agss_val_arconv_module_count', 'agss_val_arconv_offset_abs',
    'agss_val_arconv_offset_max', 'agss_val_arconv_offset_tv', 'agss_val_arconv_offset_sat_ratio',
    'agss_val_arconv_routing_entropy', 'agss_val_arconv_routing_max_prob', 'agss_val_arconv_routing_collapse_ratio',
    'agss_val_um_gate_mean', 'agss_val_um_gate_std', 'agss_val_um_gate_bg_mean',
    'agss_val_um_gate_fg_mean', 'agss_val_um_gate_fg_minus_bg',
    'agss_val_acfa_gate_mean', 'agss_val_acfa_gate_fg_mean',
    'agss_val_acfa_gate_bg_mean', 'agss_val_acfa_res_scale',
]


def _ensure_agss_val_metric_keys(d: dict):
    for k in AGSS_VAL_METRIC_KEYS:
        if k not in d:
            d[k] = np.float32(np.nan)
    return d


def _split_highres_combined_target(target):
    """Extract semantic + AGSS aux channels from combined target for validation.

    Target layout (after dataloader concatenation):
        0  semantic
        1  y_frac        6  D_core
        2  y_anat        7  D_surface
        3  y_region      8  small_weight (not extracted; only used by loss)
        4  y_side
        5  y_sacfrac

    Works with both 8-channel (sem + 7 aux) and 9-channel (sem + 8 aux) targets.
    """
    t = _highest_resolution(target)
    if not torch.is_tensor(t):
        return None, None, None, None, None, None, None, None
    sem = t[:, 0:1]
    if t.shape[1] >= 8:
        frac = t[:, 1:2]
        anat = t[:, 2:3]
        region = t[:, 3:4]
        side = t[:, 4:5]
        sacfrac = t[:, 5:6]
        struct = t[:, 6:8]  # D_core + D_surface; small_weight at index 8 is not needed for validation
    else:
        frac = (sem == 4).float()
        anat = sem.clone(); anat[anat == 4] = 0; anat[(anat < 0) | (anat > 3)] = 0
        region = torch.zeros_like(sem)
        region[anat == 1] = 1
        region[(anat == 2) | (anat == 3)] = 2
        side = torch.zeros_like(sem)
        side[anat == 2] = 1
        side[anat == 3] = 2
        sacfrac = ((frac > 0.5) & (anat == 1)).float()
        struct = None
    return sem, frac, anat, region, side, sacfrac, struct


def _binary_metrics_from_masks(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    tp = (pred & gt).float().sum()
    fp = (pred & (~gt)).float().sum()
    fn = ((~pred) & gt).float().sum()
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return dice, iou, precision, recall


def _compute_agss_validation_metrics(output, target, fracture_label: int = 4, ignore_label=None, full_metrics: bool = True):
    if not isinstance(output, dict):
        return {}
    seg_logits = _highest_resolution(output.get('seg'))
    if seg_logits is None or not torch.is_tensor(seg_logits):
        return {}

    sem_gt, frac_gt, anat_gt, region_gt, side_gt, sacfrac_gt, struct_gt = _split_highres_combined_target(target)
    if sem_gt is None:
        return {}
    sem_gt = sem_gt.to(device=seg_logits.device)
    valid = torch.ones_like(sem_gt, dtype=torch.bool)
    if ignore_label is not None:
        valid = sem_gt != ignore_label
    eps = seg_logits.new_tensor(1e-6)

    pred_sem = seg_logits.argmax(1, keepdim=True)

    metrics = {}
    # Main fused semantic output metrics
    for cls, name in [(1, 'sacrum'), (2, 'left_hip'), (3, 'right_hip'), (4, 'fracture')]:
        pred_c = (pred_sem == cls) & valid
        gt_c = (sem_gt == cls) & valid
        dice, iou, _, _ = _binary_metrics_from_masks(pred_c, gt_c, float(eps))
        metrics[f'agss_val_dice_{name}'] = dice.detach().cpu().numpy()
        metrics[f'agss_val_iou_{name}'] = iou.detach().cpu().numpy()

    # Binary fracture metrics from fused output
    pred_frac = (pred_sem == int(fracture_label)) & valid
    gt_frac = (sem_gt == int(fracture_label)) & valid
    dice, iou, precision, recall = _binary_metrics_from_masks(pred_frac, gt_frac, float(eps))
    metrics.update({
        'agss_val_fracture_dice': dice.detach().cpu().numpy(),
        'agss_val_fracture_iou': iou.detach().cpu().numpy(),
        'agss_val_fracture_precision': precision.detach().cpu().numpy(),
        'agss_val_fracture_recall': recall.detach().cpu().numpy(),
        'agss_val_pred_fracture_ratio': pred_frac.float().mean().detach().cpu().numpy(),
        'agss_val_gt_fracture_ratio': gt_frac.float().mean().detach().cpu().numpy(),
    })

    # Raw semantic auxiliary head metrics (diagnostic only, can be skipped on fast epochs)
    if full_metrics:
        sem_aux = output.get('sem_aux', None)
        if sem_aux is not None:
            sem_aux_h = _highest_resolution(sem_aux)
            if torch.is_tensor(sem_aux_h):
                pred_raw = sem_aux_h.argmax(1, keepdim=True)
                for cls, name in [(1, 'sacrum'), (2, 'left_hip'), (3, 'right_hip'), (4, 'fracture')]:
                    pred_c = (pred_raw == cls) & valid
                    gt_c = (sem_gt == cls) & valid
                    dice_raw, _, _, _ = _binary_metrics_from_masks(pred_c, gt_c, float(eps))
                    metrics[f'agss_val_rawsem_dice_{name}'] = dice_raw.detach().cpu().numpy()

    # Binary fracture auxiliary head
    frac_logits = output.get('frac', None)
    if frac_logits is not None:
        frac_logits = _highest_resolution(frac_logits)
        if torch.is_tensor(frac_logits):
            p_bin = torch.sigmoid(frac_logits)
            if p_bin.shape[2:] != sem_gt.shape[2:]:
                p_bin = torch.nn.functional.interpolate(p_bin, size=sem_gt.shape[2:], mode='trilinear', align_corners=False)
            pred_bin = (p_bin > 0.5) & valid
            dice_b, _, prec_b, rec_b = _binary_metrics_from_masks(pred_bin, gt_frac, float(eps))
            metrics.update({
                'agss_val_binfrac_dice': dice_b.detach().cpu().numpy(),
                'agss_val_binfrac_precision': prec_b.detach().cpu().numpy(),
                'agss_val_binfrac_recall': rec_b.detach().cpu().numpy(),
            })

    # Hierarchical anatomy assembly metrics
    region_logits = output.get('region', None)
    side_logits = output.get('side', None)
    if region_logits is not None and side_logits is not None and anat_gt is not None:
        region_logits = _highest_resolution(region_logits)
        side_logits = _highest_resolution(side_logits)
        if torch.is_tensor(region_logits) and torch.is_tensor(side_logits):
            anat_probs = assemble_anatomy_probs(region_logits, side_logits)
            pred_anat = anat_probs.argmax(1, keepdim=True)
            anat_gt = anat_gt.to(device=pred_anat.device)
            for cls, name in [(1, 'sacrum'), (2, 'left_hip'), (3, 'right_hip')]:
                pa = (pred_anat == cls) & valid
                ga = (anat_gt == cls) & valid
                dice_a, _, _, _ = _binary_metrics_from_masks(pa, ga, float(eps))
                metrics[f'agss_val_anat_dice_{name}'] = dice_a.detach().cpu().numpy()

            pelvis_gt = (anat_gt > 0) & valid
            leakage = (pred_frac & (~pelvis_gt) & valid).float().sum() / pred_frac.float().sum().clamp_min(1.0)
            metrics['agss_val_fracture_outside_pelvis_ratio'] = leakage.detach().cpu().numpy()

            # region/side/sacrum-fracture metrics
            pred_region = region_logits.argmax(1, keepdim=True)
            region_gt = region_gt.to(device=pred_region.device)
            hip_mask = (region_gt == 2) & valid
            frac_region = gt_frac
            metrics['agss_val_region_acc'] = ((pred_region == region_gt) & valid).float().sum().div(valid.float().sum().clamp_min(1.0)).detach().cpu().numpy()

            pred_side = side_logits.argmax(1, keepdim=True)
            side_gt = side_gt.to(device=pred_side.device)
            if bool(hip_mask.any().item()):
                metrics['agss_val_side_acc_on_hip'] = (pred_side[hip_mask] == side_gt[hip_mask]).float().mean().detach().cpu().numpy()

            # fracture anatomy accuracy from assembled anatomy
            frac_anat_gt = anat_gt[frac_region]
            if frac_anat_gt.numel() > 0:
                metrics['agss_val_frac_anat_acc'] = (pred_anat[frac_region] == frac_anat_gt).float().mean().detach().cpu().numpy()

            # side/confusion metrics (diagnostic only, can be skipped on fast epochs)
            if full_metrics:
                gt_left = (sem_gt == 2) & valid
                gt_right = (sem_gt == 3) & valid
                gt_sac = (sem_gt == 1) & valid
                metrics['agss_val_gt_left_to_right'] = ((pred_sem == 3) & gt_left).float().sum().div(gt_left.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_right_to_left'] = ((pred_sem == 2) & gt_right).float().sum().div(gt_right.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_left_to_frac'] = ((pred_sem == 4) & gt_left).float().sum().div(gt_left.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_right_to_frac'] = ((pred_sem == 4) & gt_right).float().sum().div(gt_right.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_sacrum_to_frac'] = ((pred_sem == 4) & gt_sac).float().sum().div(gt_sac.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_frac_to_bg'] = ((pred_sem == 0) & gt_frac).float().sum().div(gt_frac.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_frac_to_left'] = ((pred_sem == 2) & gt_frac).float().sum().div(gt_frac.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_frac_to_right'] = ((pred_sem == 3) & gt_frac).float().sum().div(gt_frac.float().sum().clamp_min(1.0)).detach().cpu().numpy()
                metrics['agss_val_gt_frac_to_sacrum'] = ((pred_sem == 1) & gt_frac).float().sum().div(gt_frac.float().sum().clamp_min(1.0)).detach().cpu().numpy()

    sacfrac_logits = output.get('sacfrac', None)
    if sacfrac_logits is not None and sacfrac_gt is not None:
        sacfrac_logits = _highest_resolution(sacfrac_logits)
        if torch.is_tensor(sacfrac_logits):
            sacfrac_gt = sacfrac_gt.to(device=sacfrac_logits.device)
            p_sac = torch.sigmoid(sacfrac_logits)
            if p_sac.shape[2:] != sacfrac_gt.shape[2:]:
                p_sac = torch.nn.functional.interpolate(p_sac, size=sacfrac_gt.shape[2:], mode='trilinear', align_corners=False)
            pred_sacfrac = (p_sac > 0.5) & valid
            gt_sacfrac = (sacfrac_gt > 0.5) & valid
            if bool(gt_sacfrac.any().item()):
                dice_sf, _, _, _ = _binary_metrics_from_masks(pred_sacfrac, gt_sacfrac, float(eps))
                metrics['agss_val_sacfrac_dice'] = dice_sf.detach().cpu().numpy()

    if full_metrics:
        struct_pred = output.get('struct', None)
        if struct_pred is not None and struct_gt is not None:
            sp_logits = _highest_resolution(struct_pred)
            if torch.is_tensor(sp_logits):
                sg = struct_gt.to(device=sp_logits.device, dtype=sp_logits.dtype).clamp(0, 1)
                if sg.shape[2:] != sp_logits.shape[2:]:
                    sg = torch.nn.functional.interpolate(sg, size=sp_logits.shape[2:], mode='trilinear', align_corners=False)
                sp = torch.sigmoid(sp_logits)
                roi = ((sg.sum(1, keepdim=True) > 0.05) | gt_frac.to(device=sp.device)).to(dtype=sp.dtype)
                roi_den = roi.sum().clamp_min(1.0)
                metrics.update({
                    'agss_val_core_mae': torch.abs(sp[:, 0:1] - sg[:, 0:1]).mean().detach().cpu().numpy(),
                    'agss_val_surface_mae': torch.abs(sp[:, 1:2] - sg[:, 1:2]).mean().detach().cpu().numpy(),
                    'agss_val_struct_roi_mae': ((torch.abs(sp - sg) * roi).sum() / (roi_den * sp.shape[1])).detach().cpu().numpy(),
                    'agss_val_core_gt_ratio': (sg[:, 0:1] > 0.05).float().mean().detach().cpu().numpy(),
                    'agss_val_surface_gt_ratio': (sg[:, 1:2] > 0.05).float().mean().detach().cpu().numpy(),
                })

        arconv_reg = output.get('arconv_reg', None)
        if torch.is_tensor(arconv_reg):
            metrics['agss_val_arconv_reg'] = arconv_reg.detach().cpu().numpy()
        for key in [
            'arconv_module_count', 'arconv_offset_abs', 'arconv_offset_max', 'arconv_offset_tv',
            'arconv_offset_sat_ratio', 'arconv_routing_entropy', 'arconv_routing_max_prob',
            'arconv_routing_collapse_ratio',
            'um_gate_mean', 'um_gate_std', 'um_gate_bg_mean', 'um_gate_fg_mean'
        ]:
            value = output.get(key, None)
            if torch.is_tensor(value):
                metrics['agss_val_' + key] = value.detach().cpu().numpy()
        if 'agss_val_um_gate_fg_mean' in metrics and 'agss_val_um_gate_bg_mean' in metrics:
            metrics['agss_val_um_gate_fg_minus_bg'] = float(metrics['agss_val_um_gate_fg_mean'] - metrics['agss_val_um_gate_bg_mean'])
        # ACFA diagnostics
        for key in ['acfa_gate_mean', 'acfa_gate_fg_mean', 'acfa_gate_bg_mean', 'acfa_res_scale']:
            value = output.get(key, None)
            if torch.is_tensor(value):
                metrics['agss_val_' + key] = value.detach().cpu().numpy()
    return metrics


class AGSSArconvLstmASFEBTrainer(nnUNetTrainer):
    agss_fracture_label = 4

    agss_sem_aux_weight = 0.0
    agss_frac_weight = 1.0
    agss_region_weight = 0.25
    agss_side_weight = 0.20
    agss_sacfrac_weight = 0.35
    agss_struct_weight = 0.20
    agss_prior_weight = 0.10
    agss_consistency_weight = 0.05
    agss_arconv_weight = 0.1   # was 1.0; reg must not dominate fracture loss
    agss_enable_auxiliary = True

    # Network feature flags
    agss_use_um_fusion = True
    agss_use_acfa = False       # ACFA is mutually exclusive with UM-Fusion
    agss_use_xlstm = True

    # Fracture-centric checkpoint selection
    agss_best_fracture_metric = 'agss_val_binfrac_dice'
    agss_best_fracture_weight = 1.00
    agss_best_sacfrac_weight = 0.25
    agss_best_outside_penalty = 0.10

    # Two-stage loss schedule
    agss_loss_schedule_start_epoch = 40
    agss_frac_weight_late = 1.20
    agss_region_weight_late = 0.10
    agss_side_weight_late = 0.10
    agss_sacfrac_weight_late = 0.45
    agss_struct_weight_late = 0.30
    agss_prior_weight_late = 0.05
    agss_consistency_weight_late = 0.02

    agss_arconv_stage_idxs = (2,)
    agss_cache_aux_in_ram = False

    # Left-right discrimination
    agss_disable_lr_mirroring = True
    agss_lr_axis = None  # if None, infer from agss_aux_report.json
    agss_use_coord_map = False
    agss_use_skip_se = True
    agss_aux_highres_only = True
    agss_use_raw_sem_aux = False
    agss_sacrum_frac_oversample_ratio = 0.0
    agss_balanced_fg_sampling = True
    agss_fg_classes = (4, 1, 2, 3)
    agss_fg_class_weights = (0.50, 0.20, 0.15, 0.15)

    agss_print_extra_val_metrics = True
    agss_val_loss_includes_auxiliary = False
    # Fast-training diagnostics schedule:
    # basic metrics every epoch, expensive raw/confusion/struct/misc every N epochs
    agss_full_val_metrics_every = 10
    agss_full_val_metrics_first_n_epochs = 0

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)
        label_manager = plans_manager.get_label_manager(dataset_json)

        if cls.agss_use_coord_map:
            num_input_channels = num_input_channels + 3

        common_kwargs = {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU,
            'nonlin_kwargs': {'inplace': True},
        }

        model = AGSSUXlstmBotArconv(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[
                min(configuration_manager.UNet_base_num_features * 2 ** i, configuration_manager.unet_max_num_features)
                for i in range(num_stages)
            ],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
            num_classes=label_manager.num_segmentation_heads,
            n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            deep_supervision=enable_deep_supervision,
            arconv_stage_idxs=cls.agss_arconv_stage_idxs,
            use_um_fusion=getattr(cls, 'agss_use_um_fusion', True),
            use_acfa=getattr(cls, 'agss_use_acfa', False),
            use_xlstm=getattr(cls, 'agss_use_xlstm', True),
            use_skip_se=cls.agss_use_skip_se,
            aux_highres_only=cls.agss_aux_highres_only,
            use_sem_aux=cls.agss_use_raw_sem_aux,
            **common_kwargs,
        )
        model.apply(InitWeights_He(1e-2))
        for module in model.modules():
            if hasattr(module, 'reset_parameters_for_stable_start'):
                module.reset_parameters_for_stable_start()
        return model

    def _get_agss_report_path(self) -> Path:
        return Path(self.preprocessed_dataset_folder) / 'agss_aux' / 'agss_aux_report.json'

    def _infer_lr_axis_from_report(self):
        if self.agss_lr_axis is not None:
            return int(self.agss_lr_axis)
        report_path = self._get_agss_report_path()
        if report_path.is_file():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                if 'inferred_lr_axis' in report:
                    return int(report['inferred_lr_axis'])
            except Exception:
                pass
        return None

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if self.agss_disable_lr_mirroring:
            lr_axis = self._infer_lr_axis_from_report()
            if lr_axis is not None and mirror_axes is not None:
                new_mirror_axes = tuple([i for i in mirror_axes if int(i) != int(lr_axis)])
                if len(new_mirror_axes) != len(mirror_axes):
                    self.print_to_log_file(
                        f'AGSS: disabling mirroring on inferred left-right axis {lr_axis}. '
                        f'Old mirror_axes={mirror_axes}, new mirror_axes={new_mirror_axes}'
                    )
                mirror_axes = new_mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def get_dataloaders(self):
        aux_folder = os.path.join(self.preprocessed_dataset_folder, 'agss_aux')
        os.environ['NNUNET_AGSS_USE_PRECOMPUTED'] = '1'
        os.environ['NNUNET_AGSS_AUX_FOLDER'] = aux_folder
        os.environ['NNUNET_AGSS_NUM_AUX_CHANNELS'] = str(AGSS_NUM_AUX_CHANNELS)
        os.environ['NNUNET_AGSS_CACHE_AUX'] = '1' if self.agss_cache_aux_in_ram else '0'
        if self.agss_balanced_fg_sampling:
            os.environ['NNUNET_AGSS_FG_CLASSES'] = ','.join(str(i) for i in self.agss_fg_classes)
            os.environ['NNUNET_AGSS_FG_WEIGHTS'] = ','.join(str(float(i)) for i in self.agss_fg_class_weights)
            os.environ.pop('NNUNET_FORCE_FG_CLASS', None)
        else:
            os.environ['NNUNET_FORCE_FG_CLASS'] = str(self.agss_fracture_label)
        os.environ['NNUNET_AGSS_COORD_MAP'] = '1' if self.agss_use_coord_map else '0'
        os.environ['NNUNET_AGSS_SACRUM_FRAC_OVERSAMPLE_RATIO'] = str(self.agss_sacrum_frac_oversample_ratio)
        return super().get_dataloaders()


    def _current_agss_weight_dict(self):
        if int(getattr(self, 'current_epoch', 0)) < int(self.agss_loss_schedule_start_epoch):
            return {
                'frac': float(self.agss_frac_weight),
                'region': float(self.agss_region_weight),
                'side': float(self.agss_side_weight),
                'sacfrac': float(self.agss_sacfrac_weight),
                'struct': float(self.agss_struct_weight),
                'prior': float(self.agss_prior_weight),
                'consistency': float(self.agss_consistency_weight),
            }
        return {
            'frac': float(self.agss_frac_weight_late),
            'region': float(self.agss_region_weight_late),
            'side': float(self.agss_side_weight_late),
            'sacfrac': float(self.agss_sacfrac_weight_late),
            'struct': float(self.agss_struct_weight_late),
            'prior': float(self.agss_prior_weight_late),
            'consistency': float(self.agss_consistency_weight_late),
        }

    def _apply_agss_loss_schedule(self):
        if not hasattr(self, 'loss') or not hasattr(self.loss, 'weights'):
            return
        wd = self._current_agss_weight_dict()
        self.loss.weights.frac = wd['frac']
        self.loss.weights.region = wd['region']
        self.loss.weights.side = wd['side']
        self.loss.weights.sacfrac = wd['sacfrac']
        self.loss.weights.struct = wd['struct']
        self.loss.weights.prior = wd['prior']
        self.loss.weights.consistency = wd['consistency']

    def _agss_fracture_selection_score(self, summaries: dict):
        frac = float(summaries.get(self.agss_best_fracture_metric, np.nan))
        sac = float(summaries.get('agss_val_sacfrac_dice', np.nan))
        outside = float(summaries.get('agss_val_fracture_outside_pelvis_ratio', np.nan))
        score = 0.0
        if not np.isnan(frac):
            score += float(self.agss_best_fracture_weight) * frac
        if not np.isnan(sac):
            score += float(self.agss_best_sacfrac_weight) * sac
        if not np.isnan(outside):
            score -= float(self.agss_best_outside_penalty) * outside
        return float(score)

    def _save_best_fracture_checkpoint(self, score: float, summaries: dict):
        if getattr(self, 'is_ddp', False) and getattr(self, 'local_rank', 0) != 0:
            return
        prev = getattr(self, '_agss_best_fracture_score', None)
        if (prev is None) or (score > prev):
            self._agss_best_fracture_score = float(score)
            self.print_to_log_file(
                f'AGSS new best fracture-centric score: {score:.4f} '
                f"(binfrac={summaries.get('agss_val_binfrac_dice', np.nan):.4f}, "
                f"sacfrac={summaries.get('agss_val_sacfrac_dice', np.nan):.4f}, "
                f"outside={summaries.get('agss_val_fracture_outside_pelvis_ratio', np.nan):.4f})"
            )
            ckpt = os.path.join(self.output_folder, 'checkpoint_best_fracture.pth')
            try:
                self.save_checkpoint(ckpt)
            except Exception:
                try:
                    torch.save({'network_weights': self.network.state_dict()}, ckpt)
                except Exception as e:
                    self.print_to_log_file(f'AGSS warning: could not save fracture-centric checkpoint: {e}')

    def _build_loss(self):
        base_loss = super()._build_loss()
        wd = self._current_agss_weight_dict()
        weights = AGSSLossWeights(
            sem_aux=self.agss_sem_aux_weight,
            frac=wd['frac'],
            region=wd['region'],
            side=wd['side'],
            sacfrac=wd['sacfrac'],
            struct=wd['struct'],
            prior=wd['prior'],
            consistency=wd['consistency'],
            arconv=self.agss_arconv_weight,
        )
        return AGSSAuxiliaryLoss(
            base_loss=base_loss,
            weights=weights,
            fracture_label=self.agss_fracture_label,
            enable_auxiliary=self.agss_enable_auxiliary,
        )

    def train_step(self, batch: dict) -> dict:
        self._apply_agss_loss_schedule()
        data = batch['data'].to(self.device, non_blocking=True)
        target_combined = _move_to_device(_reduce_agss_target_for_gpu(batch['target']), self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self.device):
            output = self.network(data, return_dict=True)
            del data
            l = self.loss(output, target_combined)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}


    def _do_full_val_metrics(self) -> bool:
        if not self.agss_print_extra_val_metrics:
            return False
        if self.current_epoch < int(self.agss_full_val_metrics_first_n_epochs):
            return True
        every = max(int(self.agss_full_val_metrics_every), 1)
        return (int(self.current_epoch) % every) == 0

    def validation_step(self, batch: dict) -> dict:
        self._apply_agss_loss_schedule()
        data = batch['data'].to(self.device, non_blocking=True)
        target_combined = _move_to_device(_reduce_agss_target_for_gpu(batch['target']), self.device)
        target = _seg_only_target(target_combined)

        with _autocast_context(self.device):
            output = self.network(data, return_dict=True)
            del data
            if self.agss_val_loss_includes_auxiliary:
                l = self.loss(output, target_combined)
            else:
                l = self.loss(output['seg'], target_combined)

        output_seg_for_dice = output['seg'] if isinstance(output, dict) else output
        if self.enable_deep_supervision:
            output_for_dice = output_seg_for_dice[0]
            target_for_dice = target[0]
        else:
            output_for_dice = output_seg_for_dice
            target_for_dice = target

        axes = [0] + list(range(2, output_for_dice.ndim))
        if self.label_manager.has_regions:
            pred_onehot = (torch.sigmoid(output_for_dice) > 0.5).long()
        else:
            output_seg = output_for_dice.argmax(1)[:, None]
            pred_onehot = torch.zeros(output_for_dice.shape, device=output_for_dice.device, dtype=torch.float16)
            pred_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_dice != self.label_manager.ignore_label).float()
                target_for_dice = target_for_dice.clone()
                target_for_dice[target_for_dice == self.label_manager.ignore_label] = 0
            else:
                if target_for_dice.dtype == torch.bool:
                    mask = ~target_for_dice[:, -1:]
                else:
                    mask = 1 - target_for_dice[:, -1:]
                target_for_dice = target_for_dice[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target_for_dice, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        ret = {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
        if self.agss_print_extra_val_metrics:
            ignore_label = self.label_manager.ignore_label if self.label_manager.has_ignore_label else None
            extra = _compute_agss_validation_metrics(
                output,
                target_combined,
                self.agss_fracture_label,
                ignore_label,
                full_metrics=self._do_full_val_metrics(),
            )
            ret.update(extra)
            _ensure_agss_val_metric_keys(ret)
        return ret

    def on_validation_epoch_end(self, val_outputs):
        if self.agss_print_extra_val_metrics:
            for i in range(len(val_outputs)):
                _ensure_agss_val_metric_keys(val_outputs[i])
        super().on_validation_epoch_end(val_outputs)
        if not self.agss_print_extra_val_metrics:
            return

        metric_names = [
            'agss_val_dice_sacrum', 'agss_val_dice_left_hip', 'agss_val_dice_right_hip', 'agss_val_dice_fracture',
            'agss_val_rawsem_dice_sacrum', 'agss_val_rawsem_dice_left_hip', 'agss_val_rawsem_dice_right_hip', 'agss_val_rawsem_dice_fracture',
            'agss_val_fracture_dice', 'agss_val_fracture_iou', 'agss_val_fracture_precision', 'agss_val_fracture_recall',
            'agss_val_pred_fracture_ratio', 'agss_val_gt_fracture_ratio',
            'agss_val_binfrac_dice', 'agss_val_binfrac_precision', 'agss_val_binfrac_recall',
            'agss_val_fracture_outside_pelvis_ratio',
            'agss_val_anat_dice_sacrum', 'agss_val_anat_dice_left_hip', 'agss_val_anat_dice_right_hip',
            'agss_val_region_acc', 'agss_val_side_acc_on_hip', 'agss_val_frac_anat_acc', 'agss_val_sacfrac_dice',
            'agss_val_gt_left_to_right', 'agss_val_gt_right_to_left', 'agss_val_gt_left_to_frac',
            'agss_val_gt_right_to_frac', 'agss_val_gt_sacrum_to_frac', 'agss_val_gt_frac_to_bg',
            'agss_val_gt_frac_to_left', 'agss_val_gt_frac_to_right', 'agss_val_gt_frac_to_sacrum',
            'agss_val_core_mae', 'agss_val_surface_mae', 'agss_val_struct_roi_mae',
            'agss_val_core_gt_ratio', 'agss_val_surface_gt_ratio',
            'agss_val_arconv_reg', 'agss_val_arconv_module_count', 'agss_val_arconv_offset_abs',
            'agss_val_arconv_offset_max', 'agss_val_arconv_offset_tv', 'agss_val_arconv_offset_sat_ratio',
            'agss_val_arconv_routing_entropy', 'agss_val_arconv_routing_max_prob', 'agss_val_arconv_routing_collapse_ratio',
            'agss_val_um_gate_mean', 'agss_val_um_gate_std', 'agss_val_um_gate_bg_mean', 'agss_val_um_gate_fg_mean', 'agss_val_um_gate_fg_minus_bg',
        ]
        summaries = {}
        full_metrics_epoch = self._do_full_val_metrics()
        for name in metric_names:
            mean_value = _safe_mean_numpy([o.get(name, None) for o in val_outputs])
            if mean_value is not None and not np.isnan(mean_value):
                summaries[name] = mean_value
        if len(summaries) == 0:
            return

        def _mk(names):
            out = []
            for name in names:
                if name in summaries:
                    out.append(f"{name.replace('agss_val_', '')}: {summaries[name]:.4f}")
            return out

        sem_line = _mk(['agss_val_dice_sacrum', 'agss_val_dice_left_hip', 'agss_val_dice_right_hip', 'agss_val_dice_fracture'])
        if sem_line:
            self.print_to_log_file('AGSS val semantic metrics - ' + ', '.join(sem_line))

        if full_metrics_epoch:
            raw_line = _mk(['agss_val_rawsem_dice_sacrum', 'agss_val_rawsem_dice_left_hip', 'agss_val_rawsem_dice_right_hip', 'agss_val_rawsem_dice_fracture'])
            if raw_line:
                self.print_to_log_file('AGSS val rawsem metrics - ' + ', '.join(raw_line))

        frac_line = _mk(['agss_val_fracture_iou', 'agss_val_fracture_precision', 'agss_val_fracture_recall', 'agss_val_pred_fracture_ratio', 'agss_val_gt_fracture_ratio', 'agss_val_binfrac_dice'])
        if frac_line:
            self.print_to_log_file('AGSS val fracture metrics - ' + ', '.join(frac_line))

        hier_line = _mk(['agss_val_fracture_outside_pelvis_ratio', 'agss_val_anat_dice_sacrum', 'agss_val_anat_dice_left_hip', 'agss_val_anat_dice_right_hip', 'agss_val_region_acc', 'agss_val_side_acc_on_hip', 'agss_val_frac_anat_acc', 'agss_val_sacfrac_dice'])
        if hier_line:
            self.print_to_log_file('AGSS val hierarchy metrics - ' + ', '.join(hier_line))

        if full_metrics_epoch:
            conf_line = _mk(['agss_val_gt_left_to_right', 'agss_val_gt_right_to_left', 'agss_val_gt_left_to_frac', 'agss_val_gt_right_to_frac', 'agss_val_gt_sacrum_to_frac', 'agss_val_gt_frac_to_bg', 'agss_val_gt_frac_to_left', 'agss_val_gt_frac_to_right', 'agss_val_gt_frac_to_sacrum'])
            if conf_line:
                self.print_to_log_file('AGSS val confusion metrics - ' + ', '.join(conf_line))

        if full_metrics_epoch:
            struct_line = _mk(['agss_val_core_mae', 'agss_val_surface_mae', 'agss_val_struct_roi_mae', 'agss_val_core_gt_ratio', 'agss_val_surface_gt_ratio'])
            if struct_line:
                self.print_to_log_file('AGSS val struct metrics - ' + ', '.join(struct_line))

        if full_metrics_epoch:
            misc_line = []
            sci_names = {'agss_val_arconv_reg', 'agss_val_arconv_offset_abs', 'agss_val_arconv_offset_max', 'agss_val_arconv_offset_tv'}
            for name in [
                'agss_val_arconv_reg', 'agss_val_arconv_module_count', 'agss_val_arconv_offset_abs',
                'agss_val_arconv_offset_max', 'agss_val_arconv_offset_tv', 'agss_val_arconv_offset_sat_ratio',
                'agss_val_arconv_routing_entropy', 'agss_val_arconv_routing_max_prob', 'agss_val_arconv_routing_collapse_ratio',
                'agss_val_um_gate_mean', 'agss_val_um_gate_std', 'agss_val_um_gate_bg_mean',
                'agss_val_um_gate_fg_mean', 'agss_val_um_gate_fg_minus_bg'
            ]:
                if name in summaries:
                    if name in sci_names:
                        misc_line.append(f"{name.replace('agss_val_', '')}: {summaries[name]:.3e}")
                    else:
                        misc_line.append(f"{name.replace('agss_val_', '')}: {summaries[name]:.6f}")
            if misc_line:
                self.print_to_log_file('AGSS val misc metrics - ' + ', '.join(misc_line))

            # ACFA diagnostics (only if ACFA is active)
            acfa_line = []
            for name in ['agss_val_acfa_gate_mean', 'agss_val_acfa_gate_fg_mean',
                         'agss_val_acfa_gate_bg_mean', 'agss_val_acfa_res_scale']:
                if name in summaries:
                    acfa_line.append(f"{name.replace('agss_val_', '')}: {summaries[name]:.6f}")
            if acfa_line:
                self.print_to_log_file('AGSS val ACFA metrics - ' + ', '.join(acfa_line))
        else:
            self.print_to_log_file(
                f'AGSS val diagnostics schedule - basic metrics only this epoch '
                f'(full diagnostics every {int(self.agss_full_val_metrics_every)} epochs)'
            )

        frac_score = self._agss_fracture_selection_score(summaries)
        if not np.isnan(frac_score):
            self._save_best_fracture_checkpoint(frac_score, summaries)


class AGSSArconvLstmASFEBTrainer_NoAux(AGSSArconvLstmASFEBTrainer):
    agss_enable_auxiliary = False
    agss_sem_aux_weight = 0.0
    agss_frac_weight = 0.0
    agss_region_weight = 0.0
    agss_side_weight = 0.0
    agss_sacfrac_weight = 0.0
    agss_struct_weight = 0.0
    agss_prior_weight = 0.0
    agss_consistency_weight = 0.0


class AGSSArconvLstmASFEBTrainer_StructOnly(AGSSArconvLstmASFEBTrainer):
    agss_sem_aux_weight = 0.0
    agss_frac_weight = 0.0
    agss_region_weight = 0.0
    agss_side_weight = 0.0
    agss_sacfrac_weight = 0.0
    agss_struct_weight = 0.20
    agss_prior_weight = 0.0
    agss_consistency_weight = 0.0


class AGSSArconvLstmASFEBTrainer_NoUMFusion(AGSSArconvLstmASFEBTrainer):
    agss_use_um_fusion = False


class AGSSArconvLstmASFEBTrainer_NoCoordMap(AGSSArconvLstmASFEBTrainer):
    agss_use_coord_map = False


class AGSSArconvLstmASFEBTrainer_WithCoordMap(AGSSArconvLstmASFEBTrainer):
    agss_use_coord_map = True


class AGSSArconvLstmASFEBTrainer_NoSkipSE(AGSSArconvLstmASFEBTrainer):
    agss_use_skip_se = False


class AGSSArconvLstmASFEBTrainerAggressive(AGSSArconvLstmASFEBTrainer):
    """Aggressive fast mode:
    - raw semantic auxiliary head completely disabled
    - full diagnostics every 10 epochs
    """
    agss_use_raw_sem_aux = False
    agss_sem_aux_weight = 0.0
    agss_full_val_metrics_every = 10
    agss_full_val_metrics_first_n_epochs = 0


class AGSSArconvLstmASFEBTrainerPrecomputed(AGSSArconvLstmASFEBTrainer):
    pass


class AGSSArconvLstmASFEBTrainerFast(AGSSArconvLstmASFEBTrainer):
    """Alias for the fast defaults used by the patched main trainer."""
    pass


class AGSSArconvLstmASFEBTrainerFracBest(AGSSArconvLstmASFEBTrainer):
    """Alias for the fracture-centric scheduled trainer."""
    pass


# =============================================================================
# ACFA Ablation Trainers
# =============================================================================

class AGSSArconvLstmASFEBTrainer_ACFA(AGSSArconvLstmASFEBTrainer):
    """Full-model ablation: replace UM-Fusion with ACFA.

    ACFA (Anatomy-Conditioned Fracture Attention) is the paper contribution:
    it uses the predicted anatomy probability map as a spatial prior to guide
    fracture feature refinement, focusing attention on bone surfaces.

    Key difference from UM-Fusion
    ------------------------------
    - UM-Fusion uses structural morphology fields (D_core, D_surface) as the
      gate condition; ACFA uses anatomy probability — a higher-level semantic
      prior that is more directly clinically interpretable.
    - ACFA has a detached anatomy gradient, so anatomy head training is
      completely decoupled from fracture attention updates.
    - ACFA starts near-identity (res_scale_raw=-4) and learns to be useful,
      making it safe to add without LR tuning.

    Ablation table position
    -----------------------
    Baseline → +Hierarchy → +ACFA (this trainer) → +Fracture-aware loss
    """
    agss_use_um_fusion = False
    agss_use_acfa = True
    agss_use_xlstm = True
    agss_arconv_stage_idxs = (2,)
    agss_arconv_weight = 0.1
    agss_use_skip_se = True
    agss_use_raw_sem_aux = False


class AGSSArconvLstmASFEBTrainer_ACFA_NoARConv(AGSSArconvLstmASFEBTrainer_ACFA):
    """ACFA only — no ARConv3D. Tests whether geometry-adaptive sampling
    is complementary to anatomy-conditioned attention."""
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0


class AGSSArconvLstmASFEBTrainer_ACFA_NoXLSTM(AGSSArconvLstmASFEBTrainer_ACFA):
    """ACFA + ARConv, no xLSTM bottleneck.
    Tests the contribution of the bidirectional LSTM context.
    If this matches ACFA performance, the xLSTM can be removed for speed."""
    agss_use_xlstm = False


class AGSSArconvLstmASFEBTrainer_ACFA_Lite(AGSSArconvLstmASFEBTrainer_ACFA):
    """ACFA + no ARConv + no xLSTM: minimal model suitable for fast ablation
    or deployment on limited hardware. Keeps all fracture-aware loss terms."""
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_xlstm = False
    agss_use_asfeb = False
    agss_full_val_metrics_every = 10


class AGSSArconvLstmASFEBTrainer_ACFA_RecallBoost(AGSSArconvLstmASFEBTrainer_ACFA):
    """ACFA + late-stage fracture recall push.

    Use this after the plain ACFA trainer converges. It follows the same
    recall schedule as LiteRecallFinal but with ACFA instead of UM-Fusion.
    """
    agss_loss_schedule_start_epoch = 0

    agss_frac_weight_late = 1.50
    agss_asl_weight_late = 0.15
    agss_asl_gamma_neg = 4.0
    agss_sacfrac_lovasz_weight_late = 0.15
    agss_small_component_weight_late = 0.50
    agss_boundary_weight_late = 0.05
    agss_sacrum_boost_weight_late = 0.25

    agss_tversky_weight_late = 0.10
    agss_tversky_alpha = 0.7
    agss_tversky_beta = 0.3

    agss_val_frac_override = True
    agss_val_frac_thresh = 0.35

    agss_fg_classes = (4, 1, 2, 3)
    agss_fg_class_weights = (0.60, 0.16, 0.12, 0.12)
    agss_sacrum_frac_oversample_ratio = 0.15


class AGSSArconvLstmASFEBTrainer_Hierarchy_Only(AGSSArconvLstmASFEBTrainer):
    """Ablation baseline: hierarchical assembly only, no ACFA and no UM-Fusion.

    Ablation table position: Baseline → +Hierarchy (this trainer)
    Compare against ACFA trainer to isolate ACFA's contribution.
    """
    agss_use_um_fusion = False
    agss_use_acfa = False
    agss_use_xlstm = True
    agss_arconv_stage_idxs = (2,)


class AGSSArconvLstmASFEBTrainer_Baseline_Flat(AGSSArconvLstmASFEBTrainer):
    """Flat 5-class segmentation baseline (no hierarchy, no ACFA, no xLSTM).

    Ablation table position: Baseline (this trainer) → +Hierarchy → +ACFA → ...
    This gives the starting Dice numbers for the paper ablation table.
    """
    agss_use_um_fusion = False
    agss_use_acfa = False
    agss_use_xlstm = False
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_enable_auxiliary = False
    agss_frac_weight = 0.0
    agss_region_weight = 0.0
    agss_side_weight = 0.0
    agss_sacfrac_weight = 0.0
    agss_struct_weight = 0.0
    agss_prior_weight = 0.0
    agss_consistency_weight = 0.0


# =============================================================================
# Lite Trainer Hierarchy  (restored from full trainer)
# =============================================================================

class AGSSArconvLstmASFEBTrainer_Lite(AGSSArconvLstmASFEBTrainer):
    """AGSS-Lite: keeps fracture-aware supervision, disables heavy backbone modules."""
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_xlstm = False
    agss_use_acfa = False
    agss_use_um_fusion = False
    agss_use_skip_se = False
    agss_use_coord_map = False
    agss_enable_auxiliary = True
    agss_aux_highres_only = True
    agss_use_raw_sem_aux = False
    agss_balanced_fg_sampling = True
    agss_fg_classes = (4, 1, 2, 3)
    agss_fg_class_weights = (0.50, 0.20, 0.15, 0.15)
    agss_cache_aux_in_ram = False
    agss_full_val_metrics_every = 50
    agss_full_val_metrics_first_n_epochs = 0


class AGSSArconvLstmASFEBTrainer_LiteNoStructHead(AGSSArconvLstmASFEBTrainer_Lite):
    """Lite without struct branch. Fastest variant."""
    agss_use_um_fusion = False
    agss_struct_weight = 0.0
    agss_struct_weight_late = 0.0
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_xlstm = False
    agss_use_acfa = False
    agss_use_skip_se = False
    agss_use_coord_map = False
    agss_cache_aux_in_ram = False
    agss_aux_highres_only = True
    agss_use_raw_sem_aux = False
    agss_sacrum_frac_oversample_ratio = 0.0
    agss_full_val_metrics_every = 50
    agss_full_val_metrics_first_n_epochs = 0


class AGSSArconvLstmASFEBTrainer_LiteStable(AGSSArconvLstmASFEBTrainer_Lite):
    """Stable late-stage AGSS-Lite. Use when resuming from a good checkpoint."""
    agss_frac_weight_late = 1.0
    agss_sacfrac_weight_late = 0.45
    agss_struct_weight_late = 0.15
    agss_boundary_weight_late = 0.10
    agss_small_component_weight_late = 0.5
    agss_sacrum_boost_weight_late = 0.5
    agss_prior_weight_late = 0.05
    agss_consistency_weight_late = 0.01
    agss_asl_weight = 0.0
    agss_asl_weight_late = 0.0
    agss_sacfrac_lovasz_weight = 0.0
    agss_sacfrac_lovasz_weight_late = 0.0


class AGSSArconvLstmASFEBTrainer_LiteRecallBoost(AGSSArconvLstmASFEBTrainer_Lite):
    """Lite + stronger fracture sampling + light ASL/Lovász in late stage."""
    agss_fg_classes = (4, 1, 2, 3)
    agss_fg_class_weights = (0.70, 0.12, 0.09, 0.09)
    agss_sacrum_frac_oversample_ratio = 0.15
    agss_asl_weight = 0.0
    agss_asl_weight_late = 0.20
    agss_asl_gamma_neg = 4.0
    agss_sacfrac_lovasz_weight = 0.0
    agss_sacfrac_lovasz_weight_late = 0.20
    agss_sacfrac_weight_late = 0.50
    agss_boundary_weight_late = 0.15
    agss_small_component_weight_late = 0.75
    agss_sacrum_boost_weight_late = 0.75
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_xlstm = False
    agss_use_acfa = False
    agss_use_um_fusion = False
    agss_use_skip_se = False
    agss_use_coord_map = False
    agss_cache_aux_in_ram = False
    agss_aux_highres_only = True
    agss_use_raw_sem_aux = False
    agss_full_val_metrics_every = 50
    agss_full_val_metrics_first_n_epochs = 0


class AGSSArconvLstmASFEBTrainer_LiteRecallBoostSafe(AGSSArconvLstmASFEBTrainer_LiteRecallBoost):
    agss_loss_schedule_start_epoch = 80
    agss_asl_weight_late = 0.05
    agss_sacfrac_lovasz_weight_late = 0.05
    agss_sacrum_frac_oversample_ratio = 0.05
    agss_fg_class_weights = (0.60, 0.16, 0.12, 0.12)
    agss_boundary_weight_late = 0.05
    agss_small_component_weight_late = 0.25
    agss_sacrum_boost_weight_late = 0.25
    agss_sacfrac_weight_late = 0.45


class AGSSArconvLstmASFEBTrainer_LiteRecallFinal(AGSSArconvLstmASFEBTrainer_LiteRecallBoostSafe):
    """Late-stage fracture recall push. Use only when resuming after epoch 400."""
    agss_loss_schedule_start_epoch = 0
    agss_frac_weight_late = 1.50
    agss_asl_weight_late = 0.15
    agss_asl_gamma_neg = 4.0
    agss_sacfrac_lovasz_weight_late = 0.15
    agss_small_component_weight_late = 0.50
    agss_boundary_weight_late = 0.05
    agss_sacrum_boost_weight_late = 0.25
    agss_val_frac_override = True
    agss_val_frac_thresh = 0.35
    agss_frac_gamma = 1.0


class AGSSArconvLstmASFEBTrainer_LiteRecallFinalGamma(AGSSArconvLstmASFEBTrainer_LiteRecallFinal):
    """LiteRecallFinal + boost fracture probability in assembly."""
    agss_frac_gamma = 0.70


class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_Tversky(AGSSArconvLstmASFEBTrainer_LiteRecallFinal):
    """LiteRecallFinal + conservative focal-Tversky fracture recall enhancement."""
    agss_loss_schedule_start_epoch = 0
    agss_frac_weight_late = 1.50
    agss_asl_weight_late = 0.15
    agss_asl_gamma_neg = 4.0
    agss_sacfrac_lovasz_weight_late = 0.15
    agss_small_component_weight_late = 0.50
    agss_boundary_weight_late = 0.05
    agss_sacrum_boost_weight_late = 0.25
    agss_tversky_weight = 0.0
    agss_tversky_weight_late = 0.15
    agss_tversky_alpha = 0.7
    agss_tversky_beta = 0.3
    agss_hnm_weight = 0.0
    agss_hnm_weight_late = 0.0
    agss_hnm_topk_ratio = 0.05
    agss_hnm_min_voxels = 128
    agss_val_frac_override = True
    agss_val_frac_thresh = 0.35
    agss_frac_gamma = 1.0


class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_TverskyHNM(AGSSArconvLstmASFEBTrainer_LiteRecallFinal_Tversky):
    """LiteRecallFinal + Tversky + light top-k hard-negative mining."""
    agss_hnm_weight_late = 0.03
    agss_hnm_topk_ratio = 0.05
    agss_hnm_min_voxels = 128


class AGSSArconvLstmASFEBTrainer_FracOptimized(AGSSArconvLstmASFEBTrainer_LiteRecallFinal_Tversky):
    """Aggressive fracture-oriented Lite trainer."""
    agss_asl_weight_late = 0.25
    agss_asl_gamma_neg = 3.0
    agss_small_component_weight_late = 0.75
    agss_sacrum_boost_weight_late = 0.50
    agss_sacfrac_lovasz_weight_late = 0.20
    agss_sacfrac_weight_late = 0.50
    agss_tversky_weight_late = 0.20
    agss_hnm_weight_late = 0.03
    agss_val_frac_thresh = 0.30
    agss_frac_gamma = 0.75
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_xlstm = False
    agss_use_acfa = False
    agss_use_um_fusion = False
    agss_use_skip_se = False


class AGSSArconvLstmASFEBTrainer_FracOptimized_ARUM(AGSSArconvLstmASFEBTrainer_FracOptimized):
    """FracOptimized + ARConv3D + UM-Fusion + ChannelSE.
    Train from scratch or use strict=False checkpoint loading."""
    agss_arconv_stage_idxs = (2,)
    agss_arconv_weight = 0.08
    agss_use_um_fusion = True
    agss_use_skip_se = True
    agss_use_acfa = False
    agss_use_xlstm = False
    agss_struct_weight = 0.15
    agss_struct_weight_late = 0.20


# =============================================================================
# LiteRecallFinal + ARConv + UM-Fusion + ChannelSE  (current best model family)
# =============================================================================

class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_ARUMSE(AGSSArconvLstmASFEBTrainer_LiteRecallFinal):
    """LiteRecallFinal backbone + ARConv3D + UM-Fusion + ChannelSE.

    This is the current best-performing configuration (Epoch-99 binfrac_dice ≈ 0.765).
    All three bug fixes are active:
      - ARConv saturation fix (d/h/w_max 1.0→0.5, sat_weight 0.25→1.0)
      - UM-Fusion gate inversion fix (refine(feat) not refine(struct_feat))
      - arconv_weight 1.0→0.1 (reg no longer dominates fracture loss)

    Can be resumed from existing checkpoint with strict=True.
    """
    # Backbone additions over Lite base
    agss_arconv_stage_idxs = (2,)
    agss_arconv_weight = 0.1       # fixed from 1.0 → 0.1
    agss_use_um_fusion = True
    agss_use_skip_se = True
    agss_use_acfa = False
    agss_use_xlstm = False         # keep off for speed
    agss_use_asfeb = False

    # Struct supervision needed by UM-Fusion
    agss_struct_weight = 0.15
    agss_struct_weight_late = 0.20

    # Keep Lite sampling / scheduling from parent
    agss_loss_schedule_start_epoch = 0
    agss_frac_weight_late = 1.50
    agss_asl_weight_late = 0.15
    agss_asl_gamma_neg = 4.0
    agss_sacfrac_lovasz_weight_late = 0.15
    agss_small_component_weight_late = 0.50
    agss_boundary_weight_late = 0.05
    agss_sacrum_boost_weight_late = 0.25
    agss_val_frac_override = True
    agss_val_frac_thresh = 0.35
    agss_frac_gamma = 1.0
    agss_fg_class_weights = (0.60, 0.16, 0.12, 0.12)
    agss_sacrum_frac_oversample_ratio = 0.05


class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_ARUMSE_Tversky(
    AGSSArconvLstmASFEBTrainer_LiteRecallFinal_ARUMSE
):
    """LiteRecallFinal_ARUMSE + Focal-Tversky for small-fragment recall."""
    agss_tversky_weight = 0.0
    agss_tversky_weight_late = 0.15
    agss_tversky_alpha = 0.7
    agss_tversky_beta = 0.3
    agss_hnm_weight = 0.0
    agss_hnm_weight_late = 0.0


class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_ARUMSE_TverskyHNM(
    AGSSArconvLstmASFEBTrainer_LiteRecallFinal_ARUMSE_Tversky
):
    """ARUMSE + Tversky + hard-negative mining. Add HNM only after Tversky is stable."""
    agss_hnm_weight_late = 0.03
    agss_hnm_topk_ratio = 0.05
    agss_hnm_min_voxels = 128


# =============================================================================
# CSM Variants (Contralateral Symmetry Module)
# =============================================================================

class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_CSM(AGSSArconvLstmASFEBTrainer_LiteRecallFinal):
    """LiteRecallFinal + Contralateral Symmetry Module.
    Train from scratch; CSM adds parameters incompatible with non-CSM checkpoints."""
    agss_use_csm = True
    agss_csm_lr_axis = 2
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_um_fusion = False
    agss_use_skip_se = False
    agss_use_acfa = False
    agss_use_xlstm = False
    agss_use_coord_map = False
    agss_use_raw_sem_aux = False
    agss_full_val_metrics_every = 10
    agss_full_val_metrics_first_n_epochs = 0


class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_CSM_ARUMSE(
    AGSSArconvLstmASFEBTrainer_LiteRecallFinal_CSM
):
    """CSM + ARConv3D + UM-Fusion + ChannelSE.
    Most complete heavy-module variant. Requires training from scratch."""
    agss_arconv_stage_idxs = (2,)
    agss_arconv_weight = 0.1
    agss_use_um_fusion = True
    agss_use_skip_se = True
    agss_use_acfa = False
    agss_use_xlstm = False
    agss_use_coord_map = False
    agss_use_raw_sem_aux = False
    agss_struct_weight = 0.15
    agss_struct_weight_late = 0.20


class AGSSArconvLstmASFEBTrainer_LiteRecallFinal_Tversky_CSM_SAFE(
    AGSSArconvLstmASFEBTrainer_LiteRecallFinal_Tversky
):
    """Tversky + CSM with conservative settings."""
    initial_lr = 3e-4
    agss_use_csm = True
    agss_csm_lr_axis = 2
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_um_fusion = False
    agss_use_skip_se = False
    agss_use_acfa = False
    agss_use_xlstm = False
    agss_use_coord_map = False
    agss_use_raw_sem_aux = False
    agss_tversky_weight_late = 0.05
    agss_asl_weight_late = 0.0
    agss_hnm_weight_late = 0.0
    agss_frac_weight_late = 1.30
    agss_boundary_weight_late = 0.03
    agss_small_component_weight_late = 0.25
    agss_sacrum_boost_weight_late = 0.0
    agss_sacfrac_lovasz_weight_late = 0.10
    agss_val_frac_override = True
    agss_val_frac_thresh = 0.35
    agss_frac_gamma = 1.0


# =============================================================================
# AGSS-Clean Trainer (paper-ready simplified model)
# =============================================================================

class AGSSArconvLstmASFEBTrainer_Clean(AGSSArconvLstmASFEBTrainer):
    """AGSS-Clean: simplified hierarchical model with 4 clean loss terms.

    Paper contributions:
    1. Hierarchical semantic assembly (region + side + frac -> 5-class)
    2. Anatomy-Conditioned Fracture Attention (ACFA)
    3. Clean architecture: standard nnU-Net backbone, no heavy modules
    4. Focal fracture loss: replaces 5+ ad-hoc loss terms with one principled term

    Loss terms:
    - L_semantic (1.0): CE+Dice on assembled 5-class output
    - L_frac (1.0): Focal BCE+Dice on binary fracture head (alpha=0.75, gamma=2.0)
    - L_anatomy (0.3->0.1): CE(region) + masked CE(side)
    - L_sacfrac (0.5): BCE+Dice on sacrum fracture head
    - L_prior (0.02): anatomy prior penalty
    """

    # Network: clean backbone plus lightweight bottleneck/skip/laterality modules
    agss_use_acfa = True
    agss_use_um_fusion = False
    agss_use_xlstm = False
    agss_arconv_stage_idxs = ()
    agss_arconv_weight = 0.0
    agss_use_skip_se = False
    agss_use_coord_map = False  # keep dataloader z/y/x coord maps off for Clean; LR coord is injected in side head.
    agss_use_raw_sem_aux = False
    agss_use_arconv_lite = True
    agss_use_skip_eca = True
    agss_use_lr_coord = True
    agss_use_sacfrac_head = True

    # Disable old AGSS auxiliary loss mechanism (we use AGSSCleanLoss instead)
    agss_enable_auxiliary = False
    agss_frac_weight = 0.0
    agss_region_weight = 0.0
    agss_side_weight = 0.0
    agss_sacfrac_weight = 0.0
    agss_struct_weight = 0.0
    agss_prior_weight = 0.0
    agss_consistency_weight = 0.0

    # Clean loss weights
    clean_semantic_weight = 1.0
    clean_frac_weight = 1.0
    clean_anatomy_weight = 0.30
    clean_anatomy_weight_late = 0.10
    clean_sacfrac_weight = 0.50
    clean_prior_weight = 0.02
    clean_focal_alpha = 0.75
    clean_focal_gamma = 2.0
    clean_semantic_dice_weight = 1.0
    clean_frac_dice_weight = 1.0
    clean_anatomy_schedule_epoch = 40

    # Predicted-structure SCI and geometry-aware fracture loss
    clean_small_component_weight = 0.25
    clean_geometry_weight = 0.50
    clean_struct_weight = 0.10
    clean_geometry_start_epoch = 20
    clean_small_component_start_epoch = 40
    clean_struct_downsample_factor = 2
    agss_use_sci = True
    agss_sci_detach_struct = True

    # Sampling
    agss_balanced_fg_sampling = True
    agss_fg_classes = (4, 1, 2, 3)
    agss_fg_class_weights = (0.50, 0.20, 0.15, 0.15)
    agss_sacrum_frac_oversample_ratio = 0.0
    # Cache precomputed AGSS aux maps to avoid repeated disk I/O for 9-channel targets.
    agss_cache_aux_in_ram = True

    # Fracture-centric checkpoint
    agss_best_fracture_metric = 'agss_val_binfrac_dice'
    agss_best_fracture_weight = 1.00
    agss_best_sacfrac_weight = 0.25
    agss_best_outside_penalty = 0.10

    # Validation
    agss_print_extra_val_metrics = True
    # Compute AGSS-Clean extra validation metrics sparsely; nnU-Net Dice still runs every epoch.
    agss_clean_val_metrics_every = 5
    agss_clean_val_metrics_first_n_epochs = 5
    agss_full_val_metrics_every = 20
    agss_full_val_metrics_first_n_epochs = 0

    def initialize(self):
        super().initialize()
        if getattr(self.device, "type", str(self.device)) == "cuda":
            try:
                self.grad_scaler = torch.amp.GradScaler(
                    "cuda", init_scale=2.0 ** 8, growth_interval=2000,
                    backoff_factor=0.5, growth_factor=2.0,
                )
            except Exception:
                self.grad_scaler = torch.cuda.amp.GradScaler(
                    init_scale=2.0 ** 8, growth_interval=2000,
                    backoff_factor=0.5, growth_factor=2.0,
                )

    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        use_sci = getattr(cls, 'agss_use_sci', True)
        use_struct_head = bool(use_sci or getattr(cls, 'clean_struct_weight', 0.0) > 0)
        model = get_agss_clean_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels,
            use_acfa=getattr(cls, 'agss_use_acfa', True),
            use_sci=use_sci,
            sci_detach_struct=getattr(cls, 'agss_sci_detach_struct', True),
            use_hierarchical_assembly=getattr(cls, 'agss_use_hierarchical_assembly', True),
            use_struct_head=use_struct_head,
            use_sacfrac_head=getattr(cls, 'agss_use_sacfrac_head', True),
            use_arconv_lite=getattr(cls, 'agss_use_arconv_lite', True),
            use_skip_eca=getattr(cls, 'agss_use_skip_eca', True),
            use_lr_coord=getattr(cls, 'agss_use_lr_coord', True),
        )
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        model.apply(InitWeights_He(1e-2))
        for module in model.modules():
            if hasattr(module, 'reset_parameters_for_stable_start'):
                module.reset_parameters_for_stable_start()
        return model

    def _build_loss(self):
        # AGSSCleanLoss implements its own single-resolution CE+Dice semantic loss.
        # Do not call nnUNetTrainer._build_loss() here because it may return a
        # DeepSupervisionWrapper, while AGSSNetClean intentionally outputs tensors.
        weights = CleanLossWeights(
            semantic=self.clean_semantic_weight,
            frac=self.clean_frac_weight,
            anatomy=self.clean_anatomy_weight,
            anatomy_late=self.clean_anatomy_weight_late,
            sacfrac=self.clean_sacfrac_weight,
            prior=self.clean_prior_weight,
            focal_alpha=self.clean_focal_alpha,
            focal_gamma=self.clean_focal_gamma,
            semantic_dice=getattr(self, "clean_semantic_dice_weight", 1.0),
            frac_dice=getattr(self, "clean_frac_dice_weight", 1.0),
            small_component=getattr(self, "clean_small_component_weight", 0.25),
            geometry=getattr(self, "clean_geometry_weight", 0.50),
            struct=getattr(self, "clean_struct_weight", 0.10),
            geometry_start_epoch=getattr(self, "clean_geometry_start_epoch", 20),
            small_component_start_epoch=getattr(self, "clean_small_component_start_epoch", 40),
            struct_downsample_factor=getattr(self, "clean_struct_downsample_factor", 2),
        )
        self._clean_loss = AGSSCleanLoss(
            weights=weights,
            fracture_label=self.agss_fracture_label,
            anatomy_schedule_start_epoch=self.clean_anatomy_schedule_epoch,
        )
        return self._clean_loss

    def _apply_agss_loss_schedule(self):
        """Set current epoch on clean loss (simple decay, no complex schedule)."""
        if hasattr(self, 'loss') and isinstance(self.loss, AGSSCleanLoss):
            self.loss.set_epoch(int(getattr(self, 'current_epoch', 0)))

    def train_step(self, batch: dict) -> dict:
        self._apply_agss_loss_schedule()
        data = batch['data'].to(self.device, non_blocking=True)
        target_combined = _move_to_device(_reduce_agss_target_for_gpu(batch['target']), self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self.device):
            # Strict SCI: do NOT pass GT struct_fields to the network.
            output = self.network(data, return_dict=True)
            del data
            l = self.loss(output, target_combined)

        if not torch.isfinite(l):
            self.print_to_log_file(
                f"[AGSS-Clean WARNING] non-finite loss at epoch {self.current_epoch}. "
                f"loss={l.detach().cpu().item() if torch.is_tensor(l) else l}. "
                f"batch_keys={batch.get('keys', None)}"
            )
            self.optimizer.zero_grad(set_to_none=True)
            return {'loss': np.array(0.0, dtype=np.float32)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    5.0,
                    error_if_nonfinite=True,
                )
            except RuntimeError as e:
                self.print_to_log_file(
                    f"[AGSS-Clean WARNING] non-finite gradient at epoch {self.current_epoch}: {e}. "
                    f"batch_keys={batch.get('keys', None)}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.update()
                return {'loss': l.detach().cpu().numpy()}
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    5.0,
                    error_if_nonfinite=True,
                )
            except RuntimeError as e:
                self.print_to_log_file(
                    f"[AGSS-Clean WARNING] non-finite gradient at epoch {self.current_epoch}: {e}. "
                    f"batch_keys={batch.get('keys', None)}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                return {'loss': l.detach().cpu().numpy()}
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def _do_clean_extra_val_metrics(self) -> bool:
        every = int(getattr(self, 'agss_clean_val_metrics_every', 1))
        first_n = int(getattr(self, 'agss_clean_val_metrics_first_n_epochs', 0))
        epoch = int(getattr(self, 'current_epoch', 0))
        if epoch < first_n:
            return True
        if every <= 0:
            return False
        return (epoch % every) == 0

    def validation_step(self, batch: dict) -> dict:
        self._apply_agss_loss_schedule()
        data = batch['data'].to(self.device, non_blocking=True)
        target_combined = _move_to_device(_reduce_agss_target_for_gpu(batch['target']), self.device)
        target = _seg_only_target(target_combined)

        with _autocast_context(self.device):
            output = self.network(data, return_dict=True)
            del data
            l = self.loss(output, target_combined)

        output_seg_for_dice = output['seg']
        target_for_dice = target[0] if isinstance(target, (list, tuple)) else target

        axes = [0] + list(range(2, output_seg_for_dice.ndim))
        output_seg = output_seg_for_dice.argmax(1)[:, None]
        pred_onehot = torch.zeros(output_seg_for_dice.shape, device=output_seg_for_dice.device, dtype=torch.float16)
        pred_onehot.scatter_(1, output_seg, 1)
        del output_seg

        if self.label_manager.has_ignore_label:
            mask = (target_for_dice != self.label_manager.ignore_label).float()
            target_for_dice = target_for_dice.clone()
            target_for_dice[target_for_dice == self.label_manager.ignore_label] = 0
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target_for_dice, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        ret = {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
        if self.agss_print_extra_val_metrics and self._do_clean_extra_val_metrics():
            ignore_label = self.label_manager.ignore_label if self.label_manager.has_ignore_label else None
            extra = _compute_agss_validation_metrics_clean(
                output, target_combined, self.agss_fracture_label, ignore_label,
                full_metrics=self._do_full_val_metrics(),
            )
            ret.update(extra)
        return ret


# Simplified validation metrics (no ARConv/confusion-matrix diagnostics)

AGSS_CLEAN_VAL_METRIC_KEYS = [
    'agss_val_dice_sacrum', 'agss_val_dice_left_hip', 'agss_val_dice_right_hip', 'agss_val_dice_fracture',
    'agss_val_fracture_dice', 'agss_val_fracture_iou', 'agss_val_fracture_precision', 'agss_val_fracture_recall',
    'agss_val_pred_fracture_ratio', 'agss_val_gt_fracture_ratio',
    'agss_val_binfrac_dice', 'agss_val_binfrac_precision', 'agss_val_binfrac_recall',
    'agss_val_fracture_outside_pelvis_ratio',
    'agss_val_anat_dice_sacrum', 'agss_val_anat_dice_left_hip', 'agss_val_anat_dice_right_hip',
    'agss_val_region_acc', 'agss_val_side_acc_on_hip', 'agss_val_frac_anat_acc', 'agss_val_sacfrac_dice',
    'agss_val_acfa_gate_mean', 'agss_val_acfa_gate_fg_mean',
    'agss_val_acfa_gate_bg_mean', 'agss_val_acfa_res_scale',
    'agss_val_sci_gate_mean', 'agss_val_sci_res_scale',
    'agss_val_arconv_lite_routing_entropy', 'agss_val_arconv_lite_routing_max_prob',
    'agss_val_arconv_lite_res_scale',
    'agss_val_core_mae', 'agss_val_surface_mae', 'agss_val_struct_roi_mae',
]


def _compute_agss_validation_metrics_clean(output, target, fracture_label=4, ignore_label=None, full_metrics=True):
    """Simplified AGSS validation: no ARConv, no raw-sem, no confusion matrix."""
    if not isinstance(output, dict):
        return {}
    seg_logits = _highest_resolution(output.get('seg'))
    if seg_logits is None or not torch.is_tensor(seg_logits):
        return {}

    sem_gt, frac_gt, anat_gt, region_gt, side_gt, sacfrac_gt, struct_gt = _split_highres_combined_target(target)
    if sem_gt is None:
        return {}
    sem_gt = sem_gt.to(device=seg_logits.device)
    valid = torch.ones_like(sem_gt, dtype=torch.bool)
    if ignore_label is not None:
        valid = sem_gt != ignore_label
    eps = seg_logits.new_tensor(1e-6)

    pred_sem = seg_logits.argmax(1, keepdim=True)
    metrics = {}

    # Main semantic metrics
    for cls, name in [(1, 'sacrum'), (2, 'left_hip'), (3, 'right_hip'), (4, 'fracture')]:
        pred_c = (pred_sem == cls) & valid
        gt_c = (sem_gt == cls) & valid
        dice, iou, _, _ = _binary_metrics_from_masks(pred_c, gt_c, float(eps))
        metrics[f'agss_val_dice_{name}'] = dice.detach().cpu().numpy()

    # Binary fracture metrics from fused output
    pred_frac = (pred_sem == int(fracture_label)) & valid
    gt_frac = (sem_gt == int(fracture_label)) & valid
    dice, iou, precision, recall = _binary_metrics_from_masks(pred_frac, gt_frac, float(eps))
    metrics.update({
        'agss_val_fracture_dice': dice.detach().cpu().numpy(),
        'agss_val_fracture_iou': iou.detach().cpu().numpy(),
        'agss_val_fracture_precision': precision.detach().cpu().numpy(),
        'agss_val_fracture_recall': recall.detach().cpu().numpy(),
        'agss_val_pred_fracture_ratio': pred_frac.float().mean().detach().cpu().numpy(),
        'agss_val_gt_fracture_ratio': gt_frac.float().mean().detach().cpu().numpy(),
    })

    # Binary fracture head metrics
    frac_logits = output.get('frac', None)
    if frac_logits is not None and torch.is_tensor(frac_logits):
            p_bin = torch.sigmoid(frac_logits)
            if p_bin.shape[2:] != sem_gt.shape[2:]:
                p_bin = torch.nn.functional.interpolate(p_bin, size=sem_gt.shape[2:], mode='trilinear', align_corners=False)
            pred_bin = (p_bin > 0.5) & valid
            dice_b, _, prec_b, rec_b = _binary_metrics_from_masks(pred_bin, gt_frac, float(eps))
            metrics.update({
                'agss_val_binfrac_dice': dice_b.detach().cpu().numpy(),
                'agss_val_binfrac_precision': prec_b.detach().cpu().numpy(),
                'agss_val_binfrac_recall': rec_b.detach().cpu().numpy(),
            })

    # Anatomy hierarchy metrics
    region_logits = output.get('region', None)
    side_logits = output.get('side', None)
    if region_logits is not None and side_logits is not None and anat_gt is not None:
        if torch.is_tensor(region_logits) and torch.is_tensor(side_logits):
            anat_probs = assemble_anatomy_probs(region_logits, side_logits)
            pred_anat = anat_probs.argmax(1, keepdim=True)
            anat_gt = anat_gt.to(device=pred_anat.device)
            for cls, name in [(1, 'sacrum'), (2, 'left_hip'), (3, 'right_hip')]:
                pa = (pred_anat == cls) & valid
                ga = (anat_gt == cls) & valid
                dice_a, _, _, _ = _binary_metrics_from_masks(pa, ga, float(eps))
                metrics[f'agss_val_anat_dice_{name}'] = dice_a.detach().cpu().numpy()

            pelvis_gt = (anat_gt > 0) & valid
            leakage = (pred_frac & (~pelvis_gt) & valid).float().sum() / pred_frac.float().sum().clamp_min(1.0)
            metrics['agss_val_fracture_outside_pelvis_ratio'] = leakage.detach().cpu().numpy()

            pred_region = region_logits.argmax(1, keepdim=True)
            region_gt_t = region_gt.to(device=pred_region.device)
            metrics['agss_val_region_acc'] = ((pred_region == region_gt_t) & valid).float().sum().div(valid.float().sum().clamp_min(1.0)).detach().cpu().numpy()

            pred_side = side_logits.argmax(1, keepdim=True)
            side_gt_t = side_gt.to(device=pred_side.device)
            hip_mask = (region_gt_t == 2) & valid
            if bool(hip_mask.any().item()):
                metrics['agss_val_side_acc_on_hip'] = (pred_side[hip_mask] == side_gt_t[hip_mask]).float().mean().detach().cpu().numpy()

            if bool(gt_frac.any().item()):
                frac_anat_gt = anat_gt[gt_frac]
                if frac_anat_gt.numel() > 0:
                    metrics['agss_val_frac_anat_acc'] = (pred_anat[gt_frac] == frac_anat_gt).float().mean().detach().cpu().numpy()

    # Sacfrac metrics
    sacfrac_logits = output.get('sacfrac', None)
    if sacfrac_logits is not None and sacfrac_gt is not None and torch.is_tensor(sacfrac_logits):
            sacfrac_gt = sacfrac_gt.to(device=sacfrac_logits.device)
            p_sac = torch.sigmoid(sacfrac_logits)
            if p_sac.shape[2:] != sacfrac_gt.shape[2:]:
                p_sac = torch.nn.functional.interpolate(p_sac, size=sacfrac_gt.shape[2:], mode='trilinear', align_corners=False)
            pred_sacfrac = (p_sac > 0.5) & valid
            gt_sacfrac = (sacfrac_gt > 0.5) & valid
            if bool(gt_sacfrac.any().item()):
                dice_sf, _, _, _ = _binary_metrics_from_masks(pred_sacfrac, gt_sacfrac, float(eps))
                metrics['agss_val_sacfrac_dice'] = dice_sf.detach().cpu().numpy()

    # ACFA diagnostics
    for key in ['acfa_gate_mean', 'acfa_gate_fg_mean', 'acfa_gate_bg_mean', 'acfa_res_scale']:
        value = output.get(key, None)
        if torch.is_tensor(value):
            metrics[f'agss_val_{key}'] = value.detach().cpu().numpy()

    # SCI diagnostics
    for key in ['sci_gate_mean', 'sci_res_scale']:
        value = output.get(key, None)
        if torch.is_tensor(value):
            metrics[f'agss_val_{key}'] = value.detach().cpu().numpy()

    # ARConv-Lite bottleneck diagnostics
    for key in ['arconv_lite_routing_entropy', 'arconv_lite_routing_max_prob', 'arconv_lite_res_scale']:
        value = output.get(key, None)
        if torch.is_tensor(value):
            metrics[f'agss_val_{key}'] = value.detach().cpu().numpy()

    # Structure field metrics (monitor D_core / D_surface prediction quality)
    if full_metrics:
        struct_pred = output.get('struct', None)
        if struct_pred is not None and struct_gt is not None and torch.is_tensor(struct_pred):
                sg = struct_gt.to(device=struct_pred.device, dtype=struct_pred.dtype).clamp(0, 1)
                if sg.shape[2:] != struct_pred.shape[2:]:
                    sg = torch.nn.functional.interpolate(sg, size=struct_pred.shape[2:], mode='trilinear', align_corners=False)
                sp = torch.sigmoid(struct_pred)
                roi = ((sg.sum(1, keepdim=True) > 0.05) | gt_frac.to(device=sp.device)).to(dtype=sp.dtype)
                roi_den = roi.sum().clamp_min(1.0)
                metrics.update({
                    'agss_val_core_mae': torch.abs(sp[:, 0:1] - sg[:, 0:1]).mean().detach().cpu().numpy(),
                    'agss_val_surface_mae': torch.abs(sp[:, 1:2] - sg[:, 1:2]).mean().detach().cpu().numpy(),
                    'agss_val_struct_roi_mae': ((torch.abs(sp - sg) * roi).sum() / (roi_den * sp.shape[1])).detach().cpu().numpy(),
                })

    return metrics


# =============================================================================
# Clean Ablation Variants
# =============================================================================



class AGSSArconvLstmASFEBTrainer_Clean_HipOnly(AGSSArconvLstmASFEBTrainer_Clean):
    """Variant for datasets with left/right hip fractures but no sacrum fractures.

    Removes the sacrum-fracture auxiliary head/loss and shifts sampling/model
    selection toward fracture and left/right hip anatomy. The main semantic
    hierarchy still keeps sacrum as an anatomy class if labels contain it.
    """
    agss_use_sacfrac_head = False
    clean_sacfrac_weight = 0.0
    agss_best_sacfrac_weight = 0.0
    agss_best_outside_penalty = 0.15
    agss_fg_classes = (4, 2, 3)
    agss_fg_class_weights = (0.60, 0.20, 0.20)
    clean_anatomy_weight = 0.35
    clean_anatomy_weight_late = 0.15
    clean_prior_weight = 0.03



class AGSSArconvLstmASFEBTrainer_Clean_HipOnly_FastLoss(AGSSArconvLstmASFEBTrainer_Clean_HipOnly):
    """Fast hip-only profile that avoids duplicate heavy loss terms.

    The final semantic output is assembled from anatomy + fracture predictions, so
    full 5-class CE+Dice, binary fracture Dice, anatomy CE, structure BCE/L1 and
    priors are partially redundant. This variant keeps the hierarchy trainable but
    makes semantic supervision CE-only, downweights/downsames structure supervision,
    and disables expensive geometry/small-component weighting for speed.
    """
    clean_semantic_weight = 0.50
    clean_semantic_dice_weight = 0.0
    clean_frac_weight = 1.00
    clean_frac_dice_weight = 0.50
    clean_anatomy_weight = 0.20
    clean_anatomy_weight_late = 0.08
    clean_struct_weight = 0.02
    clean_struct_downsample_factor = 4
    clean_geometry_weight = 0.0
    clean_small_component_weight = 0.0
    clean_prior_weight = 0.0
    agss_clean_val_metrics_every = 10

class AGSSArconvLstmASFEBTrainer_Clean_NoACFA(AGSSArconvLstmASFEBTrainer_Clean):
    """Ablation: Clean model without ACFA. Tests ACFA's contribution."""
    agss_use_acfa = False


class AGSSArconvLstmASFEBTrainer_Clean_CEFrac(AGSSArconvLstmASFEBTrainer_Clean):
    """Ablation: Clean model with standard BCE+Dice on fracture (no focal).
    Tests whether focal loss is needed."""
    clean_focal_alpha = 0.50
    clean_focal_gamma = 0.0


class AGSSArconvLstmASFEBTrainer_Clean_NoSacfrac(AGSSArconvLstmASFEBTrainer_Clean):
    """Ablation: Clean model without sacfrac head."""
    clean_sacfrac_weight = 0.0


class AGSSArconvLstmASFEBTrainer_Clean_HierarchyOnly(AGSSArconvLstmASFEBTrainer_Clean):
    """Ablation: Hierarchy only - no ACFA, no focal, no sacfrac, no prior.
    This is the minimal working model for the ablation table."""
    agss_use_acfa = False
    clean_focal_alpha = 0.50
    clean_focal_gamma = 0.0
    clean_sacfrac_weight = 0.0
    clean_prior_weight = 0.0

class AGSSArconvLstmASFEBTrainer_Clean_NoSCI(AGSSArconvLstmASFEBTrainer_Clean):
    """Ablation: disable predicted-structure SCI; structure loss and geometry-aware loss remain available."""
    agss_use_sci = False


class AGSSArconvLstmASFEBTrainer_Clean_NoStructLoss(AGSSArconvLstmASFEBTrainer_Clean):
    """Ablation: keep predicted-structure SCI but remove direct structure-field supervision."""
    clean_struct_weight = 0.0


class AGSSArconvLstmASFEBTrainer_Clean_NoGeometryWeight(AGSSArconvLstmASFEBTrainer_Clean):
    """Ablation: remove D_surface and small-component voxel weighting in fracture loss."""
    clean_geometry_weight = 0.0
    clean_small_component_weight = 0.0
