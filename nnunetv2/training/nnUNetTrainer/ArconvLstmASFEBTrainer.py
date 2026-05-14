
"""
AGSS trainer v3.1 hybrid.

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

from nnunetv2.net.YuNet.AGSSNet import AGSSUXlstmBotArconv, assemble_anatomy_probs
from nnunetv2.net.YuNet.agss_auxiliary import AGSSAuxiliaryLoss, AGSSLossWeights, AGSS_NUM_AUX_CHANNELS


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
            vals.append(float(np.asarray(v)))
        except Exception:
            pass
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _split_highres_combined_target(target):
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
        struct = t[:, 6:8]
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


def _compute_agss_validation_metrics(output, target, fracture_label: int = 4, ignore_label=None):
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

    # Raw semantic auxiliary head metrics (to reveal flat-head conflict)
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

            # side confusion metrics
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
    return metrics


class AGSSArconvLstmASFEBTrainer(nnUNetTrainer):
    agss_fracture_label = 4

    agss_sem_aux_weight = 0.15
    agss_frac_weight = 1.0
    agss_region_weight = 0.25
    agss_side_weight = 0.20
    agss_sacfrac_weight = 0.35
    agss_struct_weight = 0.20
    agss_prior_weight = 0.10
    agss_consistency_weight = 0.05
    agss_arconv_weight = 1.0
    agss_enable_auxiliary = True

    agss_arconv_stage_idxs = (2,)
    agss_use_um_fusion = True
    agss_cache_aux_in_ram = False

    # Left-right discrimination
    agss_disable_lr_mirroring = True
    agss_lr_axis = None  # if None, infer from agss_aux_report.json
    agss_use_coord_map = False
    agss_use_skip_se = True
    agss_aux_highres_only = True
    agss_sacrum_frac_oversample_ratio = 0.0
    agss_balanced_fg_sampling = True
    agss_fg_classes = (4, 1, 2, 3)
    agss_fg_class_weights = (0.50, 0.20, 0.15, 0.15)

    agss_print_extra_val_metrics = True
    agss_val_loss_includes_auxiliary = False

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
            use_um_fusion=cls.agss_use_um_fusion,
            use_skip_se=cls.agss_use_skip_se,
            aux_highres_only=cls.agss_aux_highres_only,
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

    def _build_loss(self):
        base_loss = super()._build_loss()
        weights = AGSSLossWeights(
            sem_aux=self.agss_sem_aux_weight,
            frac=self.agss_frac_weight,
            region=self.agss_region_weight,
            side=self.agss_side_weight,
            sacfrac=self.agss_sacfrac_weight,
            struct=self.agss_struct_weight,
            prior=self.agss_prior_weight,
            consistency=self.agss_consistency_weight,
            arconv=self.agss_arconv_weight,
        )
        return AGSSAuxiliaryLoss(
            base_loss=base_loss,
            weights=weights,
            fracture_label=self.agss_fracture_label,
            enable_auxiliary=self.agss_enable_auxiliary,
        )

    def train_step(self, batch: dict) -> dict:
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

    def validation_step(self, batch: dict) -> dict:
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
            extra = _compute_agss_validation_metrics(output, target_combined, self.agss_fracture_label, ignore_label)
            ret.update(extra)
        return ret

    def on_validation_epoch_end(self, val_outputs):
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

        raw_line = _mk(['agss_val_rawsem_dice_sacrum', 'agss_val_rawsem_dice_left_hip', 'agss_val_rawsem_dice_right_hip', 'agss_val_rawsem_dice_fracture'])
        if raw_line:
            self.print_to_log_file('AGSS val rawsem metrics - ' + ', '.join(raw_line))

        frac_line = _mk(['agss_val_fracture_iou', 'agss_val_fracture_precision', 'agss_val_fracture_recall', 'agss_val_pred_fracture_ratio', 'agss_val_gt_fracture_ratio', 'agss_val_binfrac_dice'])
        if frac_line:
            self.print_to_log_file('AGSS val fracture metrics - ' + ', '.join(frac_line))

        hier_line = _mk(['agss_val_fracture_outside_pelvis_ratio', 'agss_val_anat_dice_sacrum', 'agss_val_anat_dice_left_hip', 'agss_val_anat_dice_right_hip', 'agss_val_region_acc', 'agss_val_side_acc_on_hip', 'agss_val_frac_anat_acc', 'agss_val_sacfrac_dice'])
        if hier_line:
            self.print_to_log_file('AGSS val hierarchy metrics - ' + ', '.join(hier_line))

        conf_line = _mk(['agss_val_gt_left_to_right', 'agss_val_gt_right_to_left', 'agss_val_gt_left_to_frac', 'agss_val_gt_right_to_frac', 'agss_val_gt_sacrum_to_frac', 'agss_val_gt_frac_to_bg', 'agss_val_gt_frac_to_left', 'agss_val_gt_frac_to_right', 'agss_val_gt_frac_to_sacrum'])
        if conf_line:
            self.print_to_log_file('AGSS val confusion metrics - ' + ', '.join(conf_line))

        struct_line = _mk(['agss_val_core_mae', 'agss_val_surface_mae', 'agss_val_struct_roi_mae', 'agss_val_core_gt_ratio', 'agss_val_surface_gt_ratio'])
        if struct_line:
            self.print_to_log_file('AGSS val struct metrics - ' + ', '.join(struct_line))

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


class AGSSArconvLstmASFEBTrainerPrecomputed(AGSSArconvLstmASFEBTrainer):
    pass
