"""
Old nnU-Net v2 compatible SDAF trainer with offline precomputed auxiliary targets.

Use:
  nnUNetv2_train 102 3d_lowres 2 -tr SDAFArconvLstmASFEBTrainerPrecomputed

For ablations, you can use the subclasses at the bottom of this file.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from nnunetv2.net.YuNet.SDAFNet import SDAFUXlstmBotArconv
from nnunetv2.net.YuNet.sdaf_auxiliary import (
    DEFAULT_AFFINITY_OFFSETS,
    SDAFAuxiliaryLoss,
    SDAFLossWeights,
)


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


def _highest_resolution(x):
    """Return the highest-resolution tensor from nnU-Net deep-supervision lists."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


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


def _split_highres_combined_target(target, num_affinity_channels: int):
    """Split the combined high-resolution target into seg / struct / affinity."""
    t = _highest_resolution(target)
    if not torch.is_tensor(t):
        return None, None, None
    seg = t[:, 0:1]
    struct = t[:, 1:4] if t.shape[1] >= 4 else None
    aff_end = 4 + int(num_affinity_channels)
    affinity = t[:, 4:aff_end] if t.shape[1] >= aff_end else None
    return seg, struct, affinity


def _foreground_probability_from_logits_for_metrics(logits: torch.Tensor, fracture_start_channel: int) -> torch.Tensor:
    """
    Robust foreground/fracture probability for both binary and multi-class outputs.
    If the output channel count is smaller than fracture_start_channel, use all non-background channels.
    """
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    probs = torch.softmax(logits, dim=1)
    fs = int(fracture_start_channel)
    if logits.shape[1] > fs:
        return probs[:, fs:].sum(1, keepdim=True).clamp(0, 1)
    return probs[:, 1:].sum(1, keepdim=True).clamp(0, 1)


def _foreground_mask_from_target_for_metrics(target: torch.Tensor, logits: torch.Tensor, fracture_start_label: int, ignore_label=None):
    """Return foreground/fracture GT mask and valid mask."""
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_label is not None:
        valid = target != ignore_label
    # If the network does not have enough channels to represent labels >= fracture_start_label,
    # this is effectively a binary/region setup, so any positive label is foreground.
    if logits.shape[1] > int(fracture_start_label):
        gt = target >= int(fracture_start_label)
    else:
        gt = target > 0
    gt = gt & valid
    return gt, valid


def _compute_sdaf_validation_metrics(output, target, fracture_start_label: int, num_affinity_channels: int, ignore_label=None):
    """
    Lightweight validation metrics for checking whether SDAF heads are learning.
    These are monitoring metrics only; final paper metrics should still be computed
    on full-volume predictions after inference.
    """
    if not isinstance(output, dict):
        return {}
    seg_logits = _highest_resolution(output.get('seg'))
    if seg_logits is None or not torch.is_tensor(seg_logits):
        return {}

    seg_gt, struct_gt, aff_gt = _split_highres_combined_target(target, num_affinity_channels)
    if seg_gt is None:
        return {}
    seg_gt = seg_gt.to(device=seg_logits.device)
    gt_fg, valid = _foreground_mask_from_target_for_metrics(seg_gt, seg_logits, fracture_start_label, ignore_label)

    prob_fg = _foreground_probability_from_logits_for_metrics(seg_logits, fracture_start_label)
    pred_fg = (prob_fg > 0.5) & valid

    pred_f = pred_fg.float()
    gt_f = gt_fg.float()
    valid_f = valid.float()
    tp = (pred_f * gt_f).sum()
    fp = (pred_f * (1.0 - gt_f) * valid_f).sum()
    fn = ((1.0 - pred_f) * gt_f).sum()
    eps = seg_logits.new_tensor(1e-6)

    metrics = {
        'sdaf_val_fg_dice': ((2 * tp + eps) / (2 * tp + fp + fn + eps)).detach().cpu().numpy(),
        'sdaf_val_fg_iou': ((tp + eps) / (tp + fp + fn + eps)).detach().cpu().numpy(),
        'sdaf_val_precision': ((tp + eps) / (tp + fp + eps)).detach().cpu().numpy(),
        'sdaf_val_recall': ((tp + eps) / (tp + fn + eps)).detach().cpu().numpy(),
        'sdaf_val_pred_fg_ratio': pred_f.mean().detach().cpu().numpy(),
        'sdaf_val_gt_fg_ratio': gt_f.mean().detach().cpu().numpy(),
        'sdaf_val_prob_fg_mean': prob_fg.detach().mean().cpu().numpy(),
    }

    struct_pred = output.get('struct', None)
    if struct_pred is not None and struct_gt is not None:
        sp_logits = _highest_resolution(struct_pred)
        if torch.is_tensor(sp_logits):
            sg = struct_gt.to(device=sp_logits.device, dtype=sp_logits.dtype).clamp(0, 1)
            if sg.shape[2:] != sp_logits.shape[2:]:
                sg = torch.nn.functional.interpolate(sg, size=sp_logits.shape[2:], mode='trilinear', align_corners=False)
            sp = torch.sigmoid(sp_logits)
            # Global MAE is useful, but ROI MAE is more sensitive because background dominates 3D patches.
            roi = ((sg.sum(1, keepdim=True) > 0.05) | gt_fg.to(device=sp.device)).to(dtype=sp.dtype)
            roi_den = roi.sum().clamp_min(1.0)
            metrics.update({
                'sdaf_val_struct_core_mae': torch.abs(sp[:, 0:1] - sg[:, 0:1]).mean().detach().cpu().numpy(),
                'sdaf_val_struct_surface_mae': torch.abs(sp[:, 1:2] - sg[:, 1:2]).mean().detach().cpu().numpy(),
                'sdaf_val_struct_contact_mae': torch.abs(sp[:, 2:3] - sg[:, 2:3]).mean().detach().cpu().numpy(),
                'sdaf_val_struct_roi_mae': ((torch.abs(sp - sg) * roi).sum() / (roi_den * sp.shape[1])).detach().cpu().numpy(),
                'sdaf_val_core_gt_ratio': (sg[:, 0:1] > 0.05).float().mean().detach().cpu().numpy(),
                'sdaf_val_surface_gt_ratio': (sg[:, 1:2] > 0.05).float().mean().detach().cpu().numpy(),
                'sdaf_val_contact_gt_ratio': (sg[:, 2:3] > 0.05).float().mean().detach().cpu().numpy(),
                'sdaf_val_contact_pred_mean': sp[:, 2:3].mean().detach().cpu().numpy(),
            })

    aff_pred = output.get('affinity', None)
    if aff_pred is not None and aff_gt is not None:
        ap_logits = _highest_resolution(aff_pred)
        if torch.is_tensor(ap_logits):
            ag = aff_gt.to(device=ap_logits.device, dtype=ap_logits.dtype).clamp(0, 1)
            if ag.shape[2:] != ap_logits.shape[2:]:
                ag = torch.nn.functional.interpolate(ag, size=ap_logits.shape[2:], mode='nearest')
            ap = torch.sigmoid(ap_logits)
            gt_aff_bin = ag > 0.5
            pred_aff_bin = ap > 0.5
            # Restrict the summary to voxels that are at or near fracture/structure regions.
            if struct_gt is not None:
                sg_for_roi = struct_gt.to(device=ap_logits.device, dtype=ap_logits.dtype).clamp(0, 1)
                if sg_for_roi.shape[2:] != ap_logits.shape[2:]:
                    sg_for_roi = torch.nn.functional.interpolate(sg_for_roi, size=ap_logits.shape[2:], mode='trilinear', align_corners=False)
                aff_roi = sg_for_roi.sum(1, keepdim=True) > 0.05
            else:
                aff_roi = gt_fg.to(device=ap_logits.device)
            aff_roi = aff_roi.expand_as(ag)
            if bool(aff_roi.any().item()):
                bce = torch.nn.functional.binary_cross_entropy_with_logits(ap_logits[aff_roi], ag[aff_roi])
                acc = (pred_aff_bin[aff_roi] == gt_aff_bin[aff_roi]).float().mean()
                pos = gt_aff_bin & aff_roi
                neg = (~gt_aff_bin) & aff_roi
                if bool(pos.any().item()):
                    pos_recall = pred_aff_bin[pos].float().mean()
                else:
                    pos_recall = ap_logits.new_tensor(float('nan'))
                if bool(neg.any().item()):
                    neg_specificity = (~pred_aff_bin[neg]).float().mean()
                else:
                    neg_specificity = ap_logits.new_tensor(float('nan'))
                metrics.update({
                    'sdaf_val_aff_bce_roi': bce.detach().cpu().numpy(),
                    'sdaf_val_aff_acc_roi': acc.detach().cpu().numpy(),
                    'sdaf_val_aff_pos_recall_roi': pos_recall.detach().cpu().numpy(),
                    'sdaf_val_aff_neg_specificity_roi': neg_specificity.detach().cpu().numpy(),
                    'sdaf_val_aff_gt_pos_ratio_roi': gt_aff_bin[aff_roi].float().mean().detach().cpu().numpy(),
                    'sdaf_val_aff_pred_pos_ratio_roi': pred_aff_bin[aff_roi].float().mean().detach().cpu().numpy(),
                })

    arconv_reg = output.get('arconv_reg', None)
    if torch.is_tensor(arconv_reg):
        metrics['sdaf_val_arconv_reg'] = arconv_reg.detach().cpu().numpy()
    for key in [
        'arconv_module_count',
        'arconv_offset_abs',
        'arconv_offset_max',
        'arconv_offset_tv',
        'arconv_routing_entropy',
        'arconv_routing_max_prob',
    ]:
        value = output.get(key, None)
        if torch.is_tensor(value):
            metrics['sdaf_val_' + key] = value.detach().cpu().numpy()
    um_gate_mean = output.get('um_gate_mean', None)
    if torch.is_tensor(um_gate_mean):
        metrics['sdaf_val_um_gate_mean'] = um_gate_mean.detach().cpu().numpy()
    um_gate_std = output.get('um_gate_std', None)
    if torch.is_tensor(um_gate_std):
        metrics['sdaf_val_um_gate_std'] = um_gate_std.detach().cpu().numpy()
    return metrics


class SDAFArconvLstmASFEBTrainerPrecomputed(nnUNetTrainer):
    # Set to 1 for binary 0=background, 1=fracture datasets.
    sdaf_fracture_start_label = 4

    # Conservative first-run defaults.
    sdaf_struct_weight = 0.20
    sdaf_affinity_weight = 0.10
    sdaf_consistency_weight = 0.05
    sdaf_contact_weight = 2.00
    sdaf_arconv_weight = 1.00
    sdaf_enable_auxiliary = True
    sdaf_online_fallback = False

    # Efficiency/stability defaults.
    sdaf_arconv_stage_idxs = (2,)
    sdaf_affinity_offsets = DEFAULT_AFFINITY_OFFSETS
    sdaf_use_um_fusion = True
    sdaf_cache_aux_in_ram = False

    # Validation monitoring. Does not change training.
    # Keep val loss as base segmentation loss for compatibility with nnU-Net curves,
    # while logging auxiliary-head diagnostics separately.
    sdaf_print_extra_val_metrics = True
    sdaf_val_loss_includes_auxiliary = False

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

        common_kwargs = {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU,
            'nonlin_kwargs': {'inplace': True},
        }

        model = SDAFUXlstmBotArconv(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[
                min(
                    configuration_manager.UNet_base_num_features * 2 ** i,
                    configuration_manager.unet_max_num_features,
                )
                for i in range(num_stages)
            ],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
            num_classes=label_manager.num_segmentation_heads,
            n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            deep_supervision=enable_deep_supervision,
            arconv_stage_idxs=cls.sdaf_arconv_stage_idxs,
            affinity_offsets=cls.sdaf_affinity_offsets,
            use_um_fusion=cls.sdaf_use_um_fusion,
            fracture_start_channel=cls.sdaf_fracture_start_label,
            **common_kwargs,
        )
        model.apply(InitWeights_He(1e-2))
        for module in model.modules():
            if hasattr(module, 'reset_parameters_for_stable_start'):
                module.reset_parameters_for_stable_start()
        return model

    def get_dataloaders(self):
        aux_folder = os.path.join(self.preprocessed_dataset_folder, 'sdaf_aux')
        os.environ['NNUNET_SDAF_USE_PRECOMPUTED'] = '1'
        os.environ['NNUNET_SDAF_AUX_FOLDER'] = aux_folder
        os.environ['NNUNET_SDAF_NUM_STRUCT_CHANNELS'] = '3'
        os.environ['NNUNET_SDAF_NUM_AFFINITY_CHANNELS'] = str(len(self.sdaf_affinity_offsets))
        os.environ['NNUNET_SDAF_CACHE_AUX'] = '1' if self.sdaf_cache_aux_in_ram else '0'
        return super().get_dataloaders()

    def _build_loss(self):
        base_loss = super()._build_loss()
        weights = SDAFLossWeights(
            struct=self.sdaf_struct_weight,
            affinity=self.sdaf_affinity_weight,
            consistency=self.sdaf_consistency_weight,
            contact_weight=self.sdaf_contact_weight,
            arconv=self.sdaf_arconv_weight,
        )
        return SDAFAuxiliaryLoss(
            base_loss=base_loss,
            weights=weights,
            fracture_start_label=self.sdaf_fracture_start_label,
            affinity_offsets=self.sdaf_affinity_offsets,
            enable_auxiliary=self.sdaf_enable_auxiliary,
            online_fallback=self.sdaf_online_fallback,
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target_combined = _move_to_device(batch['target'], self.device)
        target = _seg_only_target(target_combined)

        with _autocast_context(self.device):
            # Force dict output in validation so that we can monitor struct/affinity heads.
            output = self.network(data, return_dict=True)
            del data
            if self.sdaf_val_loss_includes_auxiliary:
                l = self.loss(output, target_combined)
            else:
                # Keep the official validation loss as segmentation-only. This preserves
                # compatibility with older nnU-Net training curves and avoids mixing loss scales.
                l = self.loss(output['seg'], target_combined)

        metric_output = output
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

        ret = {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }

        if self.sdaf_print_extra_val_metrics:
            ignore_label = self.label_manager.ignore_label if self.label_manager.has_ignore_label else None
            extra = _compute_sdaf_validation_metrics(
                metric_output,
                target_combined,
                fracture_start_label=self.sdaf_fracture_start_label,
                num_affinity_channels=len(self.sdaf_affinity_offsets),
                ignore_label=ignore_label,
            )
            ret.update(extra)
        return ret

    def on_validation_epoch_end(self, val_outputs):
        # Let nnU-Net compute and log its standard pseudo Dice first.
        super().on_validation_epoch_end(val_outputs)

        if not self.sdaf_print_extra_val_metrics:
            return

        metric_names = [
            'sdaf_val_fg_dice', 'sdaf_val_fg_iou', 'sdaf_val_precision', 'sdaf_val_recall',
            'sdaf_val_pred_fg_ratio', 'sdaf_val_gt_fg_ratio', 'sdaf_val_prob_fg_mean',
            'sdaf_val_struct_core_mae', 'sdaf_val_struct_surface_mae', 'sdaf_val_struct_contact_mae',
            'sdaf_val_struct_roi_mae', 'sdaf_val_core_gt_ratio', 'sdaf_val_surface_gt_ratio',
            'sdaf_val_contact_gt_ratio', 'sdaf_val_contact_pred_mean',
            'sdaf_val_aff_bce_roi', 'sdaf_val_aff_acc_roi', 'sdaf_val_aff_pos_recall_roi',
            'sdaf_val_aff_neg_specificity_roi', 'sdaf_val_aff_gt_pos_ratio_roi',
            'sdaf_val_aff_pred_pos_ratio_roi', 'sdaf_val_arconv_reg',
            'sdaf_val_arconv_module_count', 'sdaf_val_arconv_offset_abs', 'sdaf_val_arconv_offset_max',
            'sdaf_val_arconv_offset_tv', 'sdaf_val_arconv_routing_entropy', 'sdaf_val_arconv_routing_max_prob',
            'sdaf_val_um_gate_mean', 'sdaf_val_um_gate_std',
        ]
        summaries = {}
        for name in metric_names:
            mean_value = _safe_mean_numpy([o.get(name, None) for o in val_outputs])
            if mean_value is not None and not np.isnan(mean_value):
                summaries[name] = mean_value

        if len(summaries) == 0:
            return

        # Print compact groups. These metrics are monitoring diagnostics; final paper metrics
        # should still be computed on full-volume predictions after inference.
        seg_line = []
        for name in ['sdaf_val_fg_dice', 'sdaf_val_fg_iou', 'sdaf_val_precision', 'sdaf_val_recall', 'sdaf_val_pred_fg_ratio', 'sdaf_val_gt_fg_ratio']:
            if name in summaries:
                seg_line.append(f"{name.replace('sdaf_val_', '')}: {summaries[name]:.4f}")
        if len(seg_line) > 0:
            self.print_to_log_file('SDAF val seg metrics - ' + ', '.join(seg_line))

        struct_line = []
        for name in ['sdaf_val_struct_core_mae', 'sdaf_val_struct_surface_mae', 'sdaf_val_struct_contact_mae', 'sdaf_val_struct_roi_mae', 'sdaf_val_contact_gt_ratio', 'sdaf_val_contact_pred_mean']:
            if name in summaries:
                struct_line.append(f"{name.replace('sdaf_val_', '')}: {summaries[name]:.4f}")
        if len(struct_line) > 0:
            self.print_to_log_file('SDAF val struct metrics - ' + ', '.join(struct_line))

        aff_line = []
        for name in ['sdaf_val_aff_bce_roi', 'sdaf_val_aff_acc_roi', 'sdaf_val_aff_pos_recall_roi', 'sdaf_val_aff_neg_specificity_roi', 'sdaf_val_aff_gt_pos_ratio_roi', 'sdaf_val_aff_pred_pos_ratio_roi']:
            if name in summaries:
                aff_line.append(f"{name.replace('sdaf_val_', '')}: {summaries[name]:.4f}")
        if len(aff_line) > 0:
            self.print_to_log_file('SDAF val affinity metrics - ' + ', '.join(aff_line))

        misc_line = []
        sci_names = {
            'sdaf_val_arconv_reg',
            'sdaf_val_arconv_offset_abs',
            'sdaf_val_arconv_offset_max',
            'sdaf_val_arconv_offset_tv',
        }
        for name in [
            'sdaf_val_arconv_reg', 'sdaf_val_arconv_module_count',
            'sdaf_val_arconv_offset_abs', 'sdaf_val_arconv_offset_max', 'sdaf_val_arconv_offset_tv',
            'sdaf_val_arconv_routing_entropy', 'sdaf_val_arconv_routing_max_prob',
            'sdaf_val_um_gate_mean', 'sdaf_val_um_gate_std',
        ]:
            if name in summaries:
                if name in sci_names:
                    misc_line.append(f"{name.replace('sdaf_val_', '')}: {summaries[name]:.3e}")
                else:
                    misc_line.append(f"{name.replace('sdaf_val_', '')}: {summaries[name]:.6f}")
        if len(misc_line) > 0:
            self.print_to_log_file('SDAF val misc metrics - ' + ', '.join(misc_line))



# Backward-compatible alias for your previous command name if needed.
class SDAFArconvLstmASFEBTrainerPrecomputedOld(SDAFArconvLstmASFEBTrainerPrecomputed):
    pass


# Ablation trainers for experiments.
class SDAFArconvLstmASFEBTrainerPrecomputed_NoAux(SDAFArconvLstmASFEBTrainerPrecomputed):
    sdaf_enable_auxiliary = False
    sdaf_struct_weight = 0.0
    sdaf_affinity_weight = 0.0
    sdaf_consistency_weight = 0.0


class SDAFArconvLstmASFEBTrainerPrecomputed_StructOnly(SDAFArconvLstmASFEBTrainerPrecomputed):
    sdaf_struct_weight = 0.20
    sdaf_affinity_weight = 0.0
    sdaf_consistency_weight = 0.0


class SDAFArconvLstmASFEBTrainerPrecomputed_StructAffinity(SDAFArconvLstmASFEBTrainerPrecomputed):
    sdaf_struct_weight = 0.20
    sdaf_affinity_weight = 0.10
    sdaf_consistency_weight = 0.0


class SDAFArconvLstmASFEBTrainerPrecomputed_NoUMFusion(SDAFArconvLstmASFEBTrainerPrecomputed):
    sdaf_use_um_fusion = False
