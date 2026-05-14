"""
Old nnU-Net v2 compatible trainer for SDAF-AGDSNet with offline precomputed
auxiliary targets.

Important:
1) Copy the patched nnunetv2/training/dataloading/data_loader_3d.py in this
   patch to replace your old data_loader_3d.py. Back up the original first.
2) Run tools/precompute_sdaf_auxiliary_old.py before training.
3) Use this trainer with, for example:
      nnUNetv2_train 102 3d_lowres 2 -tr SDAFArconvLstmASFEBTrainerPrecomputedOld
"""
from __future__ import annotations

import os
from contextlib import contextmanager

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
    """Strip SDAF auxiliary channels from nnU-Net target tensor/list."""
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


class SDAFArconvLstmASFEBTrainerPrecomputedOld(nnUNetTrainer):
    # Adjust this if your labels are binary: set to 1 for 0=background, 1=fracture.
    sdaf_fracture_start_label = 1

    # Conservative defaults for first run.
    sdaf_struct_weight = 0.20
    sdaf_affinity_weight = 0.10
    sdaf_consistency_weight = 0.05
    sdaf_contact_weight = 2.00
    sdaf_enable_auxiliary = True
    sdaf_online_fallback = False  # keep False for speed; use offline aux channels.

    # Only use ARConv at stage 2 by default to avoid the slow high-resolution stage.
    sdaf_arconv_stage_idxs = (2,)
    sdaf_affinity_offsets = DEFAULT_AFFINITY_OFFSETS

    @staticmethod
    def build_network_architecture(
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
            arconv_stage_idxs=SDAFArconvLstmASFEBTrainerPrecomputedOld.sdaf_arconv_stage_idxs,
            affinity_offsets=SDAFArconvLstmASFEBTrainerPrecomputedOld.sdaf_affinity_offsets,
            **common_kwargs,
        )
        model.apply(InitWeights_He(1e-2))
        for module in model.modules():
            if hasattr(module, 'reset_parameters_for_stable_start'):
                module.reset_parameters_for_stable_start()
        return model

    def get_dataloaders(self):
        """
        Reuse your old nnU-Net get_dataloaders, but enable aux-channel appending
        in the patched data_loader_3d.py through environment variables.
        """
        aux_folder = os.path.join(self.preprocessed_dataset_folder, 'sdaf_aux')
        os.environ['NNUNET_SDAF_USE_PRECOMPUTED'] = '1'
        os.environ['NNUNET_SDAF_AUX_FOLDER'] = aux_folder
        os.environ['NNUNET_SDAF_NUM_STRUCT_CHANNELS'] = '3'
        os.environ['NNUNET_SDAF_NUM_AFFINITY_CHANNELS'] = str(len(self.sdaf_affinity_offsets))
        return super().get_dataloaders()

    def _build_loss(self):
        base_loss = super()._build_loss()
        weights = SDAFLossWeights(
            struct=self.sdaf_struct_weight,
            affinity=self.sdaf_affinity_weight,
            consistency=self.sdaf_consistency_weight,
            contact_weight=self.sdaf_contact_weight,
        )
        sdaf_loss = SDAFAuxiliaryLoss(
            base_loss=base_loss,
            weights=weights,
            fracture_start_label=self.sdaf_fracture_start_label,
            affinity_offsets=self.sdaf_affinity_offsets,
            enable_auxiliary=self.sdaf_enable_auxiliary,
            online_fallback=self.sdaf_online_fallback,
        )

        def loss_with_sdaf_and_arconv_reg(output, target):
            loss = sdaf_loss(output, target)
            network = self.network.module if hasattr(self.network, 'module') else self.network
            if hasattr(network, 'get_arconv_regularization_loss'):
                reg = network.get_arconv_regularization_loss()
                if torch.is_tensor(reg):
                    loss = loss + reg.to(device=loss.device, dtype=loss.dtype)
            return loss

        return loss_with_sdaf_and_arconv_reg

    def validation_step(self, batch: dict) -> dict:
        """
        Old nnU-Net validation_step expects target to be a normal segmentation.
        Because our patched dataloader appends SDAF aux channels to target, we
        strip them here before computing pseudo Dice.
        """
        data = batch['data'].to(self.device, non_blocking=True)
        target_combined = _move_to_device(batch['target'], self.device)
        target = _seg_only_target(target_combined)

        with _autocast_context(self.device):
            output = self.network(data)
            del data
            l = self.loss(output, target_combined)

        if isinstance(output, dict):
            output = output['seg']
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            pred_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            pred_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            pred_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target = target.clone()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            # drop background channel
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }
