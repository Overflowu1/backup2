"""
Custom nnU-Net trainer for SDAF-AGDSNet.

Run with:
    nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr SDAFArconvLstmASFEBTrainer

This trainer builds SDAFUXlstmBotArconv and wraps nnU-Net's original loss with:
    - continuous structural field supervision: core / surface / contact
    - local affinity supervision: same-fragment connectivity
    - structural consistency loss
    - ARConv offset regularization
"""
from __future__ import annotations

import torch
from torch import nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from nnunetv2.net.YuNet.SDAFNet import SDAFUXlstmBotArconv
from nnunetv2.net.YuNet.sdaf_auxiliary import SDAFAuxiliaryLoss, SDAFLossWeights


class SDAFArconvLstmASFEBTrainer(nnUNetTrainer):
    # Change these values here first if your labels use a different convention.
    sdaf_fracture_start_label: int = 4
    sdaf_struct_weight: float = 0.0
    sdaf_affinity_weight: float = 0.0
    sdaf_consistency_weight: float = 0.0
    sdaf_contact_weight: float = 2.00
    sdaf_surface_sigma: float = 2.0
    sdaf_contact_sigma: float = 2.0
    sdaf_contact_window: int = 7
    sdaf_enable_auxiliary: bool = False

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

        norm_op = get_matching_instancenorm(conv_op)
        common_kwargs = {
            "conv_bias": True,
            "norm_op": norm_op,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        }

        conv_or_blocks_per_stage = {
            "n_conv_per_stage": configuration_manager.n_conv_per_stage_encoder,
            "n_conv_per_stage_decoder": configuration_manager.n_conv_per_stage_decoder,
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
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **common_kwargs,
        )
        model.apply(InitWeights_He(1e-2))

        # Restore stable ARConv initialization after global He init.
        for module in model.modules():
            if hasattr(module, "reset_parameters_for_stable_start"):
                module.reset_parameters_for_stable_start()

        return model

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
            surface_sigma=self.sdaf_surface_sigma,
            contact_window=self.sdaf_contact_window,
            contact_sigma=self.sdaf_contact_sigma,
            enable_auxiliary=self.sdaf_enable_auxiliary,
        )

        def loss_with_sdaf_and_arconv_reg(output, target):
            loss = sdaf_loss(output, target)
            network = self.network.module if hasattr(self.network, "module") else self.network
            if hasattr(network, "get_arconv_regularization_loss"):
                reg = network.get_arconv_regularization_loss()
                if torch.is_tensor(reg):
                    loss = loss + reg.to(device=loss.device, dtype=loss.dtype)
            return loss

        return loss_with_sdaf_and_arconv_reg
