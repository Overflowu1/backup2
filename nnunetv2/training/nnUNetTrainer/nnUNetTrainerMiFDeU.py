from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.net.MiFDeU.loss_functions.compound_bti_loss import DC_and_CE_and_BTI_Loss
from torch import autocast, nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from nnunetv2.net.MiFDeU.network_architecture.MiFDeU import MiFDeU
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class nnUNetTrainerMiFDeU(nnUNetTrainer):
    #
    # def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
    #              device: torch.device = torch.device('cuda')):
    #     super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        random_int = np.random.randint(0, 2)
        view_list = [0, random_int]
        view_list = [torch.tensor(x, device='cuda') for x in view_list]
        view_list = [int(tensor.item()) for tensor in view_list]

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = 'MiFDeU'  # configuration_manager.UNet_class_name
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet,
            'MiFDeU': MiFDeU
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'MiFDeU': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_blocks_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_blocks_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }

        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            patch_size=configuration_manager.patch_size,
            n_stages=num_stages,
            # features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
            #                         configuration_manager.unet_max_num_features) for i in range(num_stages)],
            features_per_stage=24,
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))

        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)

        return model

    def make_tensors(self, lists, device):
        if not lists:
            return lists
        elif isinstance(lists[0], list):
            return [self.make_tensors(sublist, device) for sublist in lists]
        else:
            return torch.tensor(lists).to(device)
    #def _build_loss(self, epoch):
    def _build_loss(self):

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        connectivity = 26
        lambda_ti = 1e-6
        inclusion_list = []

        # exclusion_list = [[1,[2,3]],[[1],[2],[3]]]
        # exclusion_list = [[1, 2, 3, 4], [[1, 2, 3], [4]], [[1, 2], [3]]]
        exclusion_list = [[1, 2, 3], [[1, 2], [3]],]
        # exclusion_list = [
        #     # 骶骨（1-10） vs 左髋骨（11-20）
        #     (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20),
        #     (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20),
        #     (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20),
        #     (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20),
        #     (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20),
        #     (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20),
        #     (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (7, 20),
        #     (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20),
        #     (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20),
        #     (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20),
        #
        #     # 骶骨（1-10） vs 右髋骨（21-30）
        #     (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30),
        #     (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (2, 26), (2, 27), (2, 28), (2, 29), (2, 30),
        #     (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (3, 27), (3, 28), (3, 29), (3, 30),
        #     (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28), (4, 29), (4, 30),
        #     (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30),
        #     (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (6, 30),
        #     (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (7, 30),
        #     (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (8, 30),
        #     (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30),
        #     (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30),
        #
        #     # 左髋骨（11-20） vs 右髋骨（21-30）
        #     (11, 21), (11, 22), (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (11, 29), (11, 30),
        #     (12, 21), (12, 22), (12, 23), (12, 24), (12, 25), (12, 26), (12, 27), (12, 28), (12, 29), (12, 30),
        #     (13, 21), (13, 22), (13, 23), (13, 24), (13, 25), (13, 26), (13, 27), (13, 28), (13, 29), (13, 30),
        #     (14, 21), (14, 22), (14, 23), (14, 24), (14, 25), (14, 26), (14, 27), (14, 28), (14, 29), (14, 30),
        #     (15, 21), (15, 22), (15, 23), (15, 24), (15, 25), (15, 26), (15, 27), (15, 28), (15, 29), (15, 30),
        #     (16, 21), (16, 22), (16, 23), (16, 24), (16, 25), (16, 26), (16, 27), (16, 28), (16, 29), (16, 30),
        #     (17, 21), (17, 22), (17, 23), (17, 24), (17, 25), (17, 26), (17, 27), (17, 28), (17, 29), (17, 30),
        #     (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 26), (18, 27), (18, 28), (18, 29), (18, 30),
        #     (19, 21), (19, 22), (19, 23), (19, 24), (19, 25), (19, 26), (19, 27), (19, 28), (19, 29), (19, 30),
        #     (20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27), (20, 28), (20, 29), (20, 30),
        # ]
        # exclusion_list = [[1, 2, 3, 4], [[1, 3, 4], [2]],[[1,4],[3]]]
        # exclusion_list = [[1, 2, 3, 4], [[1, 3, 4], [2]], [[1, 3], [4]]]
        # exclusion_list = [[1, 2, 3], [[1], [2,3]]]
        # exclusion_list = [[1, 2, 3], [[1,3], [2]]]
        # exclusion_list = self.generate_combinations(max(self.dataset_json["labels"].values()))  # Generate all pairwise combinations for all foreground classes
        # exclusion_list = [[1, 2, 3, 4], [[1, 2, 3], [4]], [[2, 3], [1]]]

        inclusion_list = self.make_tensors(inclusion_list, self.device)
        exclusion_list = self.make_tensors(exclusion_list, self.device)

        loss = DC_and_CE_and_BTI_Loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            {'dim': dim, 'connectivity': connectivity, 'inclusion': inclusion_list, 'exclusion': exclusion_list,
             'min_thick': 1},
            weight_ce=1, weight_dice=1, weight_ti=lambda_ti, ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            self.print_to_log_file("dim: %s" % str(dim))
            self.print_to_log_file("connectivity: %s" % str(connectivity))
            self.print_to_log_file("lambda_ti: %s" % str(lambda_ti))
            self.print_to_log_file("inclusion_list: %s" % str(inclusion_list))
            self.print_to_log_file("exclusion_list_len: %s" %
                                   str(len(exclusion_list)))
            self.print_to_log_file("exclusion_list: %s" % str(exclusion_list))

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
