import torch
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.net.MiFDeU.loss_functions.Bouloss import BoundaryDoULoss
from nnunetv2.net.MiFDeU.loss_functions.bti_loss import BTI_Loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import nibabel as nib
import torch.nn.functional as F


def load_nifti_as_tensor(file_path):
    nii_data = nib.load(file_path)
    array_data = nii_data.get_fdata()
    tensor_data = torch.tensor(array_data, dtype=torch.float32)
    tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
    return tensor_data


class DC_and_CE_and_BTI_Loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, ti_kwargs,weight_ce=1, weight_dice=1, weight_ti=1e-6,
                 ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_BTI_Loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        # self.epoch=None
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_ti = weight_ti
        # self.morph_weight = morph_weight
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ti = BTI_Loss(**ti_kwargs)
        self.bou = BoundaryDoULoss(n_classes=4)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        #
        # predicted_total = torch.sum(net_output, dim=1, keepdim=True)
        # predicted_total = torch.clamp(predicted_total, 0, 1)
        # normal_pelvis_template = load_nifti_as_tensor('/home/ps/wyc/average_template4.nii.gz')
        #
        # # Move normal_pelvis_template to the same device as predicted_total
        # normal_pelvis_template = normal_pelvis_template.to(predicted_total.device)
        #
        # # Resize normal_pelvis_template to match predicted_total size
        # normal_pelvis_template = F.interpolate(normal_pelvis_template, size=predicted_total.shape[2:], mode='trilinear',
        #                                        align_corners=False)

        # Calculate morph loss, differentiating between background and fracture regions
        # morph_loss_background = torch.mean((predicted_total * (1 - target) - normal_pelvis_template) ** 2)
        # morph_loss_fracture = torch.mean((predicted_total * target - normal_pelvis_template) ** 2)
        # morph_loss = morph_loss_background + self.morph_weight * morph_loss_fracture

        ti_loss = self.ti(net_output, target) if self.weight_ti != 0 else 0
        bou_loss = self.bou(net_output, target)
        result = self.weight_ce  * ce_loss +bou_loss *self.weight_ce*0.5+self.weight_ti*ti_loss
        return result
