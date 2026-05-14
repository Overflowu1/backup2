from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

# from build.lib.nnunetv2.training.loss.compound_losses import DC_and_CE_loss


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()

        return -dc


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)
    # return torch.softmax(x,dim=1)[:,1,:,:].unsqueeze(1)


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

#
# def my_get_dice_loss(preds: torch.tensor, target: torch.tensor):
#     weight = 1
#     #print("preds",preds.shape)
#     #print("target" , target.shape)
#     pred = torch.softmax(preds, dim=1)
#     #print("pred",pred.shape)
#     pred2 = torch.log(preds)
#     # loss = DC_and_CE_loss({'smooth': 1e-5, 'do_bg': False, 'ddp': None}, {}, weight_ce=1, weight_dice=1,
#     #                       ignore_label=None,dice_class=MemoryEfficientSoftDiceLoss)
#     # loss2 = loss(net_output=pred, target=target)
#     log_softmax_pred = nn.LogSoftmax(dim=1)(preds)
#     target_one_hot = target.squeeze(1).to(torch.long)
#     # ce_loss = F.cross_entropy(pred, target_one_hot,reduction='mean')
#     ce_loss = RobustCrossEntropyLoss(reduction='mean')(pred, target[:, 0].long())
#
#     loss_fn = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1)
#     dc_loss = loss_fn(pred, target)
#     result = weight * dc_loss
#     return result


def my_get_dice_loss(preds: torch.tensor, target: torch.tensor):
    pred=torch.softmax(preds,dim=1)[:,1,:,:,:].unsqueeze(1)
    #print(pred)
    #print(target.shape)
    inter = (pred * target)
    union = (pred + target)
    # pred:BCHW, target: BCHW
    if (len(pred.shape) == 5) and (len(target.shape) == 5):
        # inter = inter.sum(dim=1).sum(dim=1).sum(dim=1).sum(dim=1)
        # union = union.sum(dim=1).sum(dim=1).sum(dim=1).sum(dim=1)
        inter = inter.sum(dim=2).sum(dim=2).sum(dim=2)
        union = union.sum(dim=2).sum(dim=2).sum(dim=2)
    dice_loss = 1 - 2 * (inter + 1) / (union + 2)
    # dice_loss = DiceCELoss(
    #     to_onehot_y=False, softmax=False, squared_pred=False, smooth_nr=1e-5, smooth_dr=1e-5
    # )
    # result = dice_loss(pred, target)
    return dice_loss