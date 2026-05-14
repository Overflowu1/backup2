"""SwinUNETR with cross attention."""
from typing import MutableMapping, Sequence, Tuple, Union

import torch

from monai.networks import blocks
from monai.networks.nets import swin_unetr
from torch import nn
from torch.nn import KLDivLoss
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from nnunetv2.training.nnUNetTrainer.SwinMM.models import cross_attention

__all__ = [
    "SwinUNETR",
]

from nnunetv2.training.nnUNetTrainer.SwinMM.utils import view_ops
from nnunetv2.training.nnUNetTrainer.SwinMM.utils.view_ops import get_permute_transform

FeaturesDictType = MutableMapping[str, torch.Tensor]


class SwinUNETR(swin_unetr.SwinUNETR):
    """SwinUNETR with cross attention."""

    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            *args,
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 24,
            norm_name: Union[Tuple, str] = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            spatial_dims: int = 3,
            fusion_depths: Sequence[int] = (2, 2, 2, 2, 2, 2),
            cross_attention_in_origin_view: bool = False,
            **kwargs,
    ) -> None:
        """
        Args:
            fusion_depths: TODO(yiqing).
            cross_attention_in_origin_view: A bool indicates whether compute cross attention in origin view.
                If not, compute cross attention in the view of the first input.

        """
        super().__init__(img_size,
                         *args,
                         num_heads=num_heads,
                         feature_size=feature_size,
                         norm_name=norm_name,
                         spatial_dims=spatial_dims,
                         drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         **kwargs)

        self.encoder5 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.cross_atte6 = cross_attention.TransFusion(
            hidden_size=feature_size * 16,
            num_layers=fusion_depths[5],
            mlp_dim=feature_size * 32,
            num_heads=num_heads[3],
            dropout_rate=drop_rate,
            atte_dropout_rate=attn_drop_rate,
            roi_size=img_size,
            scale=32,
            cross_attention_in_origin_view=cross_attention_in_origin_view)

    def forward_view_encoder(self, x):
        """Encode features."""
        x_hiddens = self.swinViT(x, self.normalize)
        x_enc0 = self.encoder1(x)
        x_enc1 = self.encoder2(x_hiddens[0])
        x_enc2 = self.encoder3(x_hiddens[1])
        x_enc3 = self.encoder4(x_hiddens[2])
        x_enc4 = self.encoder5(x_hiddens[3])  # xa_hidden[3]
        x_dec4 = self.encoder10(x_hiddens[4])
        return {
            'enc0': x_enc0,
            'enc1': x_enc1,
            'enc2': x_enc2,
            'enc3': x_enc3,
            'enc4': x_enc4,
            'dec4': x_dec4,
        }

    def forward_view_decoder(self, x_encoded: FeaturesDictType) -> torch.Tensor:
        """Decode features."""
        x_dec3 = self.decoder5(x_encoded['dec4'], x_encoded['enc4'])
        x_dec2 = self.decoder4(x_dec3, x_encoded['enc3'])
        x_dec1 = self.decoder3(x_dec2, x_encoded['enc2'])
        x_dec0 = self.decoder2(x_dec1, x_encoded['enc1'])
        x_out = self.decoder1(x_dec0, x_encoded['enc0'])
        x_logits = self.out(x_out)
        return x_logits

    def forward_view_cross_attention(
            self, xa_encoded: FeaturesDictType, xb_encoded: FeaturesDictType,
            views: Sequence[int]) -> Tuple[FeaturesDictType, FeaturesDictType]:
        """Inplace cross attention between views."""
        xa_encoded['dec4'], xb_encoded['dec4'] = self.cross_atte6(
            xa_encoded['dec4'], xb_encoded['dec4'], views)
        return xa_encoded, xb_encoded

    def forward(self, xa: torch.Tensor, xb: torch.Tensor,
                views: Sequence[int]) -> Sequence[torch.Tensor]:
        """Two views forward."""
        xa_encoded = self.forward_view_encoder(xa)
        xb_encoded = self.forward_view_encoder(xb)

        xa_encoded, xb_encoded = self.forward_view_cross_attention(
            xa_encoded, xb_encoded, views)
        return [
            self.forward_view_decoder(val) for val in [xa_encoded, xb_encoded]
        ]

    def no_weight_decay(self):
        """Disable weight_decay on specific weights."""
        nwd = {'swinViT.absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    def group_matcher(self, coarse=False):
        """Layer counting helper, used by timm."""
        return dict(
            stem=r'^swinViT\.absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^swinViT\.layers(\d+)\.0' if coarse else [
                (r'^swinViT\.layers(\d+)\.0.downsample', (0,)),
                (r'^swinViT\.layers(\d+)\.0\.\w+\.(\d+)', None),
                (r'^swinViT\.norm', (99999,)),
            ])


if __name__ == '__main__':
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 1, 64, 64, 64), device=cuda0)
        y = torch.rand((1, 1, 64, 64, 64), device=cuda0)
        # x = torch.rand((1, 3, 512, 512, 250), device=cuda0)
        model = SwinUNETR(img_size=(64, 64, 64),
                          in_channels=1,
                          out_channels=4,
                          feature_size=24,
                          fusion_depths=(1, 1, 1, 1, 1, 1),
                          drop_rate=0.0,
                          attn_drop_rate=0.0,
                          dropout_path_rate=0.2,
                          use_checkpoint=None,
                          cross_attention_in_origin_view=True, )
        model.cuda()

        data_list1, view_list1 = view_ops.permute_rand(x)
        # print(data_list1, view_list1)
        data_list2, view_list2 = view_ops.permute_rand(x)
        # print("——————————————————————————————————————————————————————————————————————")
        # print(data_list2, view_list2)
        output1, output2 = model(data_list1, data_list2, views=view_list1)
        out_list = [output1, output2]
        out_list = view_ops.permute_inverse(out_list, view_list1)
        # print(out_list)
        for i in out_list:
            my = torch.tensor(i)
            print(my.shape)
        softmax_out1 = torch.softmax(out_list[0], dim=1)
        # print(softmax_out1.shape)
        softmax_out2 = torch.softmax(out_list[1], dim=1)
        average_softmax = (softmax_out1 + softmax_out2) / 2
        # print(average_softmax.shape)
        pred = torch.argmax(average_softmax, dim=1, keepdim=True)
        print(pred.shape)
        target = torch.randint(0, 4, (2, 1, 64, 64, 64), device=cuda0)
        print(target.shape)
        mutual_crit = KLDivLoss(reduction='mean')
        # z = my_get_dice_loss(preds=y, target=y2)
        # pred = torch.softmax(out_list, dim=1)
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.5, smooth_dr=0.5
        )
        l=0
        # print(len(out_list)) =2
        for i in range(len(out_list)):
            self_loss =dice_loss(out_list[i], target)
            mutual_loss = 0
            for j in range(len(out_list)):  # KL divergence
                if i != j:
                    mutual_end = mutual_crit(F.log_softmax(out_list[i], dim=1), F.softmax(out_list[j], dim=1))
                    mutual_loss += mutual_end
                    print(f"j={j},self_loss={self_loss}, mutual_loss={mutual_loss}")
            l = l + (self_loss + mutual_loss / (len(out_list) - 1)) / len(out_list)
            print(f"i={i}, j={j},self_loss={self_loss}, mutual_loss={mutual_loss}")
        # inter = (pred * target)
        # union = (pred + target)
        # # pred:BCHW, target: BCHW
        # # if (len(pred.shape) == 5) and (len(target.shape) == 5):
        # inter = (pred * target).sum(dim=(2, 3, 4))  # Sum along D, H, and W dimensions
        # union = (pred + target).sum(dim=(2, 3, 4))  # Sum along D, H, and W dimensions
        # dice_loss = 1 - 2 * (inter + 1) / (union + 2)
        # result = dice_loss(pred.float(), target.float())
        # print(result)
        # if args.squared_dice:
        #     dice_loss = DiceCELoss(
        #         to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        #     )
        # else:
        #     dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        # mutual_loss = KLDivLoss(reduction='mean')  # CosineSimilarity(dim = 1)
        # for i in range(len(out_list)):
        #     self_loss = self_crit(out_list[i], target)
        #     mutual_loss = 0
        #     for j in range(len(out_list)):  # KL divergence
        #         if i != j:
        #             mutual_end = mutual_crit(F.log_softmax(out_list[i], dim=1), F.softmax(out_list[j], dim=1))
        #             mutual_loss += mutual_end
        #     loss = loss + (self_loss + mutual_loss / (len(out_list) - 1)) / len(out_list)
        #     self_loss_list.append(self_loss.item())
        #     mutual_loss_list.append(mutual_loss.item())
        # self_loss = torch.mean(torch.tensor(self_loss_list)).cuda(args.rank)
        # mutual_loss = torch.mean(torch.tensor(mutual_loss_list)).cuda(args.rank)
