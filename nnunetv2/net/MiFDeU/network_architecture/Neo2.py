from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
from nnunetv2.net.MiFDeU.network_architecture.MiFDeU_Encoder_Decoder import NexToU_Encoder, \
    NexToU_Decoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.net.MiFDeU.network_architecture.netrpp import TransformerBlock


class AttentionFusion(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # ¶¨Òå¶àÍ·×¢ÒâÁ¦²ã
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, decoder_outputs1, decoder_outputs2):
        # ½«½âÂëÆ÷µÄÊä³ö¶ÑµþÆðÀ´
        stacked_outputs1 = torch.stack(decoder_outputs1, dim=1)  # [B, num_outputs, seq_len, d_model]
        stacked_outputs2 = torch.stack(decoder_outputs2, dim=1)

        # Ê¹ÓÃ¶àÍ·×¢ÒâÁ¦¼ÆËã×¢ÒâÁ¦·ÖÊý
        attn_output1, _ = self.multihead_attn(stacked_outputs2, stacked_outputs1, stacked_outputs1)  # ¼ÆËã×¢ÒâÁ¦·ÖÊý
        attn_weights1 = F.softmax(attn_output1, dim=1)  # ¶Ô½âÂëÆ÷1µÄÈ¨ÖØ

        # ¼ÓÈ¨ÈÚºÏ
        fused_output1 = torch.sum(attn_weights1.unsqueeze(-1) * stacked_outputs1, dim=1)

        return fused_output1


class NexToU(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_size: List[int],
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = NexToU_Encoder(input_channels, patch_size, n_stages, features_per_stage, conv_op, kernel_sizes,
                                      strides,
                                      n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                      dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                      nonlin_first=nonlin_first)
        self.encoder2 = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True)

        self.decoder = NexToU_Decoder(self.encoder, patch_size, strides, num_classes, n_conv_per_stage_decoder,
                                      deep_supervision,
                                      nonlin_first=nonlin_first)

        #self.SS = SwinUNETR
    def forward(self, x):
        tensor_list1 = self.encoder(x)
        tensor_list2 = self.encoder2(x)
        scale1 = 0.5

        scale2 = 0.5
        #input_size_list = [150, 600, 4800, 38400, 307200, 2457600]  # ¶ÔÓ¦µÄ input_size ÁÐ±í
        input_size_list = [2457600, 307200, 38400, 4800, 600, 150]
        final_tensor_list = []
        final_tensor_list1 = []
        final_tensor_list2 = []
        final_tensor_list3 = []

        # Éè¶¨Ä¿±êÉè±¸£¬¼ÙÉèÄãÒªÔÚcuda:0ÉÏÔËÐÐ
        device = torch.device("cuda:0")

        for i, tensor in enumerate(tensor_list1):
            if i>0:
                custom_input_size = input_size_list[i]
                transformer = TransformerBlock(input_size=custom_input_size, hidden_size=24, proj_size=64, num_heads=4,
                                               dropout_rate=0.2)
                transformer.to(device)  # ½«transformerÄ£ÐÍÒÆ¶¯µ½Éè±¸
                tensor = tensor.to(device)  # ½«ÊäÈëÕÅÁ¿ÒÆ¶¯µ½Éè±¸
                processed_tensor = transformer(tensor)
                final_tensor_list1.append(processed_tensor)
            else:
                final_tensor_list1.append(tensor_list1[i])

        for i, tensor in enumerate(tensor_list2):
            if i >0:
                custom_input_size = input_size_list[i]
                transformer = TransformerBlock(input_size=custom_input_size, hidden_size=24, proj_size=64, num_heads=4,
                                               dropout_rate=0.2)
                transformer.to(device)  # ½«transformerÄ£ÐÍÒÆ¶¯µ½Éè±¸
                tensor = tensor.to(device)  # ½«ÊäÈëÕÅÁ¿ÒÆ¶¯µ½Éè±¸
                processed_tensor = transformer(tensor)
                final_tensor_list2.append(processed_tensor)
            else:
                final_tensor_list2.append(tensor_list2[i])

        final_tensor_list3 = [torch.add(scale1*z1, scale2*z2) for z1, z2 in zip(final_tensor_list1, final_tensor_list2)]
        #cross_attention.TransFusion(skips1,skips2)
        #output = [torch.add(t1 * scale1, t2 * scale2) for t1, t2 in zip(skips1, skips2)]
        #return self.decoder(output)
        #return [self.decoder(val) for val in [skips1, skips2]]
        for i, tensor in enumerate(final_tensor_list3):
            if i >0:
                custom_input_size = input_size_list[i]
                transformer = TransformerBlock(input_size=custom_input_size, hidden_size=24, proj_size=64, num_heads=4,
                                               dropout_rate=0.2)
                transformer.to(device)  # ½«transformerÄ£ÐÍÒÆ¶¯µ½Éè±¸
                tensor = tensor.to(device)  # ½«ÊäÈëÕÅÁ¿ÒÆ¶¯µ½Éè±¸
                processed_tensor = transformer(tensor)
                final_tensor_list.append(processed_tensor)
            else:
                final_tensor_list.append(final_tensor_list3[i])
        return self.decoder(final_tensor_list)
    # def forward(self, xa: torch.Tensor, xb: torch.Tensor,
    #                 views: Sequence[int]) -> Sequence[torch.Tensor]:
    #     skips1 = self.encoder(xa)
    #     skips2 = self.encoder(xb)
    #     return [self.decoder(skips) for skips in [skips1,skips2]]
        #return [self.decoder(val) for val in [skips1, skips2]]


    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


if __name__ == '__main__':
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((2, 3, 96, 192, 160), device=cuda0)
        z = torch.randint(0, 4, (2, 1, 96, 192, 160), device=cuda0)
        print(z.shape)
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        # self.embedding_dim = 24
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # x = torch.rand((1, 3, 512, 512, 250), device=cuda0)
        model = NexToU(input_channels=3, base_num_features=24, num_classes=4, num_pool=4, patch_size=(96, 192, 160),
                       num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=conv_op, norm_op=norm_op,
                       norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                       dropout_op_kwargs=dropout_op_kwargs,
                       nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs,
                       deep_supervision=True, dropout_in_localization=False, final_nonlin=lambda x: x,

                       convolutional_pooling=True, convolutional_upsampling=True)
        # model = NexToU()
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)
        model.cuda()
        y = model(x)
        print(len(y))
        for i in range(0, len(y)):
            print(y[i].shape)
