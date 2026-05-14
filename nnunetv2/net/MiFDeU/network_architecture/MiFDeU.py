from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

from nnunetv2.net.BTCV.cross_attention import custom_downsampling, custom_upsampling
from nnunetv2.net.MiFDeU.network_architecture.MiFDeU_Encoder_Decoder import MiFDeU_Encoder, \
    MiFDeU_Decoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.net.MiFDeU.network_architecture.netrpp import TransformerBlock

device = torch.device(type='cuda', index=0)


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


class MiFDeU(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 patch_size: List[int],
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_blocks_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,

                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16

                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_blocks_per_stage_decoder, int):
            n_blocks_per_stage_decoder = [n_blocks_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_blocks_per_stage_decoder) == (
                    n_stages - 1), "n_blocks_per_stage_decoder must have one less entries " \
                                   f"as we have resolution stages. here: {n_stages} " \
                                   f"stages, so it should have {n_stages - 1} entries. " \
                                   f"n_blocks_per_stage_decoder: {n_blocks_per_stage_decoder}"
        # if isinstance(kernel_sizes, int):
        #     kernel_sizes = [kernel_sizes] * n_stages
        # if isinstance(features_per_stage, int):
        #     features_per_stage = [features_per_stage] * n_stages
        # if isinstance(strides, int):
        #     strides = [strides] * n_stages
        # if bottleneck_channels is None or isinstance(bottleneck_channels, int):
        #     bottleneck_channels = [bottleneck_channels] * n_stages
        # assert len(
        #     bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        # assert len(
        #     kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        # assert len(
        #     features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        # assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
        #                                  "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        # if not disable_default_stem:
        #     if stem_channels is None:
        #         stem_channels = features_per_stage[0]
        #     self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
        #                                   norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        #     input_channels = stem_channels
        # else:
        #     self.stem = None
        # crop_size = [160,96,160]
        # input_channels = 1
        # num_classes = 5
        # conv_op = nn.Conv3d
        #
        # self.embedding_dim = 96
        # depths = [2, 2, 2, 2]
        # num_heads = [3, 6, 12, 24]
        # embedding_patch_size = [1, 4, 4]
        # window_size = [[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]]
        # down_stride = [[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]]
        # self.deep_supervision = False
        self.ln = nn.LayerNorm(24)
        # self.mamba = Mamba(
        #     d_model=24,
        #     d_state=16,
        #     d_conv=4,
        #     expand=2,
        # )
        self.encoder = MiFDeU_Encoder(input_channels, patch_size, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True)
        self.encoder2 = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True)
        # self.encoder3 = ResidualMambaEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
        #                                      strides,
        #                                      n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
        #                                      dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
        #                                      return_skips=True, disable_default_stem=False, stem_channels=stem_channels)

        self.decoder = MiFDeU_Decoder(self.encoder, patch_size, strides, num_classes, n_blocks_per_stage_decoder, deep_supervision)
        # self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        # self.decoder = Decoder(pretrain_img_size=[160,96,160], embed_dim=4, window_size=window_size[::-1][1:],
        #                        patch_size=patch_size, num_heads=num_heads[::-1][1:], depths=depths[::-1][1:],
        #                        up_stride=down_stride[::-1][1:])
        # self.SS = SwinUNETR
        # self.decoder = UNetResDecoder(self.encoder, num_classes, n_blocks_per_stage_decoder, deep_supervision)

    def forward(self, x):
        tensor_list1 = self.encoder(x)
        # tensor_list11=[]
        # for i in range(len(tensor_list1)):
        #     middle_feature = tensor_list1[i]
        #     B, C = middle_feature.shape[:2]
        #     n_tokens = middle_feature.shape[2:].numel()
        #     img_dims = middle_feature.shape[2:]
        #     middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
        #     middle_feature_flat = self.ln(middle_feature_flat)
        #     out = self.mamba(middle_feature_flat)
        #     out = out.transpose(-1, -2).view(B, C, *img_dims)
        #     tensor_list11.append(out)

        tensor_list2 = self.encoder2(x)
        # tensor_list22 = []
        # for i in range(len(tensor_list2)):
        #     middle_feature = tensor_list2[i]
        #     B, C = middle_feature.shape[:2]
        #     n_tokens = middle_feature.shape[2:].numel()
        #     img_dims = middle_feature.shape[2:]
        #     middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
        #     middle_feature_flat = self.ln(middle_feature_flat)
        #     out = self.mamba(middle_feature_flat)
        #     out = out.transpose(-1, -2).view(B, C, *img_dims)
        #     tensor_list22.append(out)
        scale1 = 0.5
        scale2 = 0.5
        # view_list = [0] + np.random.permutation(3)[:1].tolist()
        # view_list = [0 if element == 0 else torch.tensor(element).to('cuda').item() for element in view_list]
        #eeeeeee
        # concatenated_tensors = [torch.cat((scale1 * z1, scale2 * z2), dim=2) for z1, z2 in
        #                         zip(tensor_list1, tensor_list2)]
        #
        # final_tensor_list = [F.avg_pool3d(z3, kernel_size=(2, 1, 1)) for z3 in concatenated_tensors]

        # final_tensor_list = [torch.add(scale1 * z1, scale2 * z2) for z1, z2 in zip(tensor_list1, tensor_list2)]
        # input_size_list11 = [150, 600, 4800, 38400, 307200, 2457600]
        # input_size_list = input_size_list11[::-1]
        # input_size_list21 = [24, 24, 48, 96, 192, 384]
        # input_size_list21 = [24, 24, 24, 24, 24, 24]
        # input_size_list2 = input_size_list21[::-1]
        input_size_list11 = [150, 600, 600, 600, 600,600]
        input_size_list11 = input_size_list11[::-1]
        # input_size_list11 = [720, 720, 46080, 5760, 720, 180]
        input_size_list21 = [384, 192, 96, 48, 24, 24]
        #
        # final_tensor_list1 = []
        # for i, tensor in enumerate(final_tensor_list):
        #         tensor = custom_downsampling(tensor)
        #         custom_input_size = input_size_list11[i]
        #         custom_hidden = input_size_list21[i]
        #         transformer = TransformerBlock(input_size=custom_input_size, hidden_size=custom_hidden, proj_size=64,
        #                                        dropout_rate=0.1).to(device)
        #         processed_tensor = transformer(tensor)
        #         processed_tensor = custom_upsampling(processed_tensor)
        #         final_tensor_list1.append(processed_tensor)

        # for i, tensor in enumerate(tensor_list2):
        #     tensor = custom_downsampling(tensor)
        #     custom_input_size = input_size_list[i]
        #     custom_hidden = input_size_list2[i]
        #     transformer = TransformerBlock(input_size=custom_input_size, hidden_size=custom_hidden, proj_size=96, num_heads=24,
        #                                    dropout_rate=0.1).to(device)
        #     processed_tensor = transformer(tensor)
        #     processed_tensor = custom_upsampling(processed_tensor)
        #     final_tensor_list2.append(processed_tensor)
        # for (tensor1,tensor2) in zip(final_tensor_list1,final_tensor_list2):
        #     res = cross_process_tensors(tensor1,tensor2,view_list=self.view_list)
        # final_tensor_list = cross_process_tensors(final_tensor_list1,final_tensor_list2,view_list=self.view_list)
        final_tensor_list = [torch.add(scale1*z1, scale2*z2) for z1, z2 in zip(tensor_list1,tensor_list2)]
        # output = [torch.add(t1 * scale1, t2 * scale2) for t1, t2 in zip(skips1, skips2)]
        # return self.decoder(output)
        # return [self.decoder(val) for val in [skips1, skips2]]
        # for i, tensor in enumerate(final_tensor_list3):
        #     if i >0:
        #         custom_input_size = input_size_list[i]
        #         transformer = TransformerBlock(input_size=custom_input_size, hidden_size=24, proj_size=64, num_heads=4,
        #                                        dropout_rate=0.2)
        #         transformer.to(device)  # ½«transformerÄ£ÐÍÒÆ¶¯µ½Éè±¸
        #         tensor = tensor.to(device)  # ½«ÊäÈëÕÅÁ¿ÒÆ¶¯µ½Éè±¸
        #         processed_tensor = transformer(tensor)
        #         final_tensor_list.append(processed_tensor)
        #     else:
        #         final_tensor_list.append(final_tensor_list3[i])

        # final_tensor_list = process_tensors(tensor_list1,tensor_list2,self.view_list)
        # final_tensor_list2 = [torch.add(scale1 * z1, scale2 * z2) for z1, z2 in zip(final_tensor_list, final_tensor_list1)]
        # neck = final_tensor_list[-1]
        # return self.decoder(final_tensor_list1)

        # skips = final_tensor_list
        # middle_feature = skips[-1]
        # B, C = middle_feature.shape[:2]
        # n_tokens = middle_feature.shape[2:].numel()
        # img_dims = middle_feature.shape[2:]
        # middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
        # middle_feature_flat = self.ln(middle_feature_flat)
        # out = self.mamba(middle_feature_flat)
        # out = out.transpose(-1, -2).view(B, C, *img_dims)
        # skips[-1] = out

        # for i in range(len(final_tensor_list)):
        #     middle_feature = final_tensor_list[i]
        #     B, C = middle_feature.shape[:2]
        #     n_tokens = middle_feature.shape[2:].numel()
        #     img_dims = middle_feature.shape[2:]
        #     middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
        #     middle_feature_flat = self.ln(middle_feature_flat)
        #     out = self.mamba(middle_feature_flat)
        #     out = out.transpose(-1, -2).view(B, C, *img_dims)
        #     final_tensor_list[i] = out

        return self.decoder(tensor_list1,tensor_list2)
    # def forward(self, xa: torch.Tensor, xb: torch.Tensor,
    #                 views: Sequence[int]) -> Sequence[torch.Tensor]:
    #     skips1 = self.encoder(xa)
    #     skips2 = self.encoder(xb)
    #     return [self.decoder(skips) for skips in [skips1,skips2]]
    # return [self.decoder(val) for val in [skips1, skips2]]

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
        model = MiFDeU(input_channels=3, base_num_features=24, num_classes=4, num_pool=4, patch_size=(96, 192, 160),
                       num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=conv_op, norm_op=norm_op,
                       norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                       dropout_op_kwargs=dropout_op_kwargs,
                       nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs,
                       deep_supervision=True, dropout_in_localization=False, final_nonlin=lambda x: x,

                       convolutional_pooling=True, convolutional_upsampling=True)
        # model = MiFDeU()
        # num_stages = len(configuration_manager.conv_kernel_sizes)
        # 
        # dim = len(configuration_manager.conv_kernel_sizes[0])
        # conv_op = convert_dim_to_conv_op(dim)
        model.cuda(0)
        y = model(x)
        print(len(y))
        for i in range(0, len(y)):
            print(y[i].shape)
