import torch
import math
import copy
from torch import nn
from einops import rearrange
from functools import partial
from typing import Union, Type, List, Tuple
import torch
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
from nnunetv2.net.Seg3d.segformer3d_e_d import MixVisionTransformer, SegFormerDecoderHead


def build_segformer3d_model(config=None):
    model = SegFormer3D(
        in_channels=config["model_parameters"]["in_channels"],
        sr_ratios=config["model_parameters"]["sr_ratios"],
        embed_dims=config["model_parameters"]["embed_dims"],
        patch_kernel_size=config["model_parameters"]["patch_kernel_size"],
        patch_stride=config["model_parameters"]["patch_stride"],
        patch_padding=config["model_parameters"]["patch_padding"],
        mlp_ratios=config["model_parameters"]["mlp_ratios"],
        num_heads=config["model_parameters"]["num_heads"],
        depths=config["model_parameters"]["depths"],
        decoder_head_embedding_dim=config["model_parameters"][
            "decoder_head_embedding_dim"
        ],
        num_classes=config["model_parameters"]["num_classes"],
        decoder_dropout=config["model_parameters"]["decoder_dropout"],
    )
    return model


class SegFormer3D(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            sr_ratios: list = [4, 2, 1, 1,1,1],
            embed_dims: list = [8,16,32, 64, 160, 256],
            patch_kernel_size: list = [7, 3, 3, 3,3,3],
            patch_stride: list = [4, 2, 2, 2,2,2],
            patch_padding: list = [3, 1, 1, 1,1,1],
            mlp_ratios: list = [4, 4, 4, 4,4,4],
            num_heads: list = [1, 1, 1, 1,1,1],
            depths: list = [2, 2, 2, 2,2,2],
            decoder_head_embedding_dim: int = 256,
            num_classes: int = 3,
            decoder_dropout: float = 0.0,
    ):
        """
        in_channels: number of the input channels
        img_volume_dim: spatial resolution of the image volume (Depth, Width, Height)
        sr_ratios: the rates at which to down sample the sequence length of the embedded patch
        embed_dims: hidden size of the PatchEmbedded input
        patch_kernel_size: kernel size for the convolution in the patch embedding module
        patch_stride: stride for the convolution in the patch embedding module
        patch_padding: padding for the convolution in the patch embedding module
        mlp_ratios: at which rate increases the projection dim of the hidden_state in the mlp
        num_heads: number of attention heads
        depths: number of attention layers
        decoder_head_embedding_dim: projection dimension of the mlp layer in the all-mlp-decoder module
        num_classes: number of the output channel of the network
        decoder_dropout: dropout rate of the concatenated feature maps

        """
        super().__init__()
        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )
        # decoder takes in the feature maps in the reversed order
        reversed_embed_dims = embed_dims[::-1]
        self.segformer_decoder = SegFormerDecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # embedding the input
        x = self.segformer_encoder(x)
        # # unpacking the embedded features generated by the transformer
        c1 = x[0]
        c2 = x[1]
        c3 = x[2]
        c4 = x[3]
        # decoding the embedded features
        x = self.segformer_decoder(c1, c2, c3, c4)
        return x


if __name__ == "__main__":
    # input = torch.randint(
    #     low=0,
    #     high=255,
    #     size=(1, 4, 128, 128, 128),
    #     dtype=torch.float,
    # )
    sizes = [
        (1, 4, 160, 160, 160),
        (1, 4, 80, 80, 80),
        (1, 4, 40, 40, 40),
        (1, 4, 20, 20, 20),
        (1, 4, 10, 10, 10),
        (1, 4, 5, 5, 5)
    ]
    input = [torch.randn(*size).to("cuda:0") for size in sizes]
    # input = input.to("cuda:0")
    segformer3D = SegFormer3D().to("cuda:0")
    for i in input:
        output = segformer3D(i)
        print(output.shape)