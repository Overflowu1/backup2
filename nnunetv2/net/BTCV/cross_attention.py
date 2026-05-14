"""TransFusion from TransFusion: Multi-view Divergent Fusion for Medical Image Segmentation with Transformers."""
import random
from typing import Sequence, Union

import math, copy

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.net.BTCV.view_ops import permute_inverse
from nnunetv2.net.SwinMM.utils.view_ops import get_permute_transform

device = torch.device(type='cuda', index=0)
class Attention(nn.Module):

    def __init__(self, num_heads=8, hidden_size=768, atte_dropout_rate=0.0):
        super(Attention, self).__init__()
        # self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(atte_dropout_rate)
        self.proj_dropout = nn.Dropout(atte_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x_1, x_2):
        mixed_query_layer_1 = self.query(x_1)
        mixed_key_layer_1 = self.key(x_1)
        mixed_value_layer_1 = self.value(x_1)
        query_layer_1 = self.transpose_for_scores(mixed_query_layer_1)
        key_layer_1 = self.transpose_for_scores(mixed_key_layer_1)
        value_layer_1 = self.transpose_for_scores(mixed_value_layer_1)
        mixed_query_layer_2 = self.query(x_2)
        mixed_key_layer_2 = self.key(x_2)
        mixed_value_layer_2 = self.value(x_2)
        query_layer_2 = self.transpose_for_scores(mixed_query_layer_2)
        key_layer_2 = self.transpose_for_scores(mixed_key_layer_2)
        value_layer_2 = self.transpose_for_scores(mixed_value_layer_2)

        attention_scores_1 = torch.matmul(query_layer_1,
                                          key_layer_2.transpose(-1, -2))
        attention_scores_1 = attention_scores_1 / math.sqrt(
            self.attention_head_size)
        attention_probs_1 = self.softmax(attention_scores_1)
        # weights_st = attention_probs_st if self.vis else None
        attention_probs_1 = self.attn_dropout(attention_probs_1)
        context_layer_1 = torch.matmul(attention_probs_1, value_layer_2)
        context_layer_1 = context_layer_1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_1 = context_layer_1.size()[:-2] + (
            self.all_head_size,)
        context_layer_1 = context_layer_1.view(*new_context_layer_shape_1)
        attention_output_1 = self.out(context_layer_1)
        attention_output_1 = self.proj_dropout(attention_output_1)

        attention_scores_2 = torch.matmul(query_layer_2,
                                          key_layer_1.transpose(-1, -2))
        attention_scores_2 = attention_scores_2 / math.sqrt(
            self.attention_head_size)
        attention_probs_2 = self.softmax(attention_scores_2)
        # weights_st = attention_probs_st if self.vis else None
        attention_probs_2 = self.attn_dropout(attention_probs_2)
        context_layer_2 = torch.matmul(attention_probs_2, value_layer_1)
        context_layer_2 = context_layer_2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_2 = context_layer_2.size()[:-2] + (
            self.all_head_size,)
        context_layer_2 = context_layer_2.view(*new_context_layer_shape_2)
        attention_output_2 = self.out(context_layer_2)
        attention_output_2 = self.proj_dropout(attention_output_2)

        return attention_output_1, attention_output_2


class Block(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 mlp_dim=1536,
                 dropout_rate=0.5,
                 num_heads=8,
                 atte_dropout_rate=0.0):
        super(Block, self).__init__()

        del mlp_dim
        del dropout_rate

        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(num_heads=num_heads,
                              hidden_size=hidden_size,
                              atte_dropout_rate=atte_dropout_rate)

    def forward(self, x_1, x_2):
        x_1 = self.attention_norm(x_1)
        x_2 = self.attention_norm(x_2)
        x_1, x_2 = self.attn(x_1, x_2)
        return x_1, x_2


class TransFusion1(nn.Module):

    def __init__(self,
                 hidden_size: int = 24,
                 num_layers: int = 2,
                 mlp_dim: int = 24 * 32,
                 dropout_rate: float = 0.1,
                 num_heads: int = 24,
                 atte_dropout_rate: float = 0.1,
                 roi_size: Union[Sequence[int], int] = (10, 6, 10),
                 scale: int = 1,
                 cross_attention_in_origin_view: bool = False):
        super().__init__()
        if isinstance(roi_size, int):
            roi_size = [roi_size for _ in range(3)]
        self.cross_attention_in_origin_view = cross_attention_in_origin_view
        patch_size = (1, 1, 1)
        n_patches = (roi_size[0] // patch_size[0] //
                     scale) * (roi_size[1] // patch_size[1] //
                               scale) * (roi_size[2] // patch_size[2] // scale)
        # n_patches = (roi_size[0] // patch_size[0]) * (roi_size[1] // patch_size[1]) * (roi_size[2] // patch_size[2])
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.patch_embeddings = nn.Conv3d(in_channels=hidden_size,
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        for _ in range(num_layers):
            layer = Block(hidden_size=hidden_size,
                          mlp_dim=mlp_dim,
                          dropout_rate=dropout_rate,
                          num_heads=num_heads,
                          atte_dropout_rate=atte_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x_1, x_2, view_list):
        # if self.cross_attention_in_origin_view:
        #     x_1, x_2 = permute_inverse([x_1, x_2], view_list)
        # else:
        #     # Align x_2 to x_1.
        #     x_2 = get_permute_transform(*view_list[::-1])(x_2)
        B, C, H, W, D = x_1.shape
        B1, C1, H1, W1, D1 = x_2.shape
        x_1 = self.patch_embeddings(x_1)
        x_2 = self.patch_embeddings(x_2)
        x_1 = x_1.flatten(2).transpose(-1, -2)
        x_2 = x_2.flatten(2).transpose(-1, -2)
        x_1 = x_1 + self.position_embeddings
        x_2 = x_2 + self.position_embeddings
        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)
        for layer_block in self.layer:
            x_1, x_2 = layer_block(x_1, x_2)
        x_1 = self.encoder_norm(x_1)
        x_2 = self.encoder_norm(x_2)
        B, n_patch, hidden = x_1.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # l, h, w = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(
        #     np.cbrt(n_patch))
        B2, n_patch2, hidden2 = x_1.size()
        l, h, w = H, W, D
        l1, h1, w1 = H1, W1, D1
        x_1 = x_1.permute(0, 2, 1).contiguous().view(B, hidden, l, h, w)
        x_2 = x_2.permute(0, 2, 1).contiguous().view(B2, hidden2, l1, h1, w1)
        # if self.cross_attention_in_origin_view:
        #     x_1, x_2 = permute_inverse([x_1, x_2], view_list)
        # else:
        #     x_2 = get_permute_transform(*view_list)(x_2)

        return x_1, x_2


class TransFusion2(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 num_layers: int = 6,
                 mlp_dim: int = 1536,
                 dropout_rate: float = 0.5,
                 num_heads: int = 8,
                 atte_dropout_rate: float = 0.0,
                 roi_size: Union[Sequence[int], int] = (10, 6, 10),
                 scale: int = 16,
                 cross_attention_in_origin_view: bool = False):
        super().__init__()
        if isinstance(roi_size, int):
            roi_size = [roi_size for _ in range(3)]
        self.cross_attention_in_origin_view = cross_attention_in_origin_view
        patch_size = (1, 1, 1)
        # n_patches = (roi_size[0] // patch_size[0] //
        #              scale) * (roi_size[1] // patch_size[1] //
        #                        scale) * (roi_size[2] // patch_size[2] // scale)
        n_patches = (roi_size[0] // patch_size[0]) * (roi_size[1] // patch_size[1]) * (roi_size[2] // patch_size[2])
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.patch_embeddings = nn.Conv3d(in_channels=hidden_size,
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        for _ in range(num_layers):
            layer = Block(hidden_size=hidden_size,
                          mlp_dim=mlp_dim,
                          dropout_rate=dropout_rate,
                          num_heads=num_heads,
                          atte_dropout_rate=atte_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x_1, x_2, view_list):
        # if self.cross_attention_in_origin_view:
        #     x_1, x_2 = permute_inverse([x_1, x_2], view_list)
        # else:
        #     # Align x_2 to x_1.
        #     x_2 = get_permute_transform(*view_list[::-1])(x_2)
        B, C, H, W, D = x_1.shape
        B1, C1, H1, W1, D1 = x_2.shape
        x_1 = self.patch_embeddings(x_1)
        x_2 = self.patch_embeddings(x_2)
        x_1 = x_1.flatten(2).transpose(-1, -2)
        x_2 = x_2.flatten(2).transpose(-1, -2)
        x_1 = x_1 + self.position_embeddings
        x_2 = x_2 + self.position_embeddings
        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)
        for layer_block in self.layer:
            x_1, x_2 = layer_block(x_1, x_2)
        x_1 = self.encoder_norm(x_1)
        x_2 = self.encoder_norm(x_2)
        B, n_patch, hidden = x_1.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # l, h, w = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(
        #     np.cbrt(n_patch))
        B2, n_patch2, hidden2 = x_1.size()
        l, h, w = H, W, D
        l1, h1, w1 = H1, W1, D1
        x_1 = x_1.permute(0, 2, 1).contiguous().view(B, hidden, l, h, w)
        x_2 = x_2.permute(0, 2, 1).contiguous().view(B2, hidden2, l1, h1, w1)
        # if self.cross_attention_in_origin_view:
        #     x_1, x_2 = permute_inverse([x_1, x_2], view_list)
        # else:
        #     x_2 = get_permute_transform(*view_list)(x_2)

        return x_1, x_2


# tensor_list1 = torch.randn(1, 192, 20, 12, 20)
# tensor_list2 = torch.randn(1, 192, 20, 12, 20)
tensor_list1 = torch.randn(1, 384, 10, 6, 10)
tensor_list2 = torch.randn(1, 384, 10, 6, 10)


view_list = [0, 1]


model1 = TransFusion1(
    hidden_size=24,
    num_layers=2,
    mlp_dim=24 * 32,
    num_heads=24,
    dropout_rate=0.1,
    atte_dropout_rate=0.1,
    roi_size=(10, 6, 10),
    scale=1,
)
model2 = TransFusion2(
    hidden_size=24,
    num_layers=2,
    mlp_dim=24 * 32,
    num_heads=24,
    dropout_rate=0.1,
    atte_dropout_rate=0.1,
    roi_size=(10, 6, 10),
    scale=1,
    cross_attention_in_origin_view=False)

# ÒÆ¶¯Ä£ÐÍºÍÕÅÁ¿µ½CUDA
# model1
# tensor_list1 = tensor_list1
# tensor_list2 = tensor_list2

# ÔËÐÐÄ£ÐÍ
# x = model1(tensor_list1, tensor_list2, view_list)
# res = torch.concat(x,dim=1)
# print(res.shape)
# conv = nn.ConvTranspose3d(48, 24, kernel_size=3, stride=1, padding=1)
# conv2 = nn.ConvTranspose3d(48, 24, kernel_size=7, stride=1, padding=3)
# conv = nn.ConvTranspose3d(768,384,kernel_size=3, stride=1, padding=1)
# conv2 = nn.ConvTranspose3d(768, 384, kernel_size=7, stride=1, padding=3)
# res2 = conv(res)
# print(res2.shape)
#
# sizes = [
#     (1, 24, 5, 6, 5),
#     (1, 24, 10, 6, 10),
#     (1, 24, 20, 12, 20),
#     (1, 24, 40, 24, 40),
#     (1, 24, 80, 48, 80),
#     (1, 24, 160, 96, 160)
# ]
#
# tensor_list11 = [torch.randn(*size) for size in sizes]
# tensor_list1 = tensor_list11[::-1]
# tensor_list22 = [torch.randn(*size) for size in sizes]
# tensor_list2 = tensor_list22[::-1]
#
# final_tensor_list = []


def custom_downsampling(input):
    B, C, H, W, D = input.shape


    downsampling_layers = nn.Sequential()
    # condition = (H > 10).item() and (W > 6).item() and (D > 10).item()
    # condition = torch.logical_and(H > 10, torch.logical_and(W > 6, D > 10))
    while (H > 10) and (W > 6)and (D > 10):
        C *= 2
        downsampling_layers.add_module(f'conv{C}',
                                       nn.Conv3d(in_channels=C // 2, out_channels=C, kernel_size=3, stride=2,
                                                 padding=1).to(device))
        downsampling_layers.add_module(f'maxpool{C}', nn.MaxPool3d(kernel_size=1))
        H //= 2
        W //= 2
        D //= 2

    # input = input.to(torch.float32)
    output_tensor = downsampling_layers(input)

    return output_tensor


def custom_upsampling(input_tensor):
    B, C, H, W, D = input_tensor.shape
    upsampling_layers = nn.Sequential()
    while C > 24:
    # condition = (C > 24).item()
    # while condition:
        C //= 2
        H *= 2
        W *= 2
        D *= 2
        upsampling_layers.add_module(f'conv{C}',
                                     nn.ConvTranspose3d(in_channels=C * 2, out_channels=C, kernel_size=3, stride=2,
                                                        padding=1, output_padding=1).to(device))
    # input_tensor = input_tensor.to(torch.float32)
    restored_tensor = upsampling_layers(input_tensor)
    return restored_tensor


def cross_process_tensors(tensor_list1, tensor_list2, view_list):
    final_tensor_list = []
    i = 5
    for (tensor1, tensor2) in zip(tensor_list1, tensor_list2):
        if i == 0:
            model = TransFusion1(roi_size=(5, 6, 5)).to(device)
            x1,x2 = model(tensor1 ,tensor2, view_list)
            res = torch.cat([x1,x2], dim=1)
            conv = nn.ConvTranspose3d(48, 24, kernel_size=3, stride=1, padding=1).to(device)
            res2 = conv(res)
            # res2 = torch.add(x1, x2)
            i -= 1
        elif i == 1:
            model = TransFusion1(roi_size=(10, 6, 10)).to(device)
            x1,x2 = model(tensor1 ,tensor2, view_list)
            res = torch.cat([x1,x2], dim=1)
            conv2 = nn.ConvTranspose3d(48, 24, kernel_size=7, stride=1, padding=3).to(device)
            res2 = conv2(res)
            # res2 = torch.add(x1, x2)
            i -= 1
        else:
            t1 = custom_downsampling(tensor1)
            t2 = custom_downsampling(tensor2)
            model = TransFusion2(hidden_size=24 * (2 ** (i - 1)), roi_size=(10, 6, 10)).to(device)
            x1,x2 = model(t1, t2, view_list)
            # C = t1.shape[1]
            # conv3 = nn.ConvTranspose3d(C * 2, C, kernel_size=3, stride=1, padding=1).to(device)
            res2 = torch.add(x1, x2)
            #res = torch.cat(x, dim=1)
            #res2 = conv3(res)
            # res2 = custom_upsampling(res2)
            i -= 1
        final_tensor_list.append(res2)
    final_tensor_list_subset = final_tensor_list[:4]

    custom_upsampled_tensors = [custom_upsampling(tensor) for tensor in final_tensor_list_subset]

    final_tensor_list[:4] = custom_upsampled_tensors
    return final_tensor_list
# view_list =[0,0]
# x = process_tensors(tensor_list1,tensor_list2,view_list)
# for i in x:
#     print(i.shape)
