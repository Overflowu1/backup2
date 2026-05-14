"""View operations."""

from typing import Sequence, Tuple

import torch
import numpy as np

from nnunetv2.net.SwinMM.utils import view_transforms

PermuteType = view_transforms.PermuteType
TransformFuncType = view_transforms.TransformFuncType


def get_permute_transform(view_src: PermuteType,
                          view_dst: PermuteType) -> TransformFuncType:
    """Gets transform function from view src to view dst."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        x_view_0 = view_transforms.permutation_inverse_transforms[view_src](x)
        return view_transforms.permutation_transforms[view_dst](
            x_view_0).contiguous()

    return transform


def permute_inverse(xs: Sequence[torch.Tensor],
                    views: Sequence[PermuteType]) -> Sequence[torch.Tensor]:
    """Transforms data back to origin view."""
    return [get_permute_transform(view, 0)(x) for x, view in zip(xs, views)]


def permute_rand(
    x: torch.Tensor,
    num_samples: int = 2
) -> Tuple[Sequence[torch.Tensor], Sequence[PermuteType]]:
    """Samples different transforms of data."""
    num_permutes = len(view_transforms.permutation_transforms)
    if num_samples > num_permutes:
        raise ValueError('Duplicate samples.')
    view_dsts1 = np.random.permutation(num_permutes)[:num_samples].tolist()
    transformed_tensors = [get_permute_transform(0, view)(x) for view in view_dsts1]
    stacked_tensor1 = torch.stack(transformed_tensors).squeeze(1)
    view_dsts2 = np.random.permutation(num_permutes)[:num_samples].tolist()
    transformed_tensors = [get_permute_transform(0, view)(x) for view in view_dsts2]
    #stacked_tensor2 = torch.stack(transformed_tensors).squeeze(1)
    return stacked_tensor1, view_dsts1

import torch
tensor = torch.randn(2, 5, 64, 64, 64)
tensor_list1 = torch.chunk(tensor, chunks=2, dim=0)
tensor_list2 = torch.split(tensor, split_size_or_sections=1, dim=0)
tensor_list3 = [t for t in tensor_list2]

import torch

# 创建一个形状为 (1, 5, 64, 64, 64) 的示例张量
tensor1 = torch.randn( 5, 64, 64, 64)
tensor2 = torch.randn( 5, 64, 64, 64)


tensor1 = tensor1.unsqueeze(0).expand(2, -1, -1, -1, -1)  # 将第一个维度的大小从1改为2
tensor2 = tensor2.unsqueeze(0).expand(2, -1, -1, -1, -1)  # 将第一个维度的大小从1改为2
transformed_xs = [tensor1,tensor2]
t = torch.stack(transformed_xs)
re = torch.cat(transformed_xs,dim=0)

import torch

# 假设有两个形状相同的张量
tensor1 = torch.randn(5, 64, 64, 64)
tensor2 = torch.randn(5, 64, 64, 64)

# 使用 torch.cat 在 dim=1 上连接这两个张量，保持 dim=0 不变
concatenated_tensor = torch.cat([tensor1.unsqueeze(0), tensor2.unsqueeze(0)], dim=0)

import torch

# 假设有两个形状相同的张量
tensor1 = torch.randn(5, 64, 64, 64)
tensor2 = torch.randn(5, 64, 64, 64)

# 使用 torch.cat 在 dim=0 上连接这两个张量，保持所有维度不变
concatenated_tensor = torch.cat([tensor1, tensor2], dim=0)


import torch

# 假设有两个形状相同的张量
tensor1 = torch.randn(5, 64, 64, 64)
tensor2 = torch.randn(5, 64, 64, 64)

# 使用 torch.cat 将它们在维度0上连接成一个新的张量
concatenated_tensor = torch.cat([tensor1.unsqueeze(0), tensor2.unsqueeze(0)], dim=0)
import torch

# 假设有两个形状相同的张量
tensor1 = torch.randn(5, 64, 64, 64)
tensor2 = torch.randn(5, 64, 64, 64)

# 使用 torch.unsqueeze 在第一个维度上增加一个维度
tensor1 = tensor1
tensor2 = tensor2

# 使用 torch.cat 在第一个维度上连接这两个张量
combined_tensor = torch.cat([tensor1, tensor2], dim=0).mean(dim=0)

# 最终得到的 combined_tensor 形状为 (2, 5, 64, 64, 64)
# 如果需要变成 (1, 5, 64, 64, 64)，可以再次使用 torch.unsqueeze
final_tensor = combined_tensor.squeeze(0)

