"""View operations.

Input format: [B, C, X, Y, Z, ...]

NOTE(meijieru): 0 is reserved for identify transform.
"""

from typing import Callable, Sequence, Union

import enum

import torch

RotateType = int
PermuteType = int
TransformFuncType = Callable[[torch.Tensor], torch.Tensor]
# A composition of multiple view transoforms.
TransformsType = Sequence[Union[PermuteType, RotateType]]


class GroupName(enum.Enum):

    ROTATE = 1
    PERMUTE = 2


DEFAULT_ORDER = (GroupName.ROTATE, GroupName.PERMUTE)



rotation_transforms = {
    0: lambda x: x,
    1: lambda x: x.rot90(1, (3, 4)),
    2: lambda x: x.rot90(2, (3, 4)),
    3: lambda x: x.rot90(3, (3, 4)),
}
rotation_inverse_transforms = {
    0: lambda x: x,
    1: lambda x: x.rot90(3, (3, 4)),
    2: lambda x: x.rot90(2, (3, 4)),
    3: lambda x: x.rot90(1, (3, 4)),
}
permutation_transforms = {
    0: lambda x: x,
    1: lambda x: x.permute(0, 1, 3, 2, 4),
    2: lambda x: x.permute(0, 1, 4, 3, 2),
}
permutation_inverse_transforms = {
    0: lambda x: x,
    1: lambda x: x.permute(0, 1, 3, 2, 4),
    2: lambda x: x.permute(0, 1, 4, 3, 2),
}

all_forward_transforms = {
    GroupName.ROTATE: rotation_transforms,
    GroupName.PERMUTE: permutation_transforms,
}
all_backward_transforms = {
    GroupName.ROTATE: rotation_inverse_transforms,
    GroupName.PERMUTE: permutation_inverse_transforms,
}


def get_transforms_func(views: TransformsType,
                        orders: Sequence[GroupName] = DEFAULT_ORDER,
                        inverse: bool = False) -> TransformFuncType:
    """Gets sequential transform functions."""
    if len(views) != len(orders):
        raise ValueError()

    all_transforms = (all_forward_transforms
                      if not inverse else all_backward_transforms)
    funcs = [
        all_transforms[group_name][view]
        for view, group_name in zip(views, orders)
    ]
    funcs = funcs if not inverse else funcs[::-1]

    def aux(val):
        for func in funcs:
            val = func(val)
        return val

    return aux


# 定义数据
input_tensor = torch.randn(2, 3, 256, 256, 128)  # 假设这是你的输入张量

# 定义你希望的视角变换序列，这里以一个示例进行说明
views = [1, 2]  # 假设你希望执行1号排列变换、2号排列变换和不变换


# 定义转换函数
def apply_view_transform(input_tensor, views):
    # 获取转换函数
    transform_func = get_transforms_func(views)

    # 应用转换函数
    transformed_tensor = transform_func(input_tensor)
    return transformed_tensor


# 应用转换
output_tensor = apply_view_transform(input_tensor, views)
