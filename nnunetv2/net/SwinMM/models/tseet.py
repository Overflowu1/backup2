import torch
from nnunetv2.training.nnUNetTrainer.SwinMM.utils.view_ops import get_permute_transform
import torch
from typing import Sequence

xs = torch.rand((1,5,64, 64, 64))
# 创建一个包含多个 PyTorch Tensor 的 Sequence[torch.Tensor]
xs: Sequence[torch.Tensor] = []

tensor = torch.rand((1,5,64, 64, 64))
xs.append(tensor)

# 现在 sequence_of_tensors 包含了五个形状相同的 Tensor

views = [2, 0]
re = [get_permute_transform(view, 0)(x) for x, view in zip(xs, views)]
import numpy as np
views = []
for _ in range(0,2):
    views.append(np.random.randint(0, 3))