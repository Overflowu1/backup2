"""View operations."""

from typing import Sequence, Tuple

import torch
import numpy as np

from nnunetv2.net.BTCV import view_transforms

PermuteType = view_transforms.PermuteType
TransformFuncType = view_transforms.TransformFuncType


def get_permute_transform(view_src: PermuteType,
                          view_dst: PermuteType) -> TransformFuncType:
    """Gets transform function from view src to view dst."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        x_view_0 = view_transforms.permutation_inverse_transforms[view_src](x).contiguous()
        return view_transforms.permutation_transforms[view_dst](
            x_view_0).contiguous()

    return transform

# def permute_inverse(xs: Sequence[torch.Tensor],
#                     views: Sequence[PermuteType]) -> Sequence[torch.Tensor]:
#     """Transforms data back to origin view."""
    # num_dimensions = xs[0].dim()
    # print(num_dimensions)
    # transformed_xs = []
    # for x, view in zip(xs, views):
    #     x = x.unsqueeze(0)
    #     transformed_x = get_permute_transform(view, 0)(x)
    #     transformed_xs.append(transformed_x)
    #re = torch.cat(transformed_xs,dim=0)

    # for x, view in zip(xs, views):
    #     if x.dim() == 4:
    #
    #         print(x.shape,view)
    #     else:
    #         re = [get_permute_transform(view, 0)(x)]
    # re = []
    # for x, view in zip(xs, views):
    #     if x.ndim == 4:
    #         print(x.shape,view)
    #         #re.append(torch.rand(1, 5, 64, 64, 64).to('cuda:0'))
    #     else:
    #         transformed_x = get_permute_transform(view, 0)(x)
    #         re.append(transformed_x)
    # return re
    # else:
    #     re = [get_permute_transform(view, 0)(x) for x, view in zip(xs, views)]
    #     return re
def permute_inverse(xs: Sequence[torch.Tensor],
                    views: Sequence[PermuteType]):
    """Transforms data back to origin view."""
    # if xs[0].ndim == 4:
    #     xss=[]
    #     # xss: Sequence[torch.Tensor] = []
    #     # xss.append(xs)
    #     # xss.append(xs)
    #     # return [get_permute_transform(view, 0)(x) for x, view in zip(xss, views)]
    #     xss[0] = xs[0]+xs[1]
    #     xss[1] = xs[0] + xs[1]
    #     return [get_permute_transform(view, 0)(x) for x, view in zip(xss, views)]
    # else:
    return [get_permute_transform(view, 0)(x) for x, view in zip(xs, views)]

def permute_rand(
        x: torch.Tensor,
        num_samples: int = 2
) -> Tuple[Sequence[torch.Tensor], Sequence[PermuteType]]:
    """Samples different transforms of data."""
    num_permutes = len(view_transforms.permutation_transforms)
    if num_samples > num_permutes:
        raise ValueError('Duplicate samples.')
    view_dsts = np.random.permutation(num_permutes)[:num_samples].tolist()
    transformed_tensors = [get_permute_transform(0, view)(x) for view in view_dsts]
    #stacked_tensor1 = torch.stack(transformed_tensors).squeeze(2)
    #stacked_tensor1 = torch.cat(transformed_tensors,dim=0)

    #return stacked_tensor1, view_dsts
    return transformed_tensors, view_dsts
