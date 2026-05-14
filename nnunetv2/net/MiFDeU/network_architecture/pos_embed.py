# 2023.03.16-Changed for building NexToU
#            Harbin Institute of Technology (Shenzhen), <pcshi@stu.hit.edu.cn>

# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# modified from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

# --------------------------------------------------------
# relative position embedding
# References: https://arxiv.org/abs/2009.13658
# --------------------------------------------------------
def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos

def get_3d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height、width and depth
    return:
    pos_embed: [grid_size*grid_size*grid_size, grid_size*grid_size*grid_size]
    """
    pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height，width and depth
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_w, grid_h)  # here d goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, Dim/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, Dim/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, Dim)
    return emb

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # 修正后的逻辑：对称地分配维度，并确保每个维度都是偶数
    # Corrected logic to symmetrically distribute dimensions
    # and ensure each is an even number.
    d_dim = h_dim = w_dim = embed_dim // 3

    # 分配余数
    # Distribute the remainder
    remainder = embed_dim % 3
    if remainder == 1:
        d_dim += 1
    elif remainder == 2:
        d_dim += 1
        h_dim += 1

    # 确保所有维度都是偶数
    # Ensure all dimensions are even
    if d_dim % 2 != 0:
        d_dim -= 1
        h_dim += 1

    if h_dim % 2 != 0:
        h_dim -= 1
        w_dim += 1

    # 此时，d_dim 和 h_dim 都是偶数。
    # 因为 embed_dim 是偶数，所以 w_dim 也必然是偶数。
    # grid 索引的顺序对应于 np.meshgrid 的 (d, w, h)
    # At this point, d_dim and h_dim are even.
    # Since embed_dim is even, w_dim must also be even.
    # The order of grid indices corresponds to (d, w, h) from np.meshgrid
    emb_d = get_1d_sincos_pos_embed_from_grid(d_dim, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(w_dim, grid[1])
    emb_h = get_1d_sincos_pos_embed_from_grid(h_dim, grid[2])

    emb = np.concatenate([emb_d, emb_w, emb_h], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb