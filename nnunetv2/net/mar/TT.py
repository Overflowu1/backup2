import numpy as np
import torch
import pytest

# 假设以下模块已在同一目录或已安装
from WNET import WNet3D, Fconv_PCA_3D, OSCNetplus3D, NIfTIDataHandler, create_3d_mask, example_training_setup

# Helper to generate synthetic 3D NIfTI-like array
def make_synthetic_data(batch_size, channels, depth, height, width):
    return np.random.rand(batch_size, channels, depth, height, width).astype(np.float32)

@pytest.fixture(scope="module")
def simple_args():
    class Args:
        S = 3
        cdiv = 4
        num_rot = 4
        num_M = 8
        num_Q = 4
        etaM = 0.1
        etaX = 0.1
        sizeP = 5
        inP = 3
        padding = 2
        ifini = 0
        T = 2
    return Args()

def test_wnet3d_forward():
    b, c, d, h, w = 2, 1, 8, 16, 16
    ximg = torch.rand(b, 1, d, h, w)
    xma  = torch.rand(b, 1, d, h, w)
    wnet = WNet3D(d, h, w)
    weights = wnet(ximg, xma)
    # 输出 weight: (b, d,1,1,1,h,w)
    assert weights.shape == (b, d, 1, 1, 1, h, w)

def test_fconv_pca_3d_forward(simple_args):
    b, d, h, w = 2, 6, 8, 8
    sizeP = simple_args.sizeP
    input_vol = torch.rand(b, 1, d, h, w)
    ximg = torch.rand(b, 1, d, h, w)
    xma  = torch.rand(b, 1, d, h, w)
    fconv = Fconv_PCA_3D(sizeP=sizeP, inNum=1, outNum=simple_args.num_M,
                         tranNum=simple_args.num_rot, inP=simple_args.inP,
                         padding=simple_args.padding, ifIni=0, bias=True, Smooth=True, iniScale=1.0, cdiv=simple_args.cdiv)
    outf, filter_kernel = fconv(input_vol, ximg, xma)
    # 输出体积和 filter kernel 维度检查
    assert outf.shape == (b, 1, d, h, w)
    # filter shape: (b, outNum * tranNum, sizeP, sizeP, sizeP)
    assert filter_kernel.shape == (b, simple_args.num_M * simple_args.num_rot,
                                   sizeP, sizeP, sizeP)

def test_oscnnetplus3d_forward(simple_args):
    args = simple_args
    model = OSCNetplus3D(args)
    # synthetic data：batch + channel + d,h,w
    data = make_synthetic_data(batch_size=1, channels=1, depth=16, height=32, width=32)
    mask_np = create_3d_mask((16,32,32), mask_type='metal_streaks')
    vol = data[0,0]
    mask = mask_np
    vol_batch = NIfTIDataHandler.prepare_batch(vol)
    mask_batch = NIfTIDataHandler.prepare_batch(mask)
    # low-dose CT simulated
    # 合法 shape: (1,1, d,h,w)
    X0, ListX, ListA = model(mask_batch, vol_batch, mask_batch)
    # X0 形状检查
    assert isinstance(X0, torch.Tensor)
    assert X0.shape == (1,1,16,32,32)
    # ListX 长度等于 S+1
    assert len(ListX) == args.S + 1
    # 每个 ListX[i] 形状一致
    for X in ListX:
        assert X.shape == (1,1,16,32,32)
    # ListA 迭代次数少于或等于 S
    assert len(ListA) <= args.S

if __name__ == "__main__":
    pytest.main([__file__])
