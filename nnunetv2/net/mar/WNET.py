# -*- coding: utf-8 -*-
"""
3D Extension for CT Artifact Removal - NIfTI Data Support
Extended from 2D OSCNet+ to handle 3D volumetric data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import nibabel as nib
from scipy import ndimage


class WNet3D(nn.Module):
    """3D Weight Network for adaptive filtering"""

    def __init__(self, d, h, w):
        super(WNet3D, self).__init__()
        self.d = d
        self.h = h
        self.w = w
        self.conv1 = nn.Conv3d(2, 6, 3, 1, 1)
        self.conv2 = nn.Conv3d(6, 6, 3, 1, 1)
        self.conv3 = nn.Conv3d(6, 6, 3, 1, 1)
        required = self.d * self.h * self.w
        self.f1 = nn.Linear(in_features=6, out_features=required,bias=True)


    def forward(self, Ximg, Xma):
        batchSize = Ximg.size()[0]
        depth, height, width = Ximg.size()[2], Ximg.size()[3], Ximg.size()[4]
        output1 = self.conv1(torch.cat((Ximg, Xma), dim=1))
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = F.avg_pool3d(output3, (depth, height, width))
        output8 = F.relu(self.f1(output4.reshape(batchSize, -1)))
        output8 = output8.reshape(-1, self.d, self.h, self.w)
        norm = torch.norm(output8, 2, dim=[2, 3])  # norm across h,w dimensions
        norm_re = norm.unsqueeze(dim=2).unsqueeze(dim=3).expand(-1, -1, self.h, self.w)
        output8 = torch.div(output8, norm_re + 1e-6)
        weight = output8.reshape(-1, self.d, 1, 1, 1, self.h, self.w)
        return weight


class Fconv_PCA_3D(nn.Module):
    """3D PCA-based Frequency Convolution"""

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0, cdiv=1):
        super(Fconv_PCA_3D, self).__init__()
        if inP is None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.cdiv = cdiv

        # Generate 3D basis functions
        Basis, Rank, weight = GetBasis_PCA_3D(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)
        self.wnet = WNet3D(outNum, outNum, Basis.size(4))  # Adjusted for 3D

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum

        iniw = Getini_reg_3D(Basis.size(4), inNum, outNum, self.expand, weight) * iniScale

        if padding is None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1, inNum, 1, 1, 1), requires_grad=bias)

    def forward(self, input, Ximg, Xma):
        batchSize, d, h, w = Ximg.size()[0], Ximg.size()[2], Ximg.size()[3], Ximg.size()[4]
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand

        weights = self.wnet(Ximg, Xma)
        Basis_per = self.Basis.permute(4, 0, 1, 2, 3).reshape(self.Basis.size(4), -1)
        tempW = torch.matmul(weights.reshape(batchSize, -1), Basis_per).reshape(
            batchSize, outNum, 1, 1, 1, self.sizeP, self.sizeP, self.sizeP, self.tranNum
        ).permute(0, 1, 8, 2, 3, 4, 5, 6, 7)

        Num = tranNum // expand
        tempWList = []
        for i in range(expand):
            if i == 0:
                tempWList.append(tempW[:, :, i * Num:(i + 1) * Num, :, :, :, :, :, :])
            else:
                # Handle circular shift for 3D
                shifted = torch.cat([
                    tempW[:, :, i * Num:(i + 1) * Num, :, :, -i:, :, :, :],
                    tempW[:, :, i * Num:(i + 1) * Num, :, :, :-i, :, :, :]
                ], dim=5)
                tempWList.append(shifted)

        tempW = torch.cat(tempWList, dim=2)
        _filter = tempW.reshape([batchSize, outNum * tranNum, self.sizeP, self.sizeP, self.sizeP])
        _bias = self.c.repeat([1, 1, inNum, 1, 1]).reshape([1, inNum * self.expand, 1, 1, 1])

        M_re = input.reshape(1, -1, d, h, w)
        output = F.conv3d(M_re, _filter / self.cdiv, groups=batchSize, stride=1, padding=self.padding)
        outf = output + _bias
        outf_re = outf.reshape(batchSize, 1, d, h, w)
        return outf_re, _filter


def GetBasis_PCA_3D(sizeP, tranNum=8, inP=None, Smooth=True):
    """Generate 3D PCA basis functions"""
    if inP is None:
        inP = sizeP

    inX, inY, inZ, Mask = MaskC_3D(sizeP)
    X0 = np.expand_dims(inX, 3)
    Y0 = np.expand_dims(inY, 3)
    Z0 = np.expand_dims(inZ, 3)
    Mask = np.expand_dims(Mask, 3)

    # Generate rotation angles for 3D
    theta_xy = np.arange(tranNum) / tranNum * 2 * np.pi
    theta_xz = np.arange(tranNum) / tranNum * np.pi  # Different rotation axis
    theta_yz = np.arange(tranNum) / tranNum * np.pi

    theta_xy = np.expand_dims(np.expand_dims(np.expand_dims(theta_xy, axis=0), axis=0), axis=0)
    theta_xz = np.expand_dims(np.expand_dims(np.expand_dims(theta_xz, axis=0), axis=0), axis=0)
    theta_yz = np.expand_dims(np.expand_dims(np.expand_dims(theta_yz, axis=0), axis=0), axis=0)

    # 3D rotations
    X = np.cos(theta_xy) * X0 - np.sin(theta_xy) * Y0
    Y = np.cos(theta_xy) * Y0 + np.sin(theta_xy) * X0
    Z = Z0  # Keep Z unchanged for xy rotation

    X = np.expand_dims(np.expand_dims(np.expand_dims(X, 4), 5), 6)
    Y = np.expand_dims(np.expand_dims(np.expand_dims(Y, 4), 5), 6)
    Z = np.expand_dims(np.expand_dims(np.expand_dims(Z, 4), 5), 6)

    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, 1, inP, 1, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, 1, inP, 1])
    m = np.reshape(np.arange(inP), [1, 1, 1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y + (m - inP * (m > p)) * v * Z)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y + (m - inP * (m > p)) * v * Z)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, sizeP, tranNum, inP * inP * inP]) * np.expand_dims(Mask, 4)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, sizeP, tranNum, inP * inP * inP]) * np.expand_dims(Mask, 4)

    BasisC = np.reshape(BasisC, [sizeP * sizeP * sizeP * tranNum, inP * inP * inP])
    BasisS = np.reshape(BasisS, [sizeP * sizeP * sizeP * tranNum, inP * inP * inP])

    BasisR = np.concatenate((BasisC, BasisS), axis=1)

    U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))

    Rank = np.sum(S > 0.0001)
    BasisR = np.matmul(np.matmul(BasisR, U[:, :Rank]), np.diag(1 / np.sqrt(S[:Rank] + 0.0000000001)))
    BasisR = np.reshape(BasisR, [sizeP, sizeP, sizeP, tranNum, Rank])

    temp = np.reshape(BasisR, [sizeP * sizeP * sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis=0) ** 2, axis=0) + np.std(np.sum(temp ** 2 * sizeP * sizeP * sizeP, axis=0),
                                                              axis=0)) / np.mean(
        np.sum(temp, axis=0) ** 2 + np.sum(temp ** 2 * sizeP * sizeP * sizeP, axis=0), axis=0)

    Trod = 1
    Ind = var < Trod
    Rank = np.sum(Ind)
    Weight = 1 / np.maximum(var, 0.04) / 125  # Adjusted for 3D (was /25 for 2D)

    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(Weight, 0), 0), 0), 0) * BasisR

    return torch.FloatTensor(BasisR), Rank, Weight


def MaskC_3D(SizeP):
    """Generate 3D circular mask"""
    p = (SizeP - 1) / 2
    x = np.arange(-p, p + 1) / p
    X, Y, Z = np.meshgrid(x, x, x)
    C = X ** 2 + Y ** 2 + Z ** 2

    Mask = np.ones([SizeP, SizeP, SizeP])
    Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)

    return X, Y, Z, Mask


def Getini_reg_3D(nNum, inNum, outNum, expand, weight=1):
    """3D weight initialization"""
    A = (np.random.rand(outNum, inNum, expand, nNum) - 0.5) * 2 * 2.4495 / np.sqrt((inNum) * nNum) * np.expand_dims(
        np.expand_dims(np.expand_dims(weight, axis=0), axis=0), axis=0)
    return torch.FloatTensor(A)


class OSCNetplus3D(nn.Module):
    """3D OSCNet+ for volumetric CT artifact removal"""

    def __init__(self, args):
        super(OSCNetplus3D, self).__init__()
        self.S = args.S
        self.iter = self.S - 1
        self.cdiv = args.cdiv
        self.num_rot = args.num_rot
        self.num_M = args.num_M
        self.num_Q = args.num_Q

        # Stepsize parameters
        self.etaM = torch.Tensor([args.etaM])
        self.etaX = torch.Tensor([args.etaX])
        self.etaM_S = self.make_eta(self.iter, self.etaM)
        self.etaX_S = self.make_eta(self.S, self.etaX)

        # 3D kernels
        kernel_3d = torch.randn(1, self.num_M, 9, 9, 9) * 0.01  # 3D kernel
        allrot_kernel = kernel_3d.repeat(1, self.num_rot, 1, 1, 1)
        self.C0 = nn.Parameter(data=allrot_kernel, requires_grad=True)

        # 3D filter parameterization
        self.fcnn = Fconv_PCA_3D(sizeP=args.sizeP, inNum=1, outNum=self.num_M, tranNum=args.num_rot,
                                 inP=args.inP, padding=args.padding, ifIni=args.ifini, bias=True,
                                 Smooth=True, iniScale=1.0, cdiv=self.cdiv)

        # 3D filter for initialization
        filter_3d = torch.ones(1, 1, 3, 3, 3) / 27  # 3D averaging filter
        self.C_q_const = filter_3d.expand(self.num_Q, 1, -1, -1, -1)
        self.C_q = nn.Parameter(self.C_q_const, requires_grad=True)

        # ProxNets
        self.proxNet_X_0 = Xnet3D(args)
        self.proxNet_X_S = self.make_Xnet(self.S, args)
        self.proxNet_M_S = self.make_Mnet(self.S, args)
        self.proxNet_X_last_layer = Xnet3D(args)

        # Sparsity parameter
        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def make_Xnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Xnet3D(args))
        return nn.Sequential(*layers)

    def make_Mnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Mnet3D(args))
        return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def forward(self, CT_ma, LIct, Mask):
        batchSize, d, h, w = CT_ma.size()[0], CT_ma.size()[2], CT_ma.size()[3], CT_ma.size()[4]

        ListX = []
        ListA = []
        input = CT_ma

        # Initialization
        Q00 = F.conv3d(LIct, self.C_q, stride=1, padding=1)
        input_ini = torch.cat((LIct, Q00), dim=1)
        XQ_ini = self.proxNet_X_0(input_ini)
        X0 = XQ_ini[:, :1, :, :, :]
        Q0 = XQ_ini[:, 1:, :, :, :]

        # First iteration
        A_hat = Mask * (input - X0)
        A_hat_cut = F.relu(A_hat - self.tau)
        Epsilon = F.conv_transpose3d(A_hat_cut, self.C0 / 10, stride=1, padding=4)
        M1 = self.proxNet_M_S[0](Epsilon)
        A, C = self.fcnn(M1, X0, CT_ma)

        A_hat = input - A.reshape(batchSize, 1, d, h, w)
        X_mid = (1 - self.etaX_S[0] * Mask / 10) * X0 + self.etaX_S[0] * Mask / 10 * A_hat
        input_concat = torch.cat((X_mid, Q0), dim=1)
        XQ = self.proxNet_X_S[0](input_concat)
        X1 = XQ[:, :1, :, :, :]
        Q1 = XQ[:, 1:, :, :, :]

        ListX.append(X1)
        ListA.append(A)

        X = X1
        Q = Q1
        M = M1

        # Iterative updates
        for i in range(self.iter):
            # M-net update
            A_hat = Mask * (input - X)
            Epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose3d(
                (Mask * A - A_hat).reshape(1, -1, d, h, w),
                C / self.cdiv, groups=batchSize, stride=1, padding=4
            ).reshape(batchSize, -1, d, h, w)
            M = self.proxNet_M_S[i + 1](M - Epsilon)

            # X-net update
            A = F.conv3d(M.reshape(1, -1, d, h, w), C / self.cdiv,
                         groups=batchSize, stride=1, padding=4).reshape(batchSize, 1, d, h, w)
            ListA.append(A)
            X_hat = input - A
            X_mid = (1 - self.etaX_S[i + 1, :] * Mask / 10) * X + self.etaX_S[i + 1, :] * Mask / 10 * X_hat
            input_concat = torch.cat((X_mid, Q), dim=1)
            XQ = self.proxNet_X_S[i + 1](input_concat)
            X = XQ[:, :1, :, :, :]
            Q = XQ[:, 1:, :, :, :]
            A, C = self.fcnn(M, X, CT_ma)
            ListX.append(X)

        # Final adjustment
        XQ_adjust = self.proxNet_X_last_layer(XQ)
        X = XQ_adjust[:, :1, :, :, :]
        ListX.append(X)

        return X0, ListX, ListA


class Mnet3D(nn.Module):
    """3D M-network (artifact estimation network)"""

    def __init__(self, args):
        super(Mnet3D, self).__init__()
        self.channels = args.num_M * args.num_rot
        self.T = args.T
        self.layer = self.make_resblock(self.T)
        self.tau0 = torch.Tensor([0.5])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1,
                                                                                                              self.channels,
                                                                                                              -1, -1,
                                                                                                              -1)
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.BatchNorm3d(self.channels),
                    nn.ReLU(),
                    nn.Conv3d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.BatchNorm3d(self.channels),
                ))
        return nn.Sequential(*layers)

    def forward(self, input):
        M = input
        for i in range(self.T):
            M = F.relu(M + self.layer[i](M))
        M = F.relu(M - self.tau)
        return M


class Xnet3D(nn.Module):
    """3D X-network (image reconstruction network)"""

    def __init__(self, args):
        super(Xnet3D, self).__init__()
        self.channels = args.num_Q + 1
        self.T = args.T
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Conv3d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm3d(self.channels),
                nn.ReLU(),
                nn.Conv3d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm3d(self.channels),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            X = F.relu(X + self.layer[i](X))
        return X


# Utility functions for NIfTI data handling
class NIfTIDataHandler:
    """Utility class for handling 3D NIfTI data"""

    @staticmethod
    def load_nifti(filepath):
        """Load NIfTI file and return data as numpy array"""
        nii = nib.load(filepath)
        data = nii.get_fdata()
        return data, nii.affine, nii.header

    @staticmethod
    def save_nifti(data, filepath, affine=None, header=None):
        """Save numpy array as NIfTI file"""
        if affine is None:
            affine = np.eye(4)
        nii = nib.Nifti1Image(data, affine, header)
        nib.save(nii, filepath)

    @staticmethod
    def normalize_data(data, method='minmax'):
        """Normalize 3D data"""
        if method == 'minmax':
            data_min, data_max = data.min(), data.max()
            return (data - data_min) / (data_max - data_min), (data_min, data_max)
        elif method == 'zscore':
            mean, std = data.mean(), data.std()
            return (data - mean) / std, (mean, std)
        return data, None

    @staticmethod
    def denormalize_data(data, normalization_params, method='minmax'):
        """Denormalize data using stored parameters"""
        if method == 'minmax':
            data_min, data_max = normalization_params
            return data * (data_max - data_min) + data_min
        elif method == 'zscore':
            mean, std = normalization_params
            return data * std + mean
        return data

    @staticmethod
    def prepare_batch(data, batch_size=1):
        """Prepare 3D data for batch processing"""
        if len(data.shape) == 3:
            data = data[np.newaxis, np.newaxis, ...]  # Add batch and channel dimensions
        elif len(data.shape) == 4:
            data = data[np.newaxis, ...]  # Add batch dimension
        return torch.FloatTensor(data)


# Example usage and training utilities
def create_3d_mask(shape, mask_type='metal_streaks'):
    """Create 3D mask for metal artifacts"""
    d, h, w = shape
    mask = np.ones((d, h, w))

    if mask_type == 'metal_streaks':
        # Simulate metal streak artifacts in 3D
        center_d, center_h, center_w = d // 2, h // 2, w // 2

        # Create streak patterns
        for i in range(d):
            # Horizontal streaks
            mask[i, center_h - 5:center_h + 5, :] = 0
            # Vertical streaks  
            mask[i, :, center_w - 5:center_w + 5] = 0
            # Diagonal streaks
            for j in range(min(h, w)):
                if 0 <= center_h - 25 + j < h and 0 <= center_w - 25 + j < w:
                    mask[i, center_h - 25 + j:center_h - 25 + j + 3, center_w - 25 + j:center_w - 25 + j + 3] = 0

    return mask


def example_training_setup():
    """Example setup for training with 3D NIfTI data"""

    class Args:
        def __init__(self):
            self.S = 10  # Number of stages
            self.cdiv = 8  # Filter control parameter
            self.num_rot = 8  # Number of rotations
            self.num_M = 32  # Feature maps for M-net
            self.num_Q = 8  # Feature maps for Q
            self.etaM = 0.1  # Learning rate for M
            self.etaX = 0.1  # Learning rate for X
            self.sizeP = 9  # Filter size
            self.inP = 5  # Input size for basis
            self.padding = 4  # Padding
            self.ifini = 0  # Initialization flag
            self.T = 5  # ResBlock iterations

    args = Args()
    model = OSCNetplus3D(args)

    print("3D OSCNet+ model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model, args


# Memory optimization utilities
class VolumeProcessor:
    """Process large 3D volumes in patches to manage memory"""

    def __init__(self, patch_size=(64, 64, 64), overlap=8):
        self.patch_size = patch_size
        self.overlap = overlap

    def extract_patches(self, volume):
        """Extract overlapping patches from 3D volume"""
        d, h, w = volume.shape
        pd, ph, pw = self.patch_size

        patches = []
        positions = []

        for z in range(0, d - pd + 1, pd - self.overlap):
            for y in range(0, h - ph + 1, ph - self.overlap):
                for x in range(0, w - pw + 1, pw - self.overlap):
                    z_end = min(z + pd, d)
                    y_end = min(y + ph, h)
                    x_end = min(x + pw, w)

                    patch = volume[z:z_end, y:y_end, x:x_end]
                    patches.append(patch)
                    positions.append((z, y, x, z_end, y_end, x_end))

        return patches, positions

    def reconstruct_volume(self, patches, positions, original_shape):
        """Reconstruct volume from processed patches"""
        volume = np.zeros(original_shape)
        weight_map = np.zeros(original_shape)

        for patch, (z, y, x, z_end, y_end, x_end) in zip(patches, positions):
            volume[z:z_end, y:y_end, x:x_end] += patch
            weight_map[z:z_end, y:y_end, x:x_end] += 1

        # Average overlapping regions
        volume = np.divide(volume, weight_map, out=np.zeros_like(volume), where=weight_map != 0)
        return volume