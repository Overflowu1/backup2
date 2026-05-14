# import dicom2nifti
# # 多个dicom文件转化为3D nii文件
# # 'E:/COVID-19CTimageAnal/data'为存放dcm2d切片文件的目录
# original_dicom_directory = '/mnt/data/DATA/nnUNet_raw/metaltest/MedData'
# # '0007686236.nii'为要生成的nii文件名
# output_file = '0007686236.nii'
# dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)\



import time
import torch
from nnunetv2.net.YuNet.ARConv3D import ARConv3D

device = torch.device('cuda:1')

# 模拟stem输入: batch=2, in_ch=1, patch=128x128x128 (最坏情况)
x_full = torch.randn(2, 1,  128, 128, 128, device=device)
x_down = torch.randn(2, 32,  64,  64,  64, device=device)

# 测试全分辨率 (旧stem)
m_full = ARConv3D(1, 32, stride=1, padding=1).to(device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    out, _ = m_full(x_full)
torch.cuda.synchronize()
print(f"ARConv full-res: {(time.time()-t0)/10*1000:.1f} ms/iter")

# 测试半分辨率 (stage0, stride=2输入已是1/2)
m_down = ARConv3D(32, 64, stride=2, padding=1).to(device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    out, _ = m_down(x_down)
torch.cuda.synchronize()
print(f"ARConv half-res: {(time.time()-t0)/10*1000:.1f} ms/iter")