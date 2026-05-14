

import numpy as np
import nibabel as nib
import os
import SimpleITK as sitk
import glob
img_name ='/mnt/data/DATA/zjyData/VISUAL/18and2246/3fenlei/2246.nii.gz'
#img_name2 = r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-10\YAN-GUI-XIN5.nii'
# count = 1
# for i in img_name:
ct = sitk.ReadImage(img_name)
modified_data = ct
# print("读入的文件为:" + i[-6:])

# ct = sitk.GetImageFromArray(ct)
o = ct.GetOrigin()
d = ct.GetDirection()
s = ct.GetSpacing()
print('指定坐标点的像素的RGB颜色值', ct.GetPixelID())
print('检查原点', ct.GetOrigin())
print('读取图像方向', ct.GetDirection())
print('读取图像数量', ct.GetSize())
print('读取图像的体素大小', ct.GetSpacing())
print('读取数据格式', ct.GetPixelIDTypeAsString())

ct = sitk.GetArrayFromImage(ct)

# mask_indexes1 = np.where(ct == -1024)

# mask_indexes2 = np.where(ct == -558)
# mask_indexes3= np.where(ct == 200)
# mask_indexes4 = np.where(ct == 1379)
mask_indexes1 = np.where(ct == 2)
mask_indexes2 = np.where(ct == 3)






modified_data = sitk.GetArrayFromImage(modified_data)

modified_data[mask_indexes1] = 3
modified_data[mask_indexes2] = 2
# modified_data[mask_indexes3] = 3
# modified_data[mask_indexes4] = 2







modified_data = sitk.GetImageFromArray(modified_data)
modified_data.SetOrigin(o)
modified_data.SetDirection(d)
modified_data.SetSpacing(s)
print('指定坐标点的像素的RGB颜色值', modified_data.GetPixelID())
print('检查原点', modified_data.GetOrigin())
print('读取图像方向', modified_data.GetDirection())
print('读取图像数量', modified_data.GetSize())
print('读取图像的体素大小', modified_data.GetSpacing())
print('读取数据格式', modified_data.GetPixelIDTypeAsString())
#modified_data = sitk.GetArrayFromImage(modified_data)
# sitk.WriteImage(modified_data, r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-07\26-77\38\i.nii')
sitk.WriteImage(modified_data,  '/mnt/data/DATA/zjyData/VISUAL/18and2246/3fenlei/2246.nii.gz'
)
# count += 1
print("修改成功")




import nibabel as nib
import numpy as np

# 用你的NIfTI文件路径替换下面的路径
# nii_file_path =  '/mnt/data/DATA/nnUNet_raw/Dataset031_301shuzhong/31-11/gt/dataset6_CLINIC_0103_data.nii.gz'
nii_file_path = '/mnt/data/DATA/zjyData/VISUAL/18and2246/3fenlei/2246.nii.gz'
# nii_file_path = '/mnt/data/DATA/zjyData/VISUAL/new/guzheLabel/CLINIC_2252.nii.gz'

# folder_path = r'D:\work\shiyan31\25'
# ground_truth_folder = r'D:\work\shiyan31\truth'
# 使用Nibabel加载NIfTI文件
img = nib.load(nii_file_path)

# 获取NIfTI文件中的像素值数据
data = img.get_fdata()

# 使用NumPy的unique函数来获取不同的像素值及其出现次数
unique_values, counts = np.unique(data, return_counts=True)

# 打印每个像素值及其出现次数
for value, count in zip(unique_values, counts):
    print(f"像素值 {value}: 出现次数 {count}")