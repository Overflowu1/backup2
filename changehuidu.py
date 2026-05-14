# import numpy as np
# import nibabel as nib
#
# nii_path = r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-07\10-25\10\L.nii'
# src_img = nib.load(nii_path)
# src_data = src_img.get_fdata()
#
# modified_data = np.copy(src_data)
# mask_indexes = np.where(src_data == 1)
# modified_data[mask_indexes] += 0913
# print(modified_data)
# nii_img = nib.Nifti1Image(modified_data, src_img.affine, src_img.header)
# nib.save(nii_img, r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-07\10-25\10\L_2.nii')



# import nibabel as nib
# import numpy as np
#
# # 用你的NIfTI文件路径替换下面的路径
# nii_file_path = r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\18WANG-KAI-MING.nii'
#
# # 使用Nibabel加载NIfTI文件
# img = nib.load(nii_file_path)
#
# # 获取NIfTI文件中的像素值数据
# data = img.get_fdata()
#
# # 使用NumPy的unique函数来获取不同的像素值及其出现次数
# unique_values, counts = np.unique(data, return_counts=True)
#
# # 打印每个像素值及其出现次数
# for value, count in zip(unique_values, counts):
#     print(f"像素值 {value}: 出现次数 {count}")
#


# import nibabel as nib
# import numpy as np
#
# # 用你的NIfTI文件路径替换下面的路径
# nii_file_path =  r'E:\Dataset012_Frc3\labelsTr\fracture_0020_data.nii'
#
# # 使用Nibabel加载NIfTI文件
# img = nib.load(nii_file_path)
#
# # 获取NIfTI文件中的像素值数据
# data = img.get_fdata()
#
# # 使用NumPy的unique函数来获取不同的像素值及其出现次数
# unique_values, counts = np.unique(data, return_counts=True)
#
# # 打印每个像素值及其出现次数
# for value, count in zip(unique_values, counts):
#     print(f"像素值 {value}: 出现次数 {count}")



# import nibabel as nib
#
# # 用你的NIfTI文件路径替换下面的路径
# nii_file_path = r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\WANG-KAI-MING.nii'
#
# # 使用Nibabel加载NIfTI文件
# img = nib.load(nii_file_path)
#
# # 获取NIfTI文件中的像素值数据
# data = img.get_fdata()
#
# # 打印数据的形状和像素值
# print(f"数据形状：{data.shape}")
# print("像素值：")
# print(data)



import numpy as np
import nibabel as nib
import os
import SimpleITK as sitk
import glob

#img_name = glob.glob(r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-07\26-77\77\*.nii')
#img_name = glob.glob(r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-07\90\90\*.nii')
img_name ='/home/ps/lxq/gz2/YANG-DE-WEN.nii.gz'
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

mask_indexes1 = np.where(ct == 0)
mask_indexes2 = np.where(ct == 1)
mask_indexes3 = np.where(ct == 2)
mask_indexes4 = np.where(ct == 3)
mask_indexes5 = np.where(ct == 4)
modified_data = sitk.GetArrayFromImage(modified_data)
# input_value1 = int(input("请输入一个需要更改的灰度值："))
modified_data[mask_indexes1] = 0 #1391
# input_value2 = int(input("请输入一个需要更改的灰度值："))
modified_data[mask_indexes2] = 0 #2254
# input_value3 = int(input("请输入一个需要更改的灰度值："))
modified_data[mask_indexes3] = 2 #  1858
# input_value4 = int(input("请输入一个需要更改的灰度值："))
modified_data[mask_indexes4] = 3 #2561
#
modified_data[mask_indexes5] = 0 #2561


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
sitk.WriteImage(modified_data,  '/home/ps/lxq/gz3/YANG-DE-WEN.nii.gz'
)
# count += 1
print("修改成功")





import nibabel as nib
import numpy as np

# 用你的NIfTI文件路径替换下面的路径
nii_file_path =  '/home/ps/lxq/gz3/YANG-DE-WEN.nii.gz'
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