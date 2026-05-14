import numpy as np
import nibabel as nib
import os
import SimpleITK as sitk
import glob

# img_name = r'E:\yiliaodata\CTPelvic\ipcai2021_dataset6_Anonymized\dataset6_CLINIC_0040_mask_4label.nii'
# img_name2 = r'D:\BaiduNetdiskDownload\CTxiugai\1\dataset6_CLINIC_0040_data2.nii'
img_name =r'/mnt/data/DATA/nnUNet_raw/Dataset123_fra/imagesTs/dataset6_CLINIC_0011_0000.nii.gz'
img_name2 =r'/mnt/data/DATA/nnUNet_raw/Dataset123_fra/labelsTs/dataset6_CLINIC_0011.nii.gz'
# count = 1
# for i in img_name:
ct = sitk.ReadImage(img_name)
modified_data = sitk.ReadImage(img_name2)
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
key2 = ct.GetMetaDataKeys()
# ct = sitk.GetArrayFromImage(ct)
# modified_data = sitk.GetImageFromArray(modified_data)
# modified_data.SetOrigin(o)
# modified_data.SetDirection(d)
# modified_data.SetSpacing(s)
print('指定坐标点的像素的RGB颜色值', modified_data.GetPixelID())
print('检查原点', modified_data.GetOrigin())
print('读取图像方向', modified_data.GetDirection())
print('读取图像数量', modified_data.GetSize())
print('读取图像的体素大小', modified_data.GetSpacing())
print('读取数据格式', modified_data.GetPixelIDTypeAsString())

modified_data.SetOrigin(o)
modified_data.SetDirection(d)
modified_data.SetSpacing(s)
key1 = modified_data.GetMetaDataKeys()
for current_key1, current_key2 in zip(key1, key2):
    # 获取 key2 对应的值
    print(ct.GetMetaData(current_key2),modified_data.GetMetaData(current_key1))
    value_to_assign = ct.GetMetaData(current_key2)

    # 将值赋给 key1
    modified_data.SetMetaData(key=current_key1, value=value_to_assign)


# modified_data = sitk.GetImageFromArray(modified_data)
sitk.WriteImage(modified_data, r'/mnt/data/DATA/nnUNet_raw/Dataset123_fra/labelsTs/dataset6_CLINIC_0011_data.nii.gz')
# count += 1
print("修改成功")
# modified_data = sitk.GetArrayFromImage(modified_data)
# sitk.WriteImage(modified_data, r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-07\26-77\1\i.nii')
# sitk.WriteImage(modified_data,  r'D:\work\shiyan31\dataset301CT_PELVIS_0002.nii')
# count += 1
# print("修改成功")
# modified_data.SetOrigin(o)
# modified_data.SetDirection(d)
# sitk.WriteImage(modified_data, r'D:\work\dataset301CT_PELVIS_0111.nii')
# print("修改成功2")


# import numpy as np
# import nibabel as nib
# import os
# import SimpleITK as sitk
# import glob
#
# # Specify the folders containing the Nii files
# folder_ct = r'E:\6m'
# folder_mask = r'D:\BaiduNetdiskDownload\CTxiugai\ff\mask2'
# output_folder = r'D:\BaiduNetdiskDownload\CTxiugai\ff\mask3'
# # Get the list of Nii files in each folder
# ct_files = glob.glob(os.path.join(folder_ct, '*.nii'))
# mask_files = glob.glob(os.path.join(folder_mask, '*.nii'))
#
# # Iterate over each pair of files
# for ct_file, mask_file in zip(ct_files, mask_files):
#     ct = sitk.ReadImage(ct_file)
#     modified_data = sitk.ReadImage(mask_file)
#
#     o = ct.GetOrigin()
#     d = ct.GetDirection()
#     s = ct.GetSpacing()
#
#     print('Processing CT file:', ct_file)
#     print('Processing Mask file:', mask_file)
#
#     # Your existing code for processing the images goes here
#
#     # Example: Setting origin, direction, and spacing for modified_data
#     modified_data.SetOrigin(o)
#     modified_data.SetDirection(d)
#     modified_data.SetSpacing(s)
#
#     # Rest of your code for metadata transfer goes here
#     # key1 = modified_data.GetMetaDataKeys()
#     # key2 = ct.GetMetaDataKeys()
#     #
#     # for current_key1, current_key2 in zip(key1, key2):
#     #     print(ct.GetMetaData(current_key2), modified_data.GetMetaData(current_key1))
#     #     value_to_assign = ct.GetMetaData(current_key2)
#     #     modified_data.SetMetaData(key=current_key1, value=value_to_assign)
#
#     # Save or do further processing with modified_data if needed
#     # For example, you can save the modified_data to a new Nii file
#     output_file = os.path.join(output_folder, f'modified_{os.path.basename(ct_file)}')
#     sitk.WriteImage(modified_data, output_file)
#
#     print('Finished processing and saved to:', output_file)

