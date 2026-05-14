# import nibabel as nib
# import numpy as np
# import os
#
# def printnum(nii_file_path):
#     # 用你的NIfTI文件路径替换下面的路径
#     # 使用Nibabel加载NIfTI文件
#     img = nib.load(nii_file_path)
#
#     # 获取NIfTI文件中的像素值数据
#     data = img.get_fdata()
#
#     # 使用NumPy的unique函数来获取不同的像素值及其出现次数
#     unique_values, counts = np.unique(data, return_counts=True)
#
#     # 打印每个像素值及其出现次数
#     for value, count in zip(unique_values, counts):
#         print(f"像素值 {value}: 出现次数 {count}")
#
# # 设置输入路径
# # ct_path = 'original.nii.gz'         # 可选：原始CT图像路径
# # stage1_mask_path = '/mnt/data/DATA/zjyData/VISUAL/new/ThreeLabel/metric/CLINIC_1002_data_0020.nii.gz'  # 第一阶段骨结构分割
# stage1_mask_path = '/mnt/data/DATA/228/simsegr/ZhangYi_preCT_003.nii.gz'  # 第一阶段骨结构分割
# stage2_mask_path = '/mnt/data/DATA/228/guzhe/final/ZhangYi_preCT_003.nii.gz'   # 第二阶段骨折分割
# # stage2_mask_path = '/mnt/data/DATA/zjyData/VISUAL/18and2246/predict/seg/CLINIC_2246_data.nii.gz'   # 第二阶段骨折分割
#
#
# printnum(stage1_mask_path)
# printnum(stage2_mask_path)
# # 设置输出路径
#
# output_mask_path = '/mnt/data/DATA/228/guzhe/final/2stage/'
#
# # 读取mask文件
# stage1_nii = nib.load(stage1_mask_path)
# stage2_nii = nib.load(stage2_mask_path)
#
# stage1_mask = stage1_nii.get_fdata().astype(np.uint8)  # 骨结构：1=左髋，2=右髋，3=骶骨
# stage2_mask = stage2_nii.get_fdata().astype(np.uint8)  # 骨折区域：0=正常，1=骨折
#
# # 检查维度是否一致
# assert stage1_mask.shape == stage2_mask.shape, "两个mask尺寸不一致"
#
# # 创建输出mask（与原图尺寸一致）
# final_mask = np.zeros_like(stage1_mask, dtype=np.uint8)
#
# # 标签映射： (骨结构标签, 骨折状态) -> 输出标签
# label_map = {
#     (1, 0): 1,  # 骶骨主骨块
#     (1, 1): 2,  # 骶骨折块
#     (2, 0): 3,  # 左髋主骨块
#     (2, 1): 4,  # 左髋骨折块
#     (3, 0): 5,  # 右髋主骨块
#     (3, 1): 6,  # 右髋骨折块
# }
#
# # 遍历每个映射组合
# for (bone_label, fracture_flag), new_label in label_map.items():
#     mask = (stage1_mask == bone_label) & (stage2_mask == fracture_flag)
#     final_mask[mask] = new_label
#
# # 使用stage1的affine和header保持空间信息
# final_nii = nib.Nifti1Image(final_mask, affine=stage1_nii.affine, header=stage1_nii.header)
#
# # printnum(stage2_mask_path)
#
# # 保存结果
# nib.save(final_nii, output_mask_path)
# print(f"融合完成，结果保存到：{output_mask_path}")
#
# printnum(output_mask_path)
#
# # 这里插入膨胀清除周围体素的操作
# from scipy.ndimage import binary_dilation
#
# fracture_labels = [2, 4, 6]
# fracture_mask = np.isin(final_mask, fracture_labels)
# dilated_mask = binary_dilation(fracture_mask, iterations=2)
# final_mask[(dilated_mask) & (~fracture_mask)] = 0
#
# # 使用stage1的affine和header保持空间信息
# final_nii = nib.Nifti1Image(final_mask, affine=stage1_nii.affine, header=stage1_nii.header)
#
# # 保存结果
# nib.save(final_nii, output_mask_path)
# print(f"融合完成，结果保存到：{output_mask_path}")
#
# printnum(output_mask_path)
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation


def printnum(nii_file_path):
    img = nib.load(nii_file_path)
    data = img.get_fdata()
    unique_values, counts = np.unique(data, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"像素值 {value}: 出现次数 {count}")


stage1_mask_path = '/mnt/data/DATA/228/simsegr/ZhangYi_preCT_003.nii.gz'
stage2_mask_path = '/mnt/data/DATA/228/guzhe/final/ZhangYi_preCT_003.nii.gz'

printnum(stage1_mask_path)
printnum(stage2_mask_path)

# 输出目录
output_dir = '/mnt/data/DATA/228/guzhe/final/2stage'
os.makedirs(output_dir, exist_ok=True)

# 输出文件
output_mask_path = os.path.join(output_dir, 'ZhangYi_preCT_003_2stage.nii.gz')

# 读取 mask
stage1_nii = nib.load(stage1_mask_path)
stage2_nii = nib.load(stage2_mask_path)

stage1_mask = stage1_nii.get_fdata().astype(np.uint8)
stage2_mask = stage2_nii.get_fdata().astype(np.uint8)

assert stage1_mask.shape == stage2_mask.shape, "两个mask尺寸不一致"

final_mask = np.zeros_like(stage1_mask, dtype=np.uint8)

# 你这里注释和标签值描述有点不一致，我先保持你的原逻辑不变
label_map = {
    (1, 0): 1,
    (1, 1): 2,
    (2, 0): 3,
    (2, 1): 4,
    (3, 0): 5,
    (3, 1): 6,
}

for (bone_label, fracture_flag), new_label in label_map.items():
    mask = (stage1_mask == bone_label) & (stage2_mask == fracture_flag)
    final_mask[mask] = new_label

# 膨胀清除主骨块中靠近骨折块的区域
fracture_labels = [2, 4, 6]
fracture_mask = np.isin(final_mask, fracture_labels)
dilated_mask = binary_dilation(fracture_mask, iterations=2)

# 把“骨折块周围被膨胀覆盖，但本身不是骨折块”的体素清零
final_mask[dilated_mask & (~fracture_mask)] = 0

# 保存
final_nii = nib.Nifti1Image(final_mask, affine=stage1_nii.affine, header=stage1_nii.header)
nib.save(final_nii, output_mask_path)

print(f"融合完成，结果保存到：{output_mask_path}")
printnum(output_mask_path)