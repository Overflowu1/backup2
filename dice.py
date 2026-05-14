# import nibabel as nib
# import numpy as np
#
# # 读取两个NIfTI文件
# image1 = nib.load(r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\18WANG-KAI-MING.nii')
# image2 = nib.load(r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\18WANG-KAI-MING2.nii')
#
# # 获取NIfTI数据数组
# data1 = image1.get_fdata()
# data2 = image2.get_fdata()
#
# # 阈值化为二进制掩模
# threshold = 0.5  # 适当的阈值
# mask1 = (data1 > threshold).astype(np.uint8)
# mask2 = (data2 > threshold).astype(np.uint8)
#
# # 计算交集和体积
# intersection = np.logical_and(mask1, mask2)
# volume1 = np.sum(mask1)
# volume2 = np.sum(mask2)
# intersection_volume = np.sum(intersection)
#
# # 计算Dice系数
# dice_coefficient = (2.0 * intersection_volume) / (volume1 + volume2)
#
# print(f'Dice系数: {dice_coefficient}')


# import nibabel as nib
# import numpy as np


# def dice_coefficient(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2)
#     return 2.0 * intersection.sum() / (mask1.sum() + mask2.sum())
#
#
# from scipy.ndimage import distance_transform_edt
#
#
# def hd91(mask1, mask2):
#     # 计算掩模的欧氏距离变换
#     dist1 = distance_transform_edt(mask1)
#     dist2 = distance_transform_edt(mask2)
#
#     # 找到距离变换中的最大值
#     max_dist1 = np.max(dist1)
#     max_dist2 = np.max(dist2)
#
#     # 计算HD91：取距离分布的91th百分位数
#     percentile1 = 91
#     hd911_value = np.percentile([max_dist1, max_dist2], percentile1)
#
#     return hd911_value


# # 读取两个NIfTI文件
# file1 = 'path_to_mask1.nii'  # 替换为第一个NIfTI文件的路径
# file2 = 'path_to_mask2.nii'  # 替换为第二个NIfTI文件的路径

# file1 = nib.load(r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\18WANG-KAI-MING.nii')
# file2 = nib.load(r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\WANG-KAI-MING222.nii')
#
# # 使用正确的文件路径来加载NIfTI文件
# mask1_data = file1.get_fdata()
# mask2_data = file2.get_fdata()
#
# # 阈值化为二进制掩模
# threshold = 0.5  # 适当的阈值
# mask1_binary = (mask1_data > threshold).astype(np.uint8)
# mask2_binary = (mask2_data > threshold).astype(np.uint8)
#
# # 计算Dice系数
# dice = dice_coefficient(mask1_binary, mask2_binary)
# print("Dice系数:", dice)
# hd91_value = hd91(mask1_binary, mask2_binary)
# print("HD91值:", hd91_value)

import SimpleITK as sitk
import numpy as np
# 用你的预测结果NIfTI文件路径替换下面的路径
# prediction_path = r'C:\Users\13605\Desktop\23\dataset6_CLINIC_0102_data.nii'
# prediction_path = r'D:\work\shiyan11\55\dataset6_CLINIC_0099_data.nii'
#
# # 用你的标签或真实结果NIfTI文件路径替换下面的路径
# # ground_truth_path = r'E:\yiliaodata\CTPelvic\CTPelvic1K_dataset7_mask\CLINIC_metal_0010_mask_4label.nii'
# ground_truth_path = r'D:\work\shiyan11\truth\dataset6_CLINIC_0099_mask_4label.nii'

# 使用SimpleITK加载预测结果和真实结果
# prediction = sitk.ReadImage(prediction_path,sitk.sitkUInt16)
# ground_truth = sitk.ReadImage(ground_truth_path,sitk.sitkUInt16)
#
# # Assuming prediction_array and ground_truth_array are 3D numpy arrays
# prediction_array = sitk.GetArrayFromImage(prediction)
# ground_truth_array = sitk.GetArrayFromImage(ground_truth)

# 计算Dice系数
# dice_filter = sitk.LabelOverlapMeasuresImageFilter()
# dice_filter.Execute(prediction, ground_truth)
#
# # 获取Dice系数值
# dice_coefficient = dice_filter.GetDiceCoefficient()
# print(f'Dice系数: {dice_coefficient}')


def hausdorff_distance(lT, lP):
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)

    # Ensure that both images have the same origin, spacing, and direction
    labelPred.SetOrigin(labelTrue.GetOrigin())
    labelPred.SetSpacing(labelTrue.GetSpacing())
    labelPred.SetDirection(labelTrue.GetDirection())

    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)

    return hausdorffcomputer.GetAverageHausdorffDistance()

# hd95_value = hausdorff_distance(ground_truth_array, prediction_array)
# print(f"HD95 value: {hd95_value}")
import numpy as np


def compute_hausdorff_95(image1, image2):
    # Get the boundary of the two images
    contour1 = sitk.LabelContour(image1)
    contour2 = sitk.LabelContour(image2)

    # Compute the distances from contour1 to contour2 and vice-versa
    distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(contour1, squaredDistance=False, useImageSpacing=True))
    distances_1_to_2 = sitk.Mask(distance_map, contour2)

    distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(contour2, squaredDistance=False, useImageSpacing=True))
    distances_2_to_1 = sitk.Mask(distance_map, contour1)

    # 获取两个距离图的数据,将两个数组合并
    all_distances = np.concatenate(
        (sitk.GetArrayFromImage(distances_1_to_2).ravel(), sitk.GetArrayFromImage(distances_2_to_1).ravel()))

    # 过滤掉为0的距离值
    all_distances = all_distances[all_distances != 0]

    # 计算95th百分位数
    hausdorff_95 = np.percentile(all_distances, 95)

    return hausdorff_95


# hausdorff_95_distance = compute_hausdorff_95(ground_truth,prediction)
# print(hausdorff_95_distance)


import os

def compute_metrics(prediction_path, ground_truth_path):
    # Load prediction and ground truth images
    prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt8)
    ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt8)

    # Assuming prediction_array and ground_truth_array are 3D numpy arrays
    prediction_array = sitk.GetArrayFromImage(prediction)
    ground_truth_array = sitk.GetArrayFromImage(ground_truth)

    # Calculate Dice coefficient
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter.Execute(prediction, ground_truth)
    dice_coefficient = dice_filter.GetDiceCoefficient()

    # Calculate Hausdorff distance and 95th percentile
    hausdorff_avg = hausdorff_distance(ground_truth_array, prediction_array)
    hausdorff_95 = compute_hausdorff_95(ground_truth, prediction)

    return dice_coefficient, hausdorff_avg, hausdorff_95

def process_folder(folder_path):
    # Get a list of NIfTI files in the folder
    nifti_files = [file for file in os.listdir(folder_path) if file.endswith('.nii.gz')]

    # Initialize lists to store metrics
    dice_coefficients = []
    hausdorff_avgs = []
    hausdorff_95s = []

    for nifti_file in nifti_files:
        # Construct full paths for prediction and ground truth
        prediction_path = os.path.join(folder_path, nifti_file)
        ground_truth_path = os.path.join(ground_truth_folder, nifti_file)  # Adjust this accordingly

        # prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt16)
        # ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt16)
        #
        # # Assuming prediction_array and ground_truth_array are 3D numpy arrays
        # prediction_array = sitk.GetArrayFromImage(prediction)
        # ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        # Compute metrics for the current pair of files
        dice, hausdorff_avg, hausdorff_95 = compute_metrics(prediction_path, ground_truth_path)

        # Append metrics to lists
        dice_coefficients.append(dice)
        hausdorff_avgs.append(hausdorff_avg)
        hausdorff_95s.append(hausdorff_95)

    # Calculate average values
    avg_dice = np.mean(dice_coefficients)
    avg_hausdorff_avg = np.mean(hausdorff_avgs)
    avg_hausdorff_95 = np.mean(hausdorff_95s)

    print(f'Average Dice coefficient: {avg_dice}')
    print(f'Average Hausdorff distance (avg): {avg_hausdorff_avg}')
    print(f'Average Hausdorff distance (95th percentile): {avg_hausdorff_95}')

folder_path = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset031_301shuzhong/31-11/testsegresnet'
ground_truth_folder = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset031_301shuzhong/31-11/gt'
# Call the function to process the folder
process_folder(folder_path)
process_folder(folder_path)

# 计算Hausdorff距离
# hausdorff_filter = sitk.HausdorffDistanceImageFilter()
# hausdorff_filter.Execute(prediction, ground_truth)
#
# # 获取Hausdorff距离的值
# distances = hausdorff_filter.GetHausdorffDistances()
#
# # 将距离值排序
# sorted_distances = np.sort(distances)

# 计算HD91，即第91百分位的距离值
# percentile = 91
# index = int((percentile / 100.0) * len(sorted_distances))
# hd915 = hd91(prediction_array,ground_truth_array)

# print(f'HD91: {hd915}')
# print(f'Dice系数: {dice}')

#
# def calculate_hd91(prediction_path, ground_truth_path):
#     # 使用SimpleITK加载预测结果和真实结果
#     prediction = sitk.ReadImage(prediction_path)
#     ground_truth = sitk.ReadImage(ground_truth_path)
#
#     # 将预测结果和真实结果转换为NumPy数组
#     prediction_array = sitk.GetArrayFromImage(prediction)
#     ground_truth_array = sitk.GetArrayFromImage(ground_truth)
#
#     # 计算Hausdorff距离
#     hausdorff_filter = sitk.HausdorffDistanceImageFilter()
#     hausdorff_filter.Execute(prediction, ground_truth)
#
#     # 获取Hausdorff距离的值
#     distances = hausdorff_filter.GetHausdorffDistances()
#
#     # 将距离值排序
#     sorted_distances = np.sort(distances)
#
#     # 计算HD91，即第91百分位的距离值
#     percentile = 91
#     index = int((percentile / 100.0) * len(sorted_distances))
#     hd91 = sorted_distances[index]
#
#     return hd91
#
# # 使用示例：
# prediction_path = 'path_to_your_prediction.nii'
# ground_truth_path = 'path_to_your_ground_truth.nii'
# hd91 = calculate_hd91(prediction_path, ground_truth_path)
# print(f'HD91: {hd91}')
#
#
#
# import SimpleITK as sitk
# import numpy as np
# from scipy.spatial.distance import directed_hausdorff
#
# def calculate_hd91(prediction_path, ground_truth_path):
#     # 使用SimpleITK加载预测结果和真实结果
#     prediction = sitk.ReadImage(prediction_path)
#     ground_truth = sitk.ReadImage(ground_truth_path)
#
#     # 将预测结果和真实结果转换为NumPy数组
#     prediction_array = sitk.GetArrayFromImage(prediction)
#     ground_truth_array = sitk.GetArrayFromImage(ground_truth)
#
#     # 计算HD91，即第91百分位的Hausdorff距离
#     hd_distances = []
#     for i in range(len(prediction_array)):
#         # 计算每个切片的Hausdorff距离
#         hd_distance = directed_hausdorff(prediction_array[i], ground_truth_array[i])[0]
#         hd_distances.append(hd_distance)
#
#     # 将距离值排序
#     sorted_distances = np.sort(hd_distances)
#
#     # 计算HD91，即第91百分位的距离值
#     percentile = 91
#     index = int((percentile / 100.0) * len(sorted_distances))
#     hd91 = sorted_distances[index]
#
#     return hd91
#
# # 使用示例：
# prediction_path = r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\WANG-KAI-MING222.nii'
# ground_truth_path = r'D:\rep\WeChat Files\wxid_npcvpgn1m6pj22\FileStorage\File\2023-09\18WANG-KAI-MING.nii'
# hd91 = calculate_hd91(prediction_path, ground_truth_path)
# print(f'HD91: {hd91}')
