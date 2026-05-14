import SimpleITK as sitk
import numpy as np
#
#
def compute_hausdorff_95(image1, image2, labels):
    all_hd95_values = {}

    for label in labels:
        # Get the boundary of the two images for the specific label
        contour1 = sitk.LabelContour(image1 == label)
        contour2 = sitk.LabelContour(image2 == label)

        # Compute the distances from contour1 to contour2 and vice-versa
        distance_map_1_to_2 = sitk.Abs(
            sitk.SignedMaurerDistanceMap(contour1, squaredDistance=False, useImageSpacing=True))
        distance_map_2_to_1 = sitk.Abs(
            sitk.SignedMaurerDistanceMap(contour2, squaredDistance=False, useImageSpacing=True))

        # Mask the distance maps with the opposite contour
        distances_1_to_2 = sitk.Mask(distance_map_1_to_2, contour2)
        distances_2_to_1 = sitk.Mask(distance_map_2_to_1, contour1)

        # Get the distances and concatenate the arrays
        distances_1_to_2_array = sitk.GetArrayFromImage(distances_1_to_2).ravel()
        distances_2_to_1_array = sitk.GetArrayFromImage(distances_2_to_1).ravel()
        all_distances = np.concatenate((distances_1_to_2_array, distances_2_to_1_array))

        # Filter out zero distances
        all_distances = all_distances[all_distances != 0]

        # Calculate 95th percentile for the specific label
        hausdorff_95_label = np.percentile(all_distances, 95)
        all_hd95_values[label] = hausdorff_95_label

    return all_hd95_values
#
#
# # Example usage:
# prediction_path = r'D:\work\shiyan11\55\dataset6_CLINIC_0099_data.nii'
# ground_truth_path = r'D:\work\shiyan11\truth\dataset6_CLINIC_0099_mask_4label.nii'
#
# prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt16)
# ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt16)
#
# # Get unique labels in the ground truth
# unique_labels = np.unique(sitk.GetArrayFromImage(ground_truth))
#
# # Calculate HD95 for each label separately
# hd95_values_per_label = compute_hausdorff_95(prediction, ground_truth, labels=unique_labels)
#
# # Print results for each label
# for label, hd95 in hd95_values_per_label.items():
#     print(f'Label {label}: HD95={hd95:.10f}')

import os

def process_folder(folder_path):
    # 获取文件夹中的 NIfTI 文件列表
    nifti_files = [file for file in os.listdir(folder_path) if file.endswith('.nii.gz')]

    # 初始化字典以存储每个标签的 HD95 值列表
    hd95_values_per_label = {}

    for nifti_file in nifti_files:
        # 构建预测和真实结果的完整路径
        prediction_path = os.path.join(folder_path, nifti_file)
        ground_truth_path = os.path.join(ground_truth_folder, nifti_file)  # 根据需要调整

        # 读取预测和真实结果
        prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt16)
        ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt16)

        # 获取唯一的标签列表
        unique_labels = np.unique(sitk.GetArrayFromImage(ground_truth))

        # 计算每个标签的 HD95 值
        hd95_values = compute_hausdorff_95(prediction, ground_truth, labels=unique_labels)

        # 将 HD95 值添加到字典中
        for label, hd95 in hd95_values.items():
            if label not in hd95_values_per_label:
                hd95_values_per_label[label] = []
            hd95_values_per_label[label].append(hd95)

    # 计算每个标签的平均 HD95 值
    avg_hd95_values = {label: np.mean(values) for label, values in hd95_values_per_label.items()}

    # 打印每个标签的平均 HD95 值
    for label, avg_hd95 in avg_hd95_values.items():
        print(f'Label {label}: Average HD95={avg_hd95:.10f}')

# 指定包含 NIfTI 文件的文件夹
folder_path = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset011_4pelvis/11-11/testUnetr'
ground_truth_folder = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset011_4pelvis/11-11/gt'
# prediction_path = r'D:\work\shiyan11\55\dataset6_CLINIC_0099_data.nii'
# ground_truth_path = r'D:\work\shiyan11\truth\dataset6_CLINIC_0099_mask_4label.nii'
# 调用处理文件夹的函数folder_path = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset011_4pelvis/imagesTs2'
# ground_truth_folder = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset011_4pelvis/labelTs1'
process_folder(folder_path)
