import SimpleITK as sitk
import numpy as np


def compute_dice_coefficient(label_true, label_pred):
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()

    dice_coefficients = []

    for label in range(1, np.max(label_true) + 1):
        label_true_binary = label_true == label
        label_pred_binary = label_pred == label

        # Convert boolean array to integer array
        label_true_binary = label_true_binary.astype(np.uint8)
        label_pred_binary = label_pred_binary.astype(np.uint8)

        # Convert to SimpleITK images
        label_true_binary = sitk.GetImageFromArray(label_true_binary)
        label_pred_binary = sitk.GetImageFromArray(label_pred_binary)

        # Explicitly set the pixel type
        label_true_binary = sitk.Cast(label_true_binary, sitk.sitkUInt8)
        label_pred_binary = sitk.Cast(label_pred_binary, sitk.sitkUInt8)

        dice_filter.Execute(label_true_binary, label_pred_binary)
        dice_coefficients.append(dice_filter.GetDiceCoefficient())

    return dice_coefficients
#
def calculate_hausdorff_metrics(label_true, label_pred):
    label_true_binary = sitk.GetImageFromArray((label_true > 0.5).astype(np.uint8))
    label_pred_binary = sitk.GetImageFromArray((label_pred > 0.5).astype(np.uint8))

    hausdorff_computer = sitk.HausdorffDistanceImageFilter()
    hausdorff_computer.Execute(label_true_binary, label_pred_binary)

    # Compute HD95 without using compute_hausdorff_95
    contour1 = sitk.LabelContour(label_true_binary)
    contour2 = sitk.LabelContour(label_pred_binary)

    distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(contour1, squaredDistance=False, useImageSpacing=True))
    distances_1_to_2 = sitk.Mask(distance_map, contour2)

    distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(contour2, squaredDistance=False, useImageSpacing=True))
    distances_2_to_1 = sitk.Mask(distance_map, contour1)

    all_distances = np.concatenate(
        (sitk.GetArrayFromImage(distances_1_to_2).ravel(), sitk.GetArrayFromImage(distances_2_to_1).ravel()))

    all_distances = all_distances[all_distances != 0]

    hausdorff_95 = np.percentile(all_distances, 95)

    return hausdorff_computer.GetAverageHausdorffDistance(), hausdorff_95

def compute_metrics_for_labels(label_true, label_pred):
    metrics_per_label = []

    for label in range(1, np.max(label_true) + 1):
        label_true_binary = label_true == label
        label_pred_binary = label_pred == label

        hausdorff_distance, hausdorff_95 = calculate_hausdorff_metrics(label_true_binary, label_pred_binary)
        metrics_per_label.append((hausdorff_distance, hausdorff_95))

    return metrics_per_label

def compute_hausdorff_95(image1, image2, labels):
    all_hd95_values = {}

    for label in labels:
        # Get the boundary of the two images for the specific label
        contour1 = sitk.LabelContour(image1 == label)
        contour2 = sitk.LabelContour(image2 == label)

        # Compute the distances from contour1 to contour2 and vice-versa
        distance_map_1_to_2 = sitk.Abs(sitk.SignedMaurerDistanceMap(contour1, squaredDistance=False, useImageSpacing=True))
        distance_map_2_to_1 = sitk.Abs(sitk.SignedMaurerDistanceMap(contour2, squaredDistance=False, useImageSpacing=True))

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


# # Replace with your file paths
# prediction_path = r'D:\work\shiyan11\55\dataset6_CLINIC_0099_data.nii'
# ground_truth_path = r'D:\work\shiyan11\truth\dataset6_CLINIC_0099_mask_4label.nii'
#
# prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt16)
# ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt16)
#
# prediction_array = sitk.GetArrayFromImage(prediction)
# ground_truth_array = sitk.GetArrayFromImage(ground_truth)
#
# # Calculate metrics for each label
# dice_coefficients = compute_dice_coefficient(ground_truth_array, prediction_array)
# metrics_per_label = compute_metrics_for_labels(ground_truth_array, prediction_array)
# # Get unique labels in the ground truth
# unique_labels = np.unique(sitk.GetArrayFromImage(ground_truth))
# # Calculate HD95 for each label separately
# hd95_values_per_label = compute_hausdorff_95(prediction, ground_truth, labels=unique_labels)

# for label,hd in metrics_per_label:
#     print(f'Label {label}: HD={hd:.10f}')

# Print results for each label
# for label, hd95 in hd95_values_per_label.items():
#     print(f'Label {label}: HD95={hd95:.10f}')
# # Print results
# for dice in enumerate(dice_coefficients):
#     print(f"Dice Coefficient: {dice}")
#     print("-" * 30)
import os

def process_data_folder(prediction_folder, ground_truth_folder):
    # Get all file names in the prediction folder
    prediction_files = os.listdir(prediction_folder)

    # Initialize lists to store metrics for each file
    all_dice_coefficients_per_file = []
    all_metrics_per_label = []
    all_hd95_values_per_file = []

    for prediction_file in prediction_files:
        prediction_path = os.path.join(prediction_folder, prediction_file)
        ground_truth_file = prediction_file.replace("data", "data")  # Assuming the naming convention
        ground_truth_path = os.path.join(ground_truth_folder, ground_truth_file)

        # Read images
        prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt16)
        ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt16)

        prediction_array = sitk.GetArrayFromImage(prediction)
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)

        # Calculate metrics for each label
        dice_coefficients = compute_dice_coefficient(ground_truth_array, prediction_array)
        metrics_per_label = compute_metrics_for_labels(ground_truth_array, prediction_array)

        # Get unique labels in the ground truth
        unique_labels = np.unique(sitk.GetArrayFromImage(ground_truth))
        print(unique_labels)
        # Calculate HD95 for each label separately
        hd95_values_per_label = compute_hausdorff_95(prediction, ground_truth, labels=unique_labels)

        # Append metrics for the current file to lists
        all_dice_coefficients_per_file.append(dice_coefficients)
        all_metrics_per_label.extend(metrics_per_label)
        all_hd95_values_per_file.append(hd95_values_per_label)

        # Print results for each label and file
        print(f"Results for {prediction_file}:")
        for label, hd in metrics_per_label:
            print(f'Label {label}: HD={hd:.10f}')

        for label, hd95 in hd95_values_per_label.items():
            print(f'Label {label}: HD95={hd95:.10f}')

        # Print Dice Coefficient for the current file
        print(f"Dice Coefficient for {prediction_file}: {dice_coefficients}")
        print("-" * 30)

    # Calculate average Dice and Hausdorff values over all files
    average_dice_per_label = compute_average_dice(all_dice_coefficients_per_file)
    average_hd95_values = compute_average_hd95(all_hd95_values_per_file)

    # Print overall results
    print("\nOverall Results:")
    for label, average_dice in average_dice_per_label.items():
        print(f'Average Dice Coefficient for Label {label}: {average_dice:.10f}')

    for label, average_hd95 in average_hd95_values.items():
        print(f'Average HD95 for Label {label}: {average_hd95:.10f}')

def compute_average_dice(all_dice_coefficients_per_file):
    all_dice_coefficients = {}

    # Merge Dice coefficients for each label across all files
    for dice_coefficients_per_file in all_dice_coefficients_per_file:
        for label, dice_coefficient in enumerate(dice_coefficients_per_file):
            if label not in all_dice_coefficients:
                all_dice_coefficients[label] = []
            all_dice_coefficients[label].append(dice_coefficient)

    # Calculate average Dice for each label
    average_dice_per_label = {label: np.mean(values) for label, values in all_dice_coefficients.items()}

    return average_dice_per_label

def compute_average_hd95(all_hd95_values_per_file):
    all_hd95_values = {}

    # Merge HD95 values for each label across all files
    for hd95_values_per_file in all_hd95_values_per_file:
        for label, hd95 in hd95_values_per_file.items():
            if label not in all_hd95_values:
                all_hd95_values[label] = []
            all_hd95_values[label].append(hd95)

    # Calculate average HD95 for each label
    average_hd95_values = {label: np.mean(values) for label, values in all_hd95_values.items()}

    return average_hd95_values

# Replace with your folder paths
prediction_folder = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset031_301shuzhong/31-11/testsegresnet'
ground_truth_folder = '/home/ps/wyc/nnUNet/DATA/nnUNet_raw/Dataset031_301shuzhong/31-11/gt'

process_data_folder(prediction_folder, ground_truth_folder)

