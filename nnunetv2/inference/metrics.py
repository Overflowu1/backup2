import SimpleITK as sitk
import numpy as np
import os

def compute_dice_coefficient(label_true, label_pred):
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
    dice_coefficients = []
    for label in range(1, np.max(label_true) + 1):
        label_true_binary = (label_true == label).astype(np.uint8)
        label_pred_binary = (label_pred == label).astype(np.uint8)
        label_true_binary = sitk.GetImageFromArray(label_true_binary)
        label_pred_binary = sitk.GetImageFromArray(label_pred_binary)
        label_true_binary = sitk.Cast(label_true_binary, sitk.sitkUInt8)
        label_pred_binary = sitk.Cast(label_pred_binary, sitk.sitkUInt8)
        dice_filter.Execute(label_true_binary, label_pred_binary)
        dice_coefficients.append(dice_filter.GetDiceCoefficient())
    return dice_coefficients

def compute_confusion_matrix_elements(label_true, label_pred):
    TP = np.sum((label_pred == 1) & (label_true == 1))
    TN = np.sum((label_pred == 0) & (label_true == 0))
    FP = np.sum((label_pred == 1) & (label_true == 0))
    FN = np.sum((label_pred == 0) & (label_true == 1))
    return TP, TN, FP, FN

def compute_accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def compute_precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def compute_recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def compute_miou(label_true, label_pred):
    iou_per_label = []
    for label in range(1, np.max(label_true) + 1):
        intersection = np.sum((label_pred == label) & (label_true == label))
        union = np.sum((label_pred == label) | (label_true == label))
        iou = intersection / union if union > 0 else 0
        iou_per_label.append(iou)
    return np.mean(iou_per_label) if iou_per_label else 0

def compute_metrics_for_labels(label_true, label_pred):
    metrics_per_label = []
    for label in range(1, np.max(label_true) + 1):
        label_true_binary = (label_true == label).astype(np.uint8)
        label_pred_binary = (label_pred == label).astype(np.uint8)
        TP, TN, FP, FN = compute_confusion_matrix_elements(label_true_binary, label_pred_binary)
        accuracy = compute_accuracy(TP, TN, FP, FN)
        precision = compute_precision(TP, FP)
        recall = compute_recall(TP, FN)
        f1_score = compute_f1_score(precision, recall)
        miou = compute_miou(label_true_binary, label_pred_binary)
        metrics_per_label.append((label, accuracy, precision, recall, f1_score, miou))
    return metrics_per_label

def process_data_folder(prediction_folder, ground_truth_folder):
    prediction_files = os.listdir(prediction_folder)
    for prediction_file in prediction_files:
        prediction_path = os.path.join(prediction_folder, prediction_file)
        ground_truth_path = os.path.join(ground_truth_folder, prediction_file)
        prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt16)
        ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt16)
        prediction_array = sitk.GetArrayFromImage(prediction)
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        dice_coefficients = compute_dice_coefficient(ground_truth_array, prediction_array)
        metrics_per_label = compute_metrics_for_labels(ground_truth_array, prediction_array)
        print(f"Results for {prediction_file}:")
        for label, accuracy, precision, recall, f1_score, miou in metrics_per_label:
            print(f'Label {label}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1_score:.4f}, MIoU={miou:.4f}')
        print(f"Dice Coefficient for {prediction_file}: {dice_coefficients}")
        print("-" * 30)

prediction_folder = r'/mnt/data/DATA/zjyCopy/imagesTs_predlowres/Mamba/MambaMetric'
ground_truth_folder = r'/mnt/data/DATA/zjyCopy/labelsTs'
process_data_folder(prediction_folder, ground_truth_folder)
