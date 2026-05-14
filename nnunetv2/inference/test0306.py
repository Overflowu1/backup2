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


def compute_hd95(label_true, label_pred):
    label_true_binary = sitk.GetImageFromArray((label_true > 0.5).astype(np.uint8))
    label_pred_binary = sitk.GetImageFromArray((label_pred > 0.5).astype(np.uint8))
    hausdorff_computer = sitk.HausdorffDistanceImageFilter()
    hausdorff_computer.Execute(label_true_binary, label_pred_binary)
    return hausdorff_computer.GetHausdorffDistance()


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
        hd95 = compute_hd95(label_true_binary, label_pred_binary)
        metrics_per_label.append((label, accuracy, precision, recall, f1_score, miou, hd95))
    return metrics_per_label


def process_data_folder(prediction_folder, ground_truth_folder):
    prediction_files = os.listdir(prediction_folder)
    all_metrics = []
    for prediction_file in prediction_files:
        prediction_path = os.path.join(prediction_folder, prediction_file)
        ground_truth_path = os.path.join(ground_truth_folder, prediction_file)
        prediction = sitk.ReadImage(prediction_path, sitk.sitkUInt16)
        ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt16)
        prediction_array = sitk.GetArrayFromImage(prediction)
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        metrics_per_label = compute_metrics_for_labels(ground_truth_array, prediction_array)
        all_metrics.append(metrics_per_label)
        print(f"Results for {prediction_file}:")
        for label, accuracy, precision, recall, f1_score, miou, hd95 in metrics_per_label:
            print(
                f'Label {label}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1_score:.4f}, MIoU={miou:.4f}')
        print("-" * 30)

    avg_metrics = np.mean([np.array(m)[:, 1:] for m in all_metrics], axis=0)
    print("Overall Average Metrics:")
    for i, (accuracy, precision, recall, f1_score, miou, hd95) in enumerate(avg_metrics):
        print(
            f'Label {i + 1}: Avg Accuracy={accuracy:.4f}, Avg Precision={precision:.4f}, Avg Recall={recall:.4f}, Avg F1-score={f1_score:.4f}, Avg MIoU={miou:.4f}')

prediction_folder =  r'/mnt/data/DATA/zjyData/metric/kua/predict/MSAC'
ground_truth_folder = r'/mnt/data/DATA/zjyData/metric/kua/label'
process_data_folder(prediction_folder, ground_truth_folder)

#
# /home/yu/anaconda3/envs/nnunet/bin/python /home/yu/nnUNet/nnunetv2/temp/test0306.py
# Results for CLINIC_0011_data.nii.gz:
# Label 1: Accuracy=0.9995, Precision=0.7567, Recall=0.9058, F1-score=0.8245, MIoU=0.7015, HD95=186.3357
# ------------------------------
# Results for CLINIC_0082_data.nii.gz:
# Label 1: Accuracy=0.9992, Precision=0.9973, Recall=0.5187, F1-score=0.6824, MIoU=0.5180, HD95=34.4964
# ------------------------------
# Results for CLINIC_0102_data.nii.gz:
# Label 1: Accuracy=0.9999, Precision=0.9301, Recall=0.8681, F1-score=0.8980, MIoU=0.8149, HD95=342.6208
# ------------------------------
# Results for CLINIC_0103_data.nii.gz:
# Label 1: Accuracy=0.9999, Precision=0.9938, Recall=0.8414, F1-score=0.9113, MIoU=0.8370, HD95=20.1494
# ------------------------------
# Overall Average Metrics:
# Label 1: Avg Accuracy=0.9996, Avg Precision=0.9195, Avg Recall=0.7835, Avg F1-score=0.8291, Avg MIoU=0.7178, Avg HD95=145.9006