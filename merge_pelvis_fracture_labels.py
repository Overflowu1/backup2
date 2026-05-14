import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def get_stem_nii_gz(path: Path):
    """
    正确处理 .nii.gz 文件名
    例如 case001.nii.gz -> case001
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def same_geometry(img1, img2, atol=1e-5):
    """
    判断两个 NIfTI label 是否在同一空间中：
    size, spacing, origin, direction 都要一致
    """
    if img1.GetSize() != img2.GetSize():
        return False

    if not np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=atol):
        return False

    if not np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=atol):
        return False

    if not np.allclose(img1.GetDirection(), img2.GetDirection(), atol=atol):
        return False

    return True


def resample_label_to_reference(label_img, reference_img):
    """
    如果骨折 label 和骨盆 label 的空间不一致，
    将骨折 label 重采样到骨盆 label 空间。

    注意：label 图像必须使用最近邻插值。
    """
    resampled = sitk.Resample(
        label_img,
        reference_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label_img.GetPixelID()
    )
    return resampled


def merge_one_case(
    pelvis_label_path,
    fracture_label_path,
    output_path,
    fracture_new_label=4,
    restrict_fracture_to_pelvis=False
):
    """
    pelvis_label:
        0 background
        1 sacrum
        2 left_hip
        3 right_hip

    fracture_label:
        0 background
        1 fracture

    output:
        0 background
        1 sacrum
        2 left_hip
        3 right_hip
        4 fracture
    """

    pelvis_img = sitk.ReadImage(str(pelvis_label_path))
    fracture_img = sitk.ReadImage(str(fracture_label_path))

    if not same_geometry(pelvis_img, fracture_img):
        print(f"[Warning] Geometry not same, resampling fracture label:")
        print(f"  pelvis:   {pelvis_label_path}")
        print(f"  fracture: {fracture_label_path}")
        fracture_img = resample_label_to_reference(fracture_img, pelvis_img)

    pelvis_arr = sitk.GetArrayFromImage(pelvis_img)
    fracture_arr = sitk.GetArrayFromImage(fracture_img)

    # 检查骨盆标签是否合法
    pelvis_unique = np.unique(pelvis_arr)
    invalid_pelvis_labels = set(pelvis_unique.tolist()) - {0, 1, 2, 3}
    if len(invalid_pelvis_labels) > 0:
        print(f"[Warning] Invalid pelvis labels in {pelvis_label_path}: {invalid_pelvis_labels}")

    # 检查骨折标签
    fracture_unique = np.unique(fracture_arr)
    invalid_fracture_labels = set(fracture_unique.tolist()) - {0, 1}
    if len(invalid_fracture_labels) > 0:
        print(f"[Warning] Unexpected fracture labels in {fracture_label_path}: {invalid_fracture_labels}")
        print("          The script will treat all values > 0 as fracture.")

    # 以三分类骨盆为基础
    merged_arr = pelvis_arr.copy().astype(np.uint8)

    # 骨折区域，通常 fracture label 中 >0 的地方都视为骨折
    fracture_mask = fracture_arr > 0

    # 可选：只保留落在骨盆区域内部的骨折
    # 一般不建议开启，因为有些骨折块可能不完全在原始骨盆mask内
    if restrict_fracture_to_pelvis:
        fracture_mask = np.logical_and(fracture_mask, pelvis_arr > 0)

    # 骨折覆盖原始骨盆标签
    merged_arr[fracture_mask] = fracture_new_label

    merged_img = sitk.GetImageFromArray(merged_arr)
    merged_img.CopyInformation(pelvis_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(merged_img, str(output_path))


def batch_merge(
    pelvis_label_dir,
    fracture_label_dir,
    output_dir,
    restrict_fracture_to_pelvis=False
):
    pelvis_label_dir = Path(pelvis_label_dir)
    fracture_label_dir = Path(fracture_label_dir)
    output_dir = Path(output_dir)

    pelvis_files = {
        get_stem_nii_gz(p): p
        for p in pelvis_label_dir.glob("*.nii.gz")
    }

    fracture_files = {
        get_stem_nii_gz(p): p
        for p in fracture_label_dir.glob("*.nii.gz")
    }

    common_cases = sorted(set(pelvis_files.keys()) & set(fracture_files.keys()))

    if len(common_cases) == 0:
        raise RuntimeError("No matched cases found. Please check file names.")

    print(f"Found {len(common_cases)} matched cases.")

    missing_fracture = sorted(set(pelvis_files.keys()) - set(fracture_files.keys()))
    missing_pelvis = sorted(set(fracture_files.keys()) - set(pelvis_files.keys()))

    if len(missing_fracture) > 0:
        print(f"[Warning] {len(missing_fracture)} pelvis labels have no fracture label.")

    if len(missing_pelvis) > 0:
        print(f"[Warning] {len(missing_pelvis)} fracture labels have no pelvis label.")

    for case_id in tqdm(common_cases):
        pelvis_path = pelvis_files[case_id]
        fracture_path = fracture_files[case_id]
        output_path = output_dir / f"{case_id}.nii.gz"

        merge_one_case(
            pelvis_label_path=pelvis_path,
            fracture_label_path=fracture_path,
            output_path=output_path,
            fracture_new_label=4,
            restrict_fracture_to_pelvis=restrict_fracture_to_pelvis
        )


if __name__ == "__main__":
    pelvis_label_dir = "/mnt/data/DATA/nnUNet_raw/Dataset102_Frc3/labelsTr/1111"
    fracture_label_dir = "/mnt/data/DATA/nnUNet_raw/Dataset102_Frc3/labelsTr"
    output_dir = "/mnt/data/DATA/nnUNet_raw/Dataset102_Frc3/labelsTr/2222"

    batch_merge(
        pelvis_label_dir=pelvis_label_dir,
        fracture_label_dir=fracture_label_dir,
        output_dir=output_dir,
        restrict_fracture_to_pelvis=False
    )