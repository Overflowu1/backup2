import os, gc
import numpy as np
import SimpleITK as sitk

# 关键：禁用多线程（经常能直接避免 SIGSEGV）
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetTransform(sitk.Transform())
    return r.Execute(moving)


def dice_for_label(gt_bin_img: sitk.Image, pr_bin_img: sitk.Image) -> float:
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(sitk.Cast(gt_bin_img, sitk.sitkUInt8), sitk.Cast(pr_bin_img, sitk.sitkUInt8))
    return float(f.GetDiceCoefficient())


def hd95_sitk_surface(gt_bin_img: sitk.Image,
                      pr_bin_img: sitk.Image,
                      use_spacing: bool = True,
                      empty_value=np.nan):
    """
    Return (hd95, hd). If one side empty -> empty_value (nan or inf).
    """
    gt_bin_img = sitk.Cast(gt_bin_img, sitk.sitkUInt8)
    pr_bin_img = sitk.Cast(pr_bin_img, sitk.sitkUInt8)

    # 更稳：用拷贝数组求和，避免 view 引发底层引用问题
    gt_sum = int(sitk.GetArrayFromImage(gt_bin_img).sum())
    pr_sum = int(sitk.GetArrayFromImage(pr_bin_img).sum())

    if gt_sum == 0 and pr_sum == 0:
        return 0.0, 0.0
    if gt_sum == 0 or pr_sum == 0:
        return float(empty_value), float(empty_value)

    # 表面
    gt_surf = sitk.LabelContour(gt_bin_img)
    pr_surf = sitk.LabelContour(pr_bin_img)

    # 若表面为空（极少见，但要防）
    if int(sitk.GetArrayFromImage(gt_surf).sum()) == 0 or int(sitk.GetArrayFromImage(pr_surf).sum()) == 0:
        return float(empty_value), float(empty_value)

    # 距离图：显式转 Float32（降低内存占用，减少崩溃概率）
    gt_dt = sitk.Cast(
        sitk.Abs(sitk.SignedMaurerDistanceMap(gt_bin_img, squaredDistance=False, useImageSpacing=use_spacing)),
        sitk.sitkFloat32
    )
    pr_dt = sitk.Cast(
        sitk.Abs(sitk.SignedMaurerDistanceMap(pr_bin_img, squaredDistance=False, useImageSpacing=use_spacing)),
        sitk.sitkFloat32
    )

    d_pr_to_gt = sitk.GetArrayFromImage(sitk.Mask(gt_dt, pr_surf)).ravel()
    d_gt_to_pr = sitk.GetArrayFromImage(sitk.Mask(pr_dt, gt_surf)).ravel()

    d_pr_to_gt = d_pr_to_gt[d_pr_to_gt > 0]
    d_gt_to_pr = d_gt_to_pr[d_gt_to_pr > 0]

    if d_pr_to_gt.size == 0 and d_gt_to_pr.size == 0:
        return 0.0, 0.0
    if d_pr_to_gt.size == 0 or d_gt_to_pr.size == 0:
        return float(empty_value), float(empty_value)

    all_d = np.concatenate([d_pr_to_gt, d_gt_to_pr])

    hd95 = float(np.percentile(all_d, 95))
    hd = float(np.max(all_d))

    # 主动释放大对象，降低峰值内存
    del gt_surf, pr_surf, gt_dt, pr_dt, d_pr_to_gt, d_gt_to_pr, all_d
    gc.collect()

    return hd95, hd


def compute_case_metrics(gt_img: sitk.Image, pr_img: sitk.Image, empty_value=np.nan):
    pr_img = resample_to_reference(pr_img, gt_img)

    labels = np.unique(sitk.GetArrayFromImage(gt_img))
    labels = [int(x) for x in labels if int(x) != 0]

    out = {}
    for lab in labels:
        gt_bin = sitk.Cast(gt_img == lab, sitk.sitkUInt8)
        pr_bin = sitk.Cast(pr_img == lab, sitk.sitkUInt8)

        dsc = dice_for_label(gt_bin, pr_bin)
        hd95, hd = hd95_sitk_surface(gt_bin, pr_bin, empty_value=empty_value)

        out[lab] = {"dice": dsc, "hd95": hd95, "hd": hd}

        # 每个 label 后释放，避免积累
        del gt_bin, pr_bin
        gc.collect()

    return out


def process_folder(pred_folder, gt_folder, empty_value=np.nan):
    files = sorted([f for f in os.listdir(pred_folder) if f.endswith((".nii", ".nii.gz"))])

    all_results = []
    for f in files:
        pred_path = os.path.join(pred_folder, f)
        gt_path = os.path.join(gt_folder, f)  # 如命名不同，请改这里

        if not os.path.exists(gt_path):
            print(f"[Skip] GT not found: {gt_path}")
            continue

        print(f"\n>>> Processing: {f}")
        pr = sitk.ReadImage(pred_path, sitk.sitkUInt16)
        gt = sitk.ReadImage(gt_path, sitk.sitkUInt16)

        # 快速打印一下空间信息，确认对齐逻辑正确
        print("GT size/spacing:", gt.GetSize(), gt.GetSpacing())
        print("PR size/spacing:", pr.GetSize(), pr.GetSpacing())

        case = compute_case_metrics(gt, pr, empty_value=empty_value)
        all_results.append((f, case))

        for lab, m in sorted(case.items()):
            print(f"Label {lab}: Dice={m['dice']:.6f}  HD95={m['hd95']:.6f}  HD={m['hd']:.6f}")

        # 释放整例影像，避免内存累积
        del pr, gt, case
        gc.collect()

    return all_results


if __name__ == "__main__":
    prediction_folder = r"/mnt/data/DATA/zjyData/predict/STUNet/metric"
    ground_truth_folder = r"/mnt/data/DATA/nnUNet_raw/Dataset123_fra/labelsTs"
    process_folder(prediction_folder, ground_truth_folder, empty_value=np.nan)
