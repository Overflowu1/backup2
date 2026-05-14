import SimpleITK as sitk
from pathlib import Path
import shutil

dataset_root = Path("/mnt/data/DATA/nnUNet_raw/Dataset013_301frac")

images_dir = dataset_root / "imagesTr"
labels_dir = dataset_root / "labelsTr"

backup_dir = dataset_root / "labelsTr_backup_before_fix"
fixed_dir = dataset_root / "labelsTr_fixed_geometry"

# 先备份原始标签
if not backup_dir.exists():
    shutil.copytree(labels_dir, backup_dir)
    print(f"已备份原始 labelsTr 到: {backup_dir}")
else:
    print(f"备份目录已存在: {backup_dir}")

fixed_dir.mkdir(exist_ok=True)

for lab_path in sorted(labels_dir.glob("*.nii.gz")):
    case_name = lab_path.name.replace(".nii.gz", "")
    img_path = images_dir / f"{case_name}_0000.nii.gz"

    if not img_path.exists():
        print(f"[跳过] 找不到对应图像: {img_path}")
        continue

    img = sitk.ReadImage(str(img_path))
    lab = sitk.ReadImage(str(lab_path))

    # 这里非常关键：如果 size 不一致，不要强行 copy header
    if img.GetSize() != lab.GetSize():
        raise RuntimeError(
            f"Size 不一致，不能直接复制空间信息:\n"
            f"Image: {img_path}, size={img.GetSize()}\n"
            f"Label: {lab_path}, size={lab.GetSize()}"
        )

    # 如果你的标签只有 0/1/2/3/4，用 UInt8 就够
    lab_fixed = sitk.Cast(lab, sitk.sitkUInt8)

    # 复制 image 的 spacing / origin / direction 到 label
    lab_fixed.CopyInformation(img)

    out_path = fixed_dir / lab_path.name
    sitk.WriteImage(lab_fixed, str(out_path), useCompression=True)

    print(f"[OK] {lab_path.name}")

print("\n全部修复完成。修复后的标签在:")
print(fixed_dir)