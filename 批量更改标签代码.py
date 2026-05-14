import SimpleITK as sitk
from pathlib import Path


def batch_merge_hip_labels(input_dir, output_dir):
    """
    批量将文件夹中的 CT Label 的右髋(3)合并到左髋(2)

    :param input_dir: 存放原始 Label 文件的文件夹路径
    :param output_dir: 存放处理后 Label 文件的文件夹路径
    """
    # 1. 确保输出文件夹存在，如果不存在则自动创建
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_path = Path(input_dir)

    # 2. 获取所有 .nii 和 .nii.gz 文件
    # 如果你的文件在子文件夹里，可以将 glob 换成 rglob('*.nii*')
    nii_files = list(input_path.glob('*.nii')) + list(input_path.glob('*.nii.gz'))

    if not nii_files:
        print(f"⚠️ 在目录 '{input_dir}' 中没有找到 NIfTI 文件。")
        return

    print(f"🔍 共找到 {len(nii_files)} 个文件，开始批量处理...\n")
    print("-" * 40)

    success_count = 0

    # 3. 遍历并处理每个文件
    for file_path in nii_files:
        file_name = file_path.name
        output_file_path = Path(output_dir) / file_name

        try:
            # 读取图像
            label_image = sitk.ReadImage(str(file_path))

            # 转换为数组并修改标签 (3 -> 2)
            label_array = sitk.GetArrayFromImage(label_image)
            label_array[label_array == 3] = 2

            # 转回图像并复制空间信息
            new_label_image = sitk.GetImageFromArray(label_array)
            new_label_image.CopyInformation(label_image)

            # 保存文件 (推荐启用压缩以节省空间)
            sitk.WriteImage(new_label_image, str(output_file_path), useCompression=True)

            print(f"✅ 成功: {file_name}")
            success_count += 1

        except Exception as e:
            # 如果某个文件读取或处理报错，打印错误并继续处理下一个
            print(f"❌ 失败: {file_name} | 错误信息: {e}")

    print("-" * 40)
    print(f"🎉 批量处理完成！成功处理 {success_count}/{len(nii_files)} 个文件。")
    print(f"📁 结果已保存至: {output_dir}")


# ==========================================
# 使用说明：在此处修改你的输入和输出文件夹路径
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的文件夹路径，例如："D:/data/pelvis_labels_original"
    INPUT_FOLDER = "/mnt/data/DATA/nnUNet_raw/Dataset027_Intro_label2/labelsTr/sa"

    # 替换为你想保存的新文件夹路径，例如："D:/data/pelvis_labels_merged"
    OUTPUT_FOLDER = "/mnt/data/DATA/nnUNet_raw/Dataset027_Intro_label2/labelsTr/sa/tsts2"

    batch_merge_hip_labels(INPUT_FOLDER, OUTPUT_FOLDER)