import os
import SimpleITK as sitk
import shutil

# 输入的固定 NIfTI 文件
img_name = r'/mnt/data/DATA/nnUNet_raw/Dataset123_fra/labelsTs/dataset6_CLINIC_0082.nii'

# 需要处理的文件夹路径
input_folder = r'/mnt/data/DATA/zjyData/predict/ArconvLstmASFEB/metric'

# 输出文件夹
output_folder = r'/mnt/data/DATA/zjyData/predict/ArconvLstmASFEB/afterSpacing'

# 读取基准图像
ct = sitk.ReadImage(img_name)
o = ct.GetOrigin()
d = ct.GetDirection()
s = ct.GetSpacing()
key2 = ct.GetMetaDataKeys()


def process_nii_file(file_path, output_path):
    """处理单个 .nii 文件"""
    modified_data = sitk.ReadImage(file_path)

    # 更新空间信息
    modified_data.SetOrigin(o)
    modified_data.SetDirection(d)
    modified_data.SetSpacing(s)

    # 复制元数据
    key1 = modified_data.GetMetaDataKeys()
    for current_key1, current_key2 in zip(key1, key2):
        if ct.HasMetaDataKey(current_key2):
            modified_data.SetMetaData(current_key1, ct.GetMetaData(current_key2))

    # 保存到输出文件夹
    sitk.WriteImage(modified_data, output_path)
    print(f"处理完成: {file_path} -> {output_path}")


# 遍历 input_folder 中的所有子文件夹
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".nii"):  # 只处理 .nii 文件
            input_file_path = os.path.join(root, file)

            # 计算输出路径，保持子文件夹结构
            relative_path = os.path.relpath(input_file_path, input_folder)
            output_file_path = os.path.join(output_folder, relative_path)

            # 确保输出文件夹存在
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # 处理 NIfTI 文件
            process_nii_file(input_file_path, output_file_path)

print("所有文件处理完成！")