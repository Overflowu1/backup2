import nibabel as nib
import numpy as np
import os
import glob


def swap_labels_2_3(file_path, output_folder):
    """
    将NIfTI文件中的标签2和3互换

    参数:
    file_path: 输入文件路径
    output_folder: 输出文件夹路径
    """
    try:
        # 加载NIfTI文件
        img = nib.load(file_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # 打印修改前的像素值统计
        print(f"处理文件: {os.path.basename(file_path)}")
        print("修改前的像素值统计:")
        unique_values, counts = np.unique(data, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"  像素值 {value}: 出现次数 {count}")

        # 创建数据的副本
        modified_data = data.copy()

        # 找到标签1和2的位置
        mask_indexes_1 = np.where(data == 4)
        # mask_indexes_2 = np.where(data == 2)
        # mask_indexes_3= np.where(data == 3)       # 互换标签1和2
        modified_data[mask_indexes_1] = 0  # 将1改为2
        # modified_data[mask_indexes_2 ] = 3
        # modified_data[mask_indexes_3] = 1  # 将2改为1
        #
        # mask_indexes_11 = np.where(data == 1)
        # mask_indexes_22 = np.where(data == 2)
        # mask_indexes_33 = np.where(data == 3)
        #
        # # 互换标签2和3
        #
        # modified_data[mask_indexes_22] = 3  # 将2改为3
        # modified_data[mask_indexes_33] = 2  # 将3改为2

        # 创建新的NIfTI图像
        new_img = nib.Nifti1Image(modified_data, affine, header)

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 构建输出文件路径
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)

        # 保存修改后的文件
        nib.save(new_img, output_path)

        # 打印修改后的像素值统计
        print("修改后的像素值统计:")
        unique_values, counts = np.unique(modified_data, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"  像素值 {value}: 出现次数 {count}")
        print("-" * 50)

        return True

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False


def process_folder(input_folder, output_folder, file_extension=".nii"):
    """
    处理文件夹中的所有NIfTI文件

    参数:
    input_folder: 输入文件夹路径
    output_folder: 输出文件夹路径
    file_extension: 文件扩展名 (默认为.nii)
    """
    # 查找所有NIfTI文件
    search_pattern = os.path.join(input_folder, f"*{file_extension}")
    nii_files = glob.glob(search_pattern)
    nii_files.extend(glob.glob(os.path.join(input_folder, f"*{file_extension}.gz")))  # 包括.nii.gz文件

    if not nii_files:
        print(f"在文件夹 {input_folder} 中未找到{file_extension}文件")
        return

    print(f"找到 {len(nii_files)} 个文件需要处理")
    print("=" * 80)

    # 处理每个文件
    success_count = 0
    for file_path in nii_files:
        if swap_labels_2_3(file_path, output_folder):
            success_count += 1

    print("=" * 80)
    print(f"处理完成: {success_count}/{len(nii_files)} 个文件成功处理")

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = "/mnt/data/DATA/nnssl_raw/Dataset011_4pelvis/labelsTr"  # 替换为你的输入文件夹路径
    output_folder = "/mnt/data/DATA/nnssl_raw/Dataset011_4pelvis/labelsTr2"  # 替换为你的输出文件夹路径

    # 处理文件夹中的所有文件
    process_folder(input_folder, output_folder)

    # 如果你想处理单个文件，可以使用:
    # swap_labels_2_3(r"your_file_path.nii", output_folder)