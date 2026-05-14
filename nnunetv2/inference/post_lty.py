import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def make_connectivity(structure: int = 26):
    """
    返回 3D 连通域结构元素
    structure: 6 / 18 / 26
    """
    if structure == 6:
        # 仅面相邻
        return ndi.generate_binary_structure(3, 1)
    elif structure == 18:
        # 面+边相邻（近似）
        st = ndi.generate_binary_structure(3, 2)
        # generate_binary_structure(3,2) 本身就是更强邻接（包含对角），严格18邻接需手工构造；
        # 实务中一般用 6 或 26；这里保留 18 作为近似选项。
        return st
    elif structure == 26:
        # 面+边+角相邻
        return np.ones((3, 3, 3), dtype=bool)
    else:
        raise ValueError("structure must be one of {6, 18, 26}")


def split_connected_components_for_labels(
    mask: np.ndarray,
    target_labels,
    start_label: int = 7,
    min_voxels: int = 30,
    connectivity: int = 26,
    background_value: int = 0,
):
    """
    对 mask 中指定的 target_labels 做连通域拆分，并按体素数过滤小连通域。

    规则：
    - 对每个 target_label：
      - 找出该 label 的二值区域
      - 连通域标记 -> 得到多个组件
      - 小于 min_voxels 的组件置为 background_value（去噪）
      - 其余组件依次赋新标签（从 start_label 开始递增）
    - 返回：new_mask, next_label（下一可用标签）
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape={mask.shape}")

    new_mask = mask.copy()
    next_label = start_label
    st = make_connectivity(connectivity)

    target_labels = list(target_labels)

    for lb in target_labels:
        region = (new_mask == lb)
        if not np.any(region):
            continue

        cc, num = ndi.label(region, structure=st)
        if num <= 1:
            # 只有一个连通块：可选择保留原 label 不动，也可以强制从 start_label 开始重标
            # 这里默认“保留原 label 不动”
            continue

        # 统计每个组件体素数（cc 的编号从 1..num）
        counts = np.bincount(cc.ravel())
        # counts[0] 是背景
        for comp_id in range(1, num + 1):
            vox = int(counts[comp_id])
            comp_mask = (cc == comp_id)
            if vox < min_voxels:
                # 去噪：小连通域清零
                new_mask[comp_mask] = background_value
            else:
                # 大连通域：赋予新标签
                new_mask[comp_mask] = next_label
                next_label += 1

    return new_mask, next_label


def main():
    # ====== 你需要改的路径 ======
    input_nii_path = "/mnt/data/DATA/zjyData/VISUAL/18and2246/fuse/2246/swinunetr/LT_data.nii.gz"
    output_nii_path = "/mnt/data/DATA/zjyData/VISUAL/18and2246/fuse/2246/swinunetr/LT_data.nii.gz"

    # ====== 你需要改的参数 ======
    # 举例：你说“左髋上骨折块原本 label 都是 3”
    target_labels = [6]

    # 新标签从 7 开始计数
    start_label = 10

    # 小连通域阈值：例如 < 50 vox 就当噪声去掉
    min_voxels = 50

    # 连通性：26 更容易把斜对角粘连在一起；6 更严格
    connectivity = 26

    # ==========================
    nii = nib.load(input_nii_path)
    mask = nii.get_fdata().astype(np.int32)

    new_mask, next_label = split_connected_components_for_labels(
        mask=mask,
        target_labels=target_labels,
        start_label=start_label,
        min_voxels=min_voxels,
        connectivity=connectivity,
        background_value=0,
    )

    out = nib.Nifti1Image(new_mask.astype(np.uint8), affine=nii.affine, header=nii.header)
    nib.save(out, output_nii_path)
    print(f"[OK] saved: {output_nii_path}")
    print(f"[INFO] next_label would be: {next_label}")


if __name__ == "__main__":
    main()