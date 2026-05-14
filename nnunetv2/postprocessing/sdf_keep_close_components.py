import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from typing import Optional


def sdf_keep_close_components(
    segmentation: np.ndarray,
    main_region_th: int = 100000,
    sdf_th: int = 35,
    region_th: int = 2000,
    background_label: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """
    nnUNetv2-compatible SDF postprocessing for multi-class bone segmentation.

    思路（对应 CTPelvic1K 的 newsdf_post_processor）:
    1. 对 segmentation 中每个前景类别 (≠ background_label)：
       - 找出该类别所有连通域并按体素数降序排序。
       - 阶段一：保留
           a) 最大的连通域（idx 0）；
           b) 任何 area > main_region_th 的连通域。
         记录从哪个 index 开始，其余连通域就是"剩余碎片候选"。
       - 如果除了这些大块，没有剩余碎片候选 -> 直接把大块加入结果。
       - 否则进入阶段二 (SDF 阶段)：
         * 对大块mask做欧式距离变换膨胀（半径≈sdf_th体素）；
         * 对每个候选碎片（按面积降序），
           如果面积 < region_th，后面都会更小，就可以直接 break。
           如果该碎片与膨胀区域相交，则也保留。
    2. 把当前类别保留下来的连通域写入一个全局 mask。
    3. 最终结果 = 全局mask * segmentation。

    参数:
    - segmentation: nnUNetv2 的整数标签体素图 (Z,Y,X)，0=背景。
    - main_region_th: 认为是“主要骨块”的最小体素数阈值 (默认=100000)。
    - sdf_th: 距离阈值(体素)，控制主骨块向外膨胀的半径，默认=35。
    - region_th: 只考虑 >= region_th 体素数的连通域作为“中等碎片”。
                 再小就直接丢弃，默认=2000。
    - background_label: 背景标签，通常是0。
    - verbose: 打印调试信息（可选）。

    返回:
    - post_seg: 经过SDF策略过滤后的 segmentation，类型同输入。
    """

    pred = segmentation
    pred_dtype = pred.dtype
    # 这个 mask_whole 最后会告诉我们哪些位置要保留原始 pred 的标签
    mask_whole = np.zeros_like(pred, dtype=np.uint8)

    # 遍历每个类（跳过背景）
    unique_labels = np.unique(pred)
    for cls_label in unique_labels:
        if cls_label == background_label:
            continue

        # 二值化该类
        pred_single = (pred == cls_label).astype(np.uint8)
        if pred_single.sum() == 0:
            continue

        # 3D 连通域分割
        connected_label = label(pred_single, connectivity=pred_single.ndim)
        props = regionprops(connected_label)

        # 按面积降序
        props_sorted = sorted(props, key=lambda p: p.area, reverse=True)

        # 阶段一：收集“大块”
        mask_single = np.zeros_like(pred_single, dtype=np.uint8)
        split_index: Optional[int] = None

        for idx_r, region_info in enumerate(props_sorted):
            area = region_info.area
            if verbose:
                print(f"[Class {cls_label}] CC#{idx_r} area={area}")

            if (area > main_region_th) or (idx_r == 0):
                # 保留足够大的连通域 & 最大的那个
                mask_single[connected_label == region_info.label] = 1
            else:
                # 记录从这里开始，后面的都是候选碎片
                split_index = idx_r
                break

        # 如果没有候选碎片，就直接把这些大块并到全局 mask
        if split_index is None:
            mask_whole[mask_single > 0] = 1
            continue

        # 阶段二：SDF 邻近碎片回填
        #
        # 原始实现用 SimpleITK.SignedMaurerDistanceMap，然后阈值 sdf_th。
        # 这里我们用 scipy.ndimage.distance_transform_edt：
        # distance_transform_edt(mask_single == 0) 给出“每个点到最近已保留大块的距离(体素)”
        # 把 < sdf_th 的位置当成“在主骨附近”的区域。
        dist_outside = distance_transform_edt(mask_single == 0)
        sdf_mask_single = (dist_outside < sdf_th).astype(np.uint8)

        # 遍历剩余的连通域（降序排列后从 split_index 开始）
        for region_info in props_sorted[split_index:]:
            area = region_info.area
            if area < region_th:
                # 剩下的都会更小，可以直接停
                break

            part = (connected_label == region_info.label).astype(np.uint8)

            # 如果这个小碎片和 sdf_mask_single 有交集 -> 认为它紧贴主骨, 保留
            if np.any(part & sdf_mask_single):
                mask_single[connected_label == region_info.label] = 1
                if verbose:
                    print(
                        f"[Class {cls_label}] keeping fragment "
                        f"(area={area}) due to SDF proximity"
                    )
            else:
                if verbose:
                    print(
                        f"[Class {cls_label}] discarding fragment "
                        f"(area={area}) far from main bone"
                    )

        # 把该类别保留下来的体素标到全局 mask
        mask_whole[mask_single > 0] = 1

    # 最终输出：对没保留的地方清零（背景）
    post_seg = (mask_whole.astype(pred_dtype)) * pred
    # 理论上乘法已经把未保留区域变成0；如果背景不是0，可额外强制赋值
    if background_label != 0:
        post_seg[(mask_whole == 0)] = background_label

    return post_seg
