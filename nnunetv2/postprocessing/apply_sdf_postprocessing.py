import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
from nnunetv2.postprocessing.sdf_keep_close_components import sdf_keep_close_components

if __name__ == "__main__":
    # 1. 预测输出的文件夹 (nnUNetv2_predict 的输出)
    input_folder = "/mnt/data/DATA/1215/28simseg5"
    # 2. 我们希望把SDF后处理的输出放在哪
    output_folder = os.path.join(input_folder, "postprocessed_SDF")
    maybe_mkdir_p(output_folder)

    # 3. 把我们的 SDF 后处理函数塞进 pp_fns 列表
    pp_fns = [sdf_keep_close_components]
    pp_fn_kwargs = [dict(
        main_region_th=100000,
        sdf_th=35,
        region_th=2000,
        background_label=0,
        verbose=False,
    )]

    # 4. 跑批处理
    # plans_file_or_dict / dataset_json_file_or_dict 可以不传，
    # apply_postprocessing_to_folder 会自动在 input_folder 下找 plans.json 和 dataset.json
    apply_postprocessing_to_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        pp_fns=pp_fns,
        pp_fn_kwargs=pp_fn_kwargs,
        plans_file_or_dict=None,
        dataset_json_file_or_dict=None,
        num_processes=8,
    )

    print("Done. SDF postprocessed segmentations are in:", output_folder)
