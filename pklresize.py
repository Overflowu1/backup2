import numpy as np
import pickle as pkl
from batchgenerators.utilities.file_and_folder_operations import *

path = r'/home/wu/wyc/nnUNet/DATA/nnUNet_results/Dataset031_301pelvis/nnUNetTrainer__nnUNetPlans__3d_lowres/nnUNetPlans_plans_3D.pkl'

with (open(path, 'rb')) as f:
    s = pkl.load(f)
    print(s['plans_per_stage'][0]['batch_size'])
    print(s['plans_per_stage'][1]['patch_size'])

    plans = load_pickle(path)
    plans['plans_per_stage'][0]['batch_size'] = 2
    plans['plans_per_stage'][0]['patch_size'] = np.array((32, 160, 128))

    plans['plans_per_stage'][1]['batch_size'] = 2
    plans['plans_per_stage'][1]['patch_size'] = np.array((32, 160, 128))

    save_pickle(plans, join(r'/home/all_data/nnUNet/nnUNet_processed/Task11_CTPelvic1K/nnUNetPlans_plans_3D.pkl'))