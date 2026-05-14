import numpy as np
import nibabel as nib

# 創建一個小型體積 (64x64x32)
small_volume = np.random.rand(32, 64, 64).astype(np.float32)

# 保存為NII.GZ
affine = np.eye(4)
nib_img = nib.Nifti1Image(small_volume, affine)
nib.save(nib_img, "small_test.nii.gz")