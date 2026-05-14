import ants
import torch
import numpy as np

def read_images():
    image_paths = [
        "/home/ps/wyc/muban/dataset6_CLINIC_0096_data.nii.gz",
        "/home/ps/wyc/muban/dataset6_CLINIC_0005_data.nii.gz",
        "/home/ps/wyc/muban/dataset6_CLINIC_0019_data.nii.gz",
        "/home/ps/wyc/muban/dataset6_CLINIC_0022_data.nii.gz",
        "/home/ps/wyc/muban/dataset6_CLINIC_0046_data.nii.gz",
        "/home/ps/wyc/muban/dataset6_CLINIC_0052_data.nii.gz"
    ]
    moving_images = [ants.image_read(path) for path in image_paths]
    return moving_images

def get_multiscale_resolutions(original_shape, min_size=32):
    scales = []
    current_shape = original_shape

    while all(dim >= min_size for dim in current_shape):
        scales.append(current_shape)
        current_shape = tuple(max(min_size, dim // 2) for dim in current_shape)

    return scales

def register_images_multiscale(fixed_img, moving_img):
    original_shape = fixed_img.shape
    scales = get_multiscale_resolutions(original_shape, min_size=64)  # Adjusted min_size to reduce levels

    for scale in reversed(scales):  # Start from the coarsest scale to the finest
        fixed_img_resampled = ants.resample_image(fixed_img, scale, use_voxels=True)
        moving_img_resampled = ants.resample_image(moving_img, scale, use_voxels=True)

        registration = ants.registration(
            fixed=fixed_img_resampled,
            moving=moving_img_resampled,
            type_of_transform='SyNCC',
            reg_iterations=(100, 50, 25),  # Adjusted reg_iterations to reduce memory usage
            verbose=True
        )
        moving_img = ants.apply_transforms(
            fixed=fixed_img,
            moving=moving_img,
            transformlist=registration['fwdtransforms'],
            interpolator='linear'
        )

        # Explicitly delete variables to free memory
        del fixed_img_resampled, moving_img_resampled, registration

    return moving_img

def accumulate_images(images):
    accumulated_image = np.zeros(images[0].shape)
    for img in images:
        accumulated_image += img.numpy()
    average_image = accumulated_image / len(images)
    return ants.from_numpy(average_image, origin=images[0].origin, spacing=images[0].spacing,
                           direction=images[0].direction)

def smooth_image(image, sigma=1.0):
    smoothed_image = ants.smooth_image(image, sigma)
    return smoothed_image

def save_image(image, path):
    ants.image_write(image, path)

def to_torch(image):
    np_image = image.numpy()
    return torch.tensor(np_image, dtype=torch.float32).unsqueeze(0).cuda()

def from_torch(tensor, template):
    np_image = tensor.squeeze(0).cpu().numpy()
    return ants.from_numpy(np_image, origin=template.origin, spacing=template.spacing, direction=template.direction)

def main():
    moving_images = read_images()
    template = moving_images[0]

    for iteration in range(5):  # Reduced the number of iterations to reduce memory usage
        print(f"Iteration {iteration + 1}")

        registered_images = []
        for idx, m_img in enumerate(moving_images):
            if idx == 0 and iteration == 0:
                registered_images.append(m_img)
                continue
            registered_img = register_images_multiscale(template, m_img)
            registered_img.set_direction(template.direction)
            registered_img.set_origin(template.origin)
            registered_img.set_spacing(template.spacing)
            registered_images.append(registered_img)

            # Explicitly delete variables to free memory
            del registered_img

        # Step 3: Compute the average template
        torch_images = [to_torch(img) for img in registered_images]
        accumulated_image = torch.sum(torch.stack(torch_images), dim=0) / len(torch_images)
        new_template = from_torch(accumulated_image, template)
        template = new_template

        # Explicitly delete variables to free memory
        del registered_images, new_template

    # Apply smoothing to the final template
    smoothed_template = smooth_image(template, sigma=1.0)

    # Verify the resolution of the final template
    print(f"Final template resolution: {smoothed_template.shape}")

    # Save the final template
    save_image(smoothed_template, "/home/ps/wyc/muban/result/average_template3.nii.gz")

if __name__ == "__main__":
    main()