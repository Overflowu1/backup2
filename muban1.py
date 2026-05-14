import ants
import numpy as np
import torch

def read_images():
    moving_images = [
        ants.image_read("/home/ps/wyc/muban/dataset6_CLINIC_0096_data.nii.gz"),
        ants.image_read("/home/ps/wyc/muban/dataset6_CLINIC_0005_data.nii.gz"),
        ants.image_read("/home/ps/wyc/muban/dataset6_CLINIC_0019_data.nii.gz"),
        ants.image_read("/home/ps/wyc/muban/dataset6_CLINIC_0022_data.nii.gz"),
        ants.image_read("/home/ps/wyc/muban/dataset6_CLINIC_0046_data.nii.gz"),
        ants.image_read("/home/ps/wyc/muban/dataset6_CLINIC_0052_data.nii.gz")
    ]
    return moving_images

def register_images(fixed_img, moving_img, type_of_transform='SyNCC'):
    registration = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=type_of_transform)
    warped_img = ants.apply_transforms(fixed=fixed_img, moving=moving_img, transformlist=registration['fwdtransforms'], interpolator="linear")
    return warped_img

def accumulate_images(images):
    # 将图像转为torch张量并在GPU上进行累积
    accumulated_image = torch.zeros(images[0].shape, device='cuda')
    for img in images:
        accumulated_image += torch.tensor(img.numpy(), device='cuda')
    average_image = accumulated_image / len(images)
    average_image = average_image.cpu().numpy()  # 将张量移回CPU以便使用ANTs库
    return ants.from_numpy(average_image, origin=images[0].origin, spacing=images[0].spacing, direction=images[0].direction)

def smooth_image(image, sigma=1.0):
    smoothed_image = ants.smooth_image(image, sigma)
    return smoothed_image

def save_image(image, path):
    ants.image_write(image, path)

def main():
    moving_images = read_images()

    # Step 1: Select the first image as the initial template
    template = moving_images[0]

    for iteration in range(10):  # Increase the number of iterations
        print(f"Iteration {iteration + 1}")

        registered_images = []
        for idx, m_img in enumerate(moving_images):
            if idx == 0 and iteration == 0:
                registered_images.append(m_img)
                continue
            registered_img = register_images(template, m_img)
            registered_img.set_direction(template.direction)
            registered_img.set_origin(template.origin)
            registered_img.set_spacing(template.spacing)
            registered_images.append(registered_img)

        # Step 3: Compute the average template
        new_template = accumulate_images(registered_images)
        template = new_template

    # Apply smoothing to the final template
    smoothed_template = smooth_image(template, sigma=1.0)

    # Save the final template
    save_image(smoothed_template, "/home/ps/wyc/muban/result/average_template3.nii.gz")

if __name__ == "__main__":
    main()