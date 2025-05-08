import os
import cv2
import numpy as np

def merge_masks_for_images(image_dir, mask_base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    for img_file in image_files:
        img_id = os.path.splitext(img_file)[0]
        subfolder = str(int(img_id) // 2000)
        mask_dir = os.path.join(mask_base_dir, subfolder)
        if not os.path.exists(mask_dir):
            print(f"Error: Mask directory {mask_dir} not found for image {img_id}")
            continue
        combined_mask = np.zeros((512, 512), dtype=np.uint8)
        for mask_file in os.listdir(mask_dir):
            if mask_file.startswith(str(int(img_id)).zfill(5)):
                mask_path = os.path.join(mask_dir, mask_file)
                part_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if part_mask is not None:
                    combined_mask = np.maximum(combined_mask, part_mask)
        output_path = os.path.join(output_dir, f"{img_id}.png")
        cv2.imwrite(output_path, combined_mask)
        print(f"Saved mask for {img_id} to {output_path}")

train_image_dir = 'data/train/images'
val_image_dir = 'data/val/images'
mask_base_dir = '/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
train_output_dir = 'data/train/masks'
val_output_dir = 'data/val/masks'

merge_masks_for_images(train_image_dir, mask_base_dir, train_output_dir)
merge_masks_for_images(val_image_dir, mask_base_dir, val_output_dir)