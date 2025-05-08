import os
import shutil

def split_dataset(image_dir, train_dir, val_dir, train_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:]
    
    for img in train_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_dir, img))
    for img in val_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(val_dir, img))
    
    print(f"Copied {len(train_images)} images to {train_dir}")
    print(f"Copied {len(val_images)} images to {val_dir}")

split_dataset(
    image_dir= '/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/',
    train_dir='data/train/images',
    val_dir='data/val/images'
)