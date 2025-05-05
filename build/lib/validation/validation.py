import os
import cv2
import torch
import numpy as np
from torchmetrics import JaccardIndex, Accuracy
import sys
import torch.nn.functional as F
from tqdm import tqdm
from mtcnn import MTCNN
from rembg import remove  # Import Rembg for background removal

# Add parent directory to path (adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segfacebg.api.api import BackgroundRemover  # Assuming this is your SAM implementation

# Define paths (update these to match your setup)
VAL_IMAGES_DIR = "data/val/images"
VAL_MASKS_DIR = "data/val/masks"
ORIGINAL_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
FINE_TUNED_CHECKPOINT = "train/checkpoints/best_fine_tuned_sam.pth"

# Initialize SAM models and MTCNN
remover_fine_tuned = BackgroundRemover(ORIGINAL_CHECKPOINT, FINE_TUNED_CHECKPOINT)
remover_original = BackgroundRemover(ORIGINAL_CHECKPOINT)
mtcnn_detector = MTCNN()

# Define metrics for all models
# SAM metrics on GPU (if available), MTCNN and Rembg on CPU
metrics = {
    "fine_tuned_sam": {
        "iou": JaccardIndex(num_classes=2, task="binary").to(remover_fine_tuned.device),
        "acc": Accuracy(num_classes=2, task="binary").to(remover_fine_tuned.device)
    },
    "original_sam": {
        "iou": JaccardIndex(num_classes=2, task="binary").to(remover_original.device),
        "acc": Accuracy(num_classes=2, task="binary").to(remover_original.device)
    },
    "mtcnn": {
        "iou": JaccardIndex(num_classes=2, task="binary").to('cpu'),
        "acc": Accuracy(num_classes=2, task="binary").to('cpu')
    },
    "rembg": {
        "iou": JaccardIndex(num_classes=2, task="binary").to('cpu'),
        "acc": Accuracy(num_classes=2, task="binary").to('cpu')
    }
}

# Load validation images (limited to 100 for efficiency)
val_images = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith('.jpg')])[:100]

# Track skipped images due to errors or no face detection
skipped_images = []

# Evaluation loop with progress bar
for img_name in tqdm(val_images, desc="Evaluating images"):
    img_path = os.path.join(VAL_IMAGES_DIR, img_name)
    mask_path = os.path.join(VAL_MASKS_DIR, img_name.replace('.jpg', '.png'))

    # Load and preprocess ground truth mask
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"Skipping {img_name}: Ground truth mask not found")
        continue
    gt_mask = (gt_mask > 0).astype(np.uint8)  # Ensure binary (0 or 1)
    gt_mask_tensor_gpu = torch.tensor(gt_mask, dtype=torch.long, device=remover_fine_tuned.device)
    gt_mask_tensor_cpu = torch.tensor(gt_mask, dtype=torch.long, device='cpu')

    # Load image for MTCNN and Rembg
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        # Fine-tuned SAM mask
        mask_fine_tuned = remover_fine_tuned.get_mask(img_path)  # Expected shape: [1024, 1024]
        mask_fine_tuned = torch.tensor(mask_fine_tuned, dtype=torch.float32, device=remover_fine_tuned.device).unsqueeze(0).unsqueeze(0)
        mask_fine_tuned = F.interpolate(mask_fine_tuned, size=(512, 512), mode='nearest').squeeze()
        mask_fine_tuned = (mask_fine_tuned > 0.5).long()

        # Original SAM mask
        mask_original = remover_original.get_mask(img_path)  # Expected shape: [1024, 1024]
        mask_original = torch.tensor(mask_original, dtype=torch.float32, device=remover_original.device).unsqueeze(0).unsqueeze(0)
        mask_original = F.interpolate(mask_original, size=(512, 512), mode='nearest').squeeze()
        mask_original = (mask_original > 0.5).long()

        # Update SAM metrics
        metrics["fine_tuned_sam"]["iou"].update(mask_fine_tuned, gt_mask_tensor_gpu)
        metrics["fine_tuned_sam"]["acc"].update(mask_fine_tuned, gt_mask_tensor_gpu)
        metrics["original_sam"]["iou"].update(mask_original, gt_mask_tensor_gpu)
        metrics["original_sam"]["acc"].update(mask_original, gt_mask_tensor_gpu)

        # MTCNN: Create mask from bounding box
        faces = mtcnn_detector.detect_faces(img_rgb)
        if len(faces) > 0:
            # Use the first detected face's bounding box
            x, y, w, h = faces[0]['box']
            mask_mtcnn = np.zeros_like(gt_mask)
            mask_mtcnn[y:y+h, x:x+w] = 1  # Set bounding box area to 1
            mask_mtcnn = cv2.resize(mask_mtcnn, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_mtcnn_tensor = torch.tensor(mask_mtcnn, dtype=torch.long, device='cpu')
            metrics["mtcnn"]["iou"].update(mask_mtcnn_tensor, gt_mask_tensor_cpu)
            metrics["mtcnn"]["acc"].update(mask_mtcnn_tensor, gt_mask_tensor_cpu)
        else:
            print(f"No face detected by MTCNN in {img_name}")
            skipped_images.append(img_name)

        # Rembg: Get segmentation mask
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        output_bytes = remove(img_bytes)
        output_img = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask_rembg = (output_img > 0).astype(np.uint8)
        mask_rembg = cv2.resize(mask_rembg, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_rembg_tensor = torch.tensor(mask_rembg, dtype=torch.long, device='cpu')
        metrics["rembg"]["iou"].update(mask_rembg_tensor, gt_mask_tensor_cpu)
        metrics["rembg"]["acc"].update(mask_rembg_tensor, gt_mask_tensor_cpu)

    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        skipped_images.append(img_name)
        continue

# Compute and print final metrics
print("\nBenchmark Results:")
for model_name in metrics:
    miou = metrics[model_name]["iou"].compute()
    accuracy = metrics[model_name]["acc"].compute()
    print(f"{model_name.replace('_', ' ').capitalize()} - mIoU: {miou:.4f}, Accuracy: {accuracy:.4f}")

# Report skipped images
if skipped_images:
    print(f"\nNumber of images skipped due to errors or no face detection: {len(skipped_images)}")
    print("Images skipped:", skipped_images)
else:
    print("\nAll images processed successfully.")