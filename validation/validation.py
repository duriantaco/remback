import os
import cv2
import torch
import numpy as np
from torchmetrics import JaccardIndex, Accuracy
import sys
import torch.nn.functional as F
from tqdm import tqdm
from mtcnn import MTCNN
from rembg import remove

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from remback.remover import BackgroundRemover

VAL_IMAGES_DIR = "/remback/data/val/images"
VAL_MASKS_DIR = "/remback/data/val/masks"
ORIGINAL_CHECKPOINT = "/remback/checkpoints/sam_vit_b_01ec64.pth"
FINE_TUNED_CHECKPOINT = "/remback/train/checkpoints/remback_v4.pth"

remover_fine_tuned = BackgroundRemover(FINE_TUNED_CHECKPOINT)
remover_original = BackgroundRemover(ORIGINAL_CHECKPOINT)
mtcnn_detector = MTCNN()

metrics = {
    "fine_tuned_sam": {
        "iou": JaccardIndex(num_classes=2, task="binary").to(remover_fine_tuned.dev),
        "acc": Accuracy(num_classes=2, task="binary").to(remover_fine_tuned.dev)
    },
    "original_sam": {
        "iou": JaccardIndex(num_classes=2, task="binary").to(remover_original.dev),
        "acc": Accuracy(num_classes=2, task="binary").to(remover_original.dev)
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

val_images = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith('.jpg')])[:100]

skipped_images = []

for img_name in tqdm(val_images, desc="Evaluating images"):
    img_path = os.path.join(VAL_IMAGES_DIR, img_name)
    mask_path = os.path.join(VAL_MASKS_DIR, img_name.replace('.jpg', '.png'))

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"Skipping {img_name}: Ground truth mask not found")
        continue
    gt_mask = (gt_mask > 0).astype(np.uint8) 
    gt_mask_tensor_gpu = torch.tensor(gt_mask, dtype=torch.long, device=remover_fine_tuned.dev)
    gt_mask_tensor_cpu = torch.tensor(gt_mask, dtype=torch.long, device='cpu')

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        mask_fine_tuned = remover_fine_tuned.get_mask(img_path)
        mask_fine_tuned = torch.tensor(mask_fine_tuned, dtype=torch.float32, device=remover_fine_tuned.dev).unsqueeze(0).unsqueeze(0)
        mask_fine_tuned = F.interpolate(mask_fine_tuned, size=(512, 512), mode='nearest').squeeze()
        mask_fine_tuned = (mask_fine_tuned > 0.5).long()

        mask_original = remover_original.get_mask(img_path)
        mask_original = torch.tensor(mask_original, dtype=torch.float32, device=remover_original.dev).unsqueeze(0).unsqueeze(0)
        mask_original = F.interpolate(mask_original, size=(512, 512), mode='nearest').squeeze()
        mask_original = (mask_original > 0.5).long()

        metrics["fine_tuned_sam"]["iou"].update(mask_fine_tuned, gt_mask_tensor_gpu)
        metrics["fine_tuned_sam"]["acc"].update(mask_fine_tuned, gt_mask_tensor_gpu)
        metrics["original_sam"]["iou"].update(mask_original, gt_mask_tensor_gpu)
        metrics["original_sam"]["acc"].update(mask_original, gt_mask_tensor_gpu)

        faces = mtcnn_detector.detect_faces(img_rgb)
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            mask_mtcnn = np.zeros_like(gt_mask)
            mask_mtcnn[y:y+h, x:x+w] = 1
            mask_mtcnn = cv2.resize(mask_mtcnn, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_mtcnn_tensor = torch.tensor(mask_mtcnn, dtype=torch.long, device='cpu')
            metrics["mtcnn"]["iou"].update(mask_mtcnn_tensor, gt_mask_tensor_cpu)
            metrics["mtcnn"]["acc"].update(mask_mtcnn_tensor, gt_mask_tensor_cpu)
        else:
            print(f"No face detected by MTCNN in {img_name}")
            skipped_images.append(img_name)

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

print("\nBenchmark Results:")
for model_name in metrics:
    miou = metrics[model_name]["iou"].compute()
    accuracy = metrics[model_name]["acc"].compute()
    print(f"{model_name.replace('_', ' ').capitalize()} - mIoU: {miou:.4f}, Accuracy: {accuracy:.4f}")

if skipped_images:
    print(f"\nNumber of images skipped due to errors or no face detection: {len(skipped_images)}")
    print("Images skipped:", skipped_images)
else:
    print("\nAll images processed successfully.")