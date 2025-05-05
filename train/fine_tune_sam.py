import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from remback.models import SamPredictor, sam_model_registry
from mtcnn import MTCNN
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CelebAMaskHQDataset(Dataset):
    def __init__(self, image_dir, mask_dir, prompt_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.prompt_dir = prompt_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        logging.info(f"Dataset initialized with {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        prompt_path = os.path.join(self.prompt_dir, img_name.replace('.jpg', '.json'))

        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Image not found: {img_path}, skipping")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"Mask not found for {img_name}, using empty mask")
            mask = np.zeros((1024, 1024), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask = mask / 255.0

        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        if not prompts:
            logging.warning(f"Empty prompt for {img_name}, skipping")
            return None
        prompt = prompts[0] if isinstance(prompts[0], list) else prompts

        logging.debug(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Prompt: {prompt}")
        
        return (
            torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0,
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(prompt, dtype=torch.float32)
        )

def fine_tune_sam(model_checkpoint, image_dir, mask_dir, prompt_dir, epochs=5, batch_size=4, patience=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU Memory Available: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    sam = sam_model_registry["vit_b"](checkpoint=model_checkpoint)
    sam.train()

    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    logging.info("Image encoder parameters frozen")

    sam.to(device)
    logging.info(f"Model moved to {device}")

    dataset = CelebAMaskHQDataset(image_dir, mask_dir, prompt_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: [item for item in x if item is not None]
    )
    logging.info(f"Dataloader created with batch_size={batch_size}, num_workers=4")

    optimizer = torch.optim.Adam(
        list(sam.prompt_encoder.parameters()) + list(sam.mask_decoder.parameters()), lr=1e-5
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            if not batch:
                logging.warning(f"Empty batch at index {batch_idx}, skipping")
                continue
            images, masks, prompts = zip(*batch)
            images = torch.stack(images).to(device)
            masks = torch.stack(masks).to(device)
            prompts = torch.stack(prompts).to(device)  # [batch_size, 4]

            batch_loss = 0
            for i in range(len(images)):
                with autocast():
                    image_embedding = sam.image_encoder(images[i].unsqueeze(0))
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=prompts[i].unsqueeze(0).unsqueeze(0),  # [1, 1, 4]
                        masks=None,
                    )
                    low_res_masks, _ = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    predicted_masks = torch.nn.functional.interpolate(
                        low_res_masks, size=(1024, 1024), mode='bicubic', align_corners=False
                    )
                    loss = criterion(predicted_masks.squeeze(1), masks[i].unsqueeze(0))
                    batch_loss += loss

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += batch_loss.item()
            logging.debug(f"Batch {batch_idx} - Loss: {batch_loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(sam.state_dict(), "checkpoints/best_fine_tuned_sam.pth")
            logging.info("Model checkpoint saved due to improved loss")
        else:
            patience_counter += 1
            logging.info(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

    torch.save(sam.state_dict(), "checkpoints/fine_tuned_sam.pth")
    logging.info("Final fine-tuned model saved")

def segment_face(image_path, model_checkpoint):
    sam = sam_model_registry["vit_b"](checkpoint=model_checkpoint)
    sam.load_state_dict(torch.load("checkpoints/fine_tuned_sam.pth"))
    sam.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    logging.info(f"Inference model loaded and moved to {device}")

    predictor = SamPredictor(sam)
    detector = MTCNN()
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image at {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if not faces:
        logging.warning("No face detected!")
        return
    x, y, w, h = faces[0]['box']
    padding = 0.3  # Increased padding for better context
    x = max(0, x - int(w * padding))
    y = max(0, y - int(h * padding))
    w = min(image.shape[1] - x, int(w * (1 + 2 * padding)))
    h = min(image.shape[0] - y, int(h * (1 + 2 * padding)))
    prompt = np.array([x, y, x + w, y + h])
    logging.debug(f"Detected face prompt: {prompt}")

    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(box=prompt, multimask_output=False)
    mask = masks[0].astype(np.float32)
    
    # Post-process mask to smooth boundaries and reduce artifacts
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    result = image.copy()
    result[mask == 0] = [255, 255, 255]  # Set background to white
    output_path = "output.jpg"
    cv2.imwrite(output_path, result)
    logging.info(f"Output saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM for face segmentation")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--image_dir", default="data/train/images")
    parser.add_argument("--mask_dir", default="data/train/masks")
    parser.add_argument("--prompt_dir", default="data/train/prompts")
    parser.add_argument("--model_checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--image_path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    if args.mode == "train":
        fine_tune_sam(
            args.model_checkpoint, args.image_dir, args.mask_dir, args.prompt_dir,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
        )
    elif args.mode == "infer":
        if not args.image_path:
            logging.error("Please provide an image path for inference!")
        else:
            segment_face(args.image_path, args.model_checkpoint)