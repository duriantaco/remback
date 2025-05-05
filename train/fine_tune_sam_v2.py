import os, cv2, json, argparse, logging, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
from mtcnn import MTCNN
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("gx_base", gx)
        self.register_buffer("gy_base", gy)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, p, t):
        gx, gy = self.gx_base.to(p), self.gy_base.to(p)
        p = torch.sigmoid(p)
        ex = torch.abs(nn.functional.conv2d(p, gx, padding=1))
        ey = torch.abs(nn.functional.conv2d(p, gy, padding=1))
        et = (torch.abs(nn.functional.conv2d(t, gx, padding=1)) +
              torch.abs(nn.functional.conv2d(t, gy, padding=1)) > 0).to(p)
        return self.bce(ex + ey, et)


bnd_loss = BoundaryLoss()


def dice_loss(p, y, eps=1e-6):
    if y.dim() == 3:
        y = y.unsqueeze(1)
    p = torch.sigmoid(p)
    inter = (p * y).flatten(1).sum(1)
    union = p.flatten(1).sum(1) + y.flatten(1).sum(1)
    return (1 - (2 * inter + eps) / (union + eps)).mean()

class CelebAMaskHQDataset(Dataset):
    def __init__(self, iroot, mroot, proot):
        self.iroot, self.mroot, self.proot = iroot, mroot, proot
        self.files = sorted(f for f in os.listdir(iroot) if f.endswith(".jpg"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        name = self.files[k]
        img = cv2.imread(os.path.join(self.iroot, name))
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mroot, name.replace(".jpg", ".png"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (1024, 1024),
                          interpolation=cv2.INTER_NEAREST) if mask is not None else np.zeros((1024, 1024), np.uint8)

        prompt_path = os.path.join(self.proot, name.replace(".jpg", ".json"))
        if not os.path.exists(prompt_path):
            return None
        pj = json.load(open(prompt_path))

        # parse bbox
        pr = None
        if isinstance(pj, list) and pj:
            a = pj[0] if isinstance(pj[0], (list, tuple)) else pj
            if len(a) >= 4:
                pr = np.asarray(a[:4], np.float32)
        elif isinstance(pj, dict):
            vals = [pj.get(k) for k in
                    ("x", "y", "x2", "y2", "xmin", "ymin", "xmax", "ymax")
                    if pj.get(k) is not None]
            if len(vals) >= 4:
                pr = np.asarray(vals[:4], np.float32)
        if pr is None or pr.shape[0] != 4:
            return None

        return (torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255,
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255,
                torch.tensor(pr, dtype=torch.float32))

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return tuple(torch.stack(z) for z in zip(*batch))

def clean_mask(prob: np.ndarray, box: np.ndarray, guide_rgb1024: np.ndarray) -> np.ndarray:
    """
    prob : (1024,1024) float32  -  probability map from SAM
    box  : [x0,y0,x1,y1]  -  bbox that should be kept
    """

    mask = (prob > 0.5).astype(np.uint8)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        area_bx = (box[2] - box[0]) * (box[3] - box[1])
        best, best_iou = None, 0
        for i in range(1, n):
            x, y, w, h = st[i, :4]
            bb = (x, y, x + w, y + h)
            ix0, iy0 = max(box[0], bb[0]), max(box[1], bb[1])
            ix1, iy1 = min(box[2], bb[2]), min(box[3], bb[3])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                iou = inter / (area_bx + w * h - inter + 1e-6)
                if iou > best_iou:
                    best_iou, best = iou, i
        mask = (lab == best).astype(np.uint8) if best is not None else np.zeros_like(mask)

    guide_gray = cv2.cvtColor(guide_rgb1024, cv2.COLOR_RGB2GRAY)
    mask = cv2.ximgproc.guidedFilter(guide_gray, mask.astype(np.float32), 16, 1e-3)

    hard = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 3)
    alpha = np.clip(1 - dist / 8.0, 0, 1)

    return alpha.astype(np.float32)

def fine_tune_sam(ckpt, img_dir, msk_dir, prm_dir,
                  epochs, batch_size, patience):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt).to(dev)

    for p in sam.image_encoder.parameters(): p.requires_grad = False
    for blk in sam.image_encoder.blocks[-2:]:
        for p in blk.parameters(): p.requires_grad = True

    dl = DataLoader(CelebAMaskHQDataset(img_dir, msk_dir, prm_dir),
                    batch_size=batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate)

    params = (list(sam.prompt_encoder.parameters()) +
              list(sam.mask_decoder.parameters()) +
              [p for p in sam.image_encoder.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=1e-4)
    scaler = GradScaler()

    best, wait = 1e9, 0
    sam.train()

    for ep in range(epochs):
        epoch_loss = 0

        for batch in tqdm(dl, leave=False, total=len(dl)):
            if batch is None:
                continue

            imgs, masks, bxs = [t.to(dev) for t in batch]
            loss = 0

            for i in range(imgs.size(0)):
                with autocast():
                    emb = sam.image_encoder(imgs[i:i + 1])
                    sparse, _ = sam.prompt_encoder(
                        points=None,
                        boxes=bxs[i].view(1, 1, 4),
                        masks=None,
                    )
                    blank = torch.zeros_like(emb)
                    logits, _ = sam.mask_decoder(
                        image_embeddings=emb,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=blank,
                        multimask_output=False,
                    )

                    up = torch.nn.functional.interpolate(
                        logits, (1024, 1024),
                        mode="bicubic", align_corners=False,
                        antialias=True,
                    )

                    tgt = masks[i:i + 1]
                    loss += (
                        0.35 * nn.functional.binary_cross_entropy_with_logits(up, tgt)
                        + 0.35 * dice_loss(up, tgt)
                        + 0.30 * bnd_loss(up, tgt)
                    )

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dl)
        logging.info(f"epoch {ep + 1}   loss {avg:.4f}")

        if avg < best:
            best, wait = avg, 0
            torch.save(sam.state_dict(), "checkpoints/best_fine_tuned_sam.pth")
        else:
            wait += 1
            if wait >= patience:
                break

    torch.save(sam.state_dict(), "checkpoints/fine_tuned_sam_last.pth")

def segment_face(img_path, ckpt):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.load_state_dict(torch.load("checkpoints/best_fine_tuned_sam.pth",
                                   map_location=dev))
    sam.eval().to(dev)

    predictor = SamPredictor(sam)
    mtcnn = MTCNN()

    bgr = cv2.imread(img_path);  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    faces = mtcnn.detect_faces(rgb)
    if not faces: raise RuntimeError("no face detected")

    x, y, w, h = faces[0]["box"];  pad = 0.35
    x = max(0, int(x - w * pad));  y = max(0, int(y - h * pad))
    w = min(rgb.shape[1] - x, int(w * (1 + 2 * pad)))
    h = min(rgb.shape[0] - y, int(h * (1 + 2 * pad)))
    box = np.array([x, y, x + w, y + h])

    predictor.set_image(rgb)
    _, _, logits = predictor.predict(box=box, multimask_output=False)

    prob = 1 / (1 + np.exp(-logits[0]))
    prob = cv2.resize(prob, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    alpha = clean_mask(prob, box, cv2.resize(rgb, (1024, 1024)))

    rgba = np.dstack([bgr, (alpha * 255).astype(np.uint8)])
    cv2.imwrite("output.png", rgba)
    preview = (alpha[..., None] * bgr + (1 - alpha[..., None]) * 255).astype(np.uint8)
    cv2.imwrite("output_white.jpg", preview)
    logging.info("saved output.png + output_white.jpg")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", choices=["train", "infer"], required=True)
    pa.add_argument("--image_dir", default="data/train/images")
    pa.add_argument("--mask_dir", default="data/train/masks")
    pa.add_argument("--prompt_dir", default="data/train/prompts")
    pa.add_argument("--model_checkpoint",
                    default="checkpoints/sam_vit_b_01ec64.pth")
    pa.add_argument("--image_path")
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--epochs", type=int, default=5)
    pa.add_argument("--patience", type=int, default=2)
    args = pa.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    if args.mode == "train":
        fine_tune_sam(args.model_checkpoint,
                      args.image_dir, args.mask_dir, args.prompt_dir,
                      args.epochs, args.batch_size, args.patience)
    else:
        if not args.image_path:
            logging.error("--image_path required in infer mode")
        else:
            segment_face(args.image_path, args.model_checkpoint)
