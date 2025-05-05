import cv2, torch, numpy as np, argparse, io
from PIL import Image
from rembg import remove
from remback.sam_models.predictor import SamPredictor
from remback.sam_models.build_sam import sam_model_registry
from mtcnn import MTCNN

def remove_background_sam(img_path, ckpt, out_path):
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(dev).eval()
    pred = SamPredictor(sam)
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pred.set_image(rgb)
    h, w = bgr.shape[:2]
    masks, scores, _ = pred.predict(box=np.array([0, 0, w, h]), multimask_output=True)
    mask = masks[np.argmax(scores)].astype(bool)
    if mask.mean() > .5:
        mask = ~mask
    _, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if stats.shape[0] > 1:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == keep)
    res = bgr.copy()
    res[~mask] = 255
    cv2.imwrite(out_path, res)

def remove_background_mtcnn(img_path, out_path):
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box = MTCNN().detect_faces(rgb)[0]["box"]
    x, y, w, h = box
    m = np.zeros(bgr.shape[:2], bool)
    m[y:y + h, x:x + w] = True
    res = bgr.copy()
    res[~m] = 255
    cv2.imwrite(out_path, res)

def remove_background_rembg(img_path, out_path):
    with open(img_path, "rb") as f:
        out = remove(f.read())
    fg = Image.open(io.BytesIO(out))
    bg = Image.new("RGB", fg.size, (255, 255, 255))
    bg.paste(fg, (0, 0), fg.split()[3] if fg.mode == "RGBA" else None)
    bg.save(out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", required=True)
    p.add_argument("--output_image_path", required=True)
    p.add_argument("--method", choices=["sam", "mtcnn", "rembg"], required=True)
    p.add_argument("--checkpoint")
    a = p.parse_args()
    if a.method == "sam":
        if not a.checkpoint:
            raise ValueError("checkpoint required for sam")
        remove_background_sam(a.image_path, a.checkpoint, a.output_image_path)
    elif a.method == "mtcnn":
        remove_background_mtcnn(a.image_path, a.output_image_path)
    else:
        remove_background_rembg(a.image_path, a.output_image_path)