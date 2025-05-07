import cv2, torch, numpy as np
from remback.sam_models.predictor import SamPredictor
from remback.sam_models.build_sam import sam_model_registry
from mtcnn import MTCNN
from remback.utils import get_checkpoint_path

class BackgroundRemover:
    def __init__(self, checkpoint=None):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model_registry["vit_b"]()
        if checkpoint is None:
            checkpoint = get_checkpoint_path()
        self.sam.load_state_dict(torch.load(checkpoint, map_location=self.dev))
        self.sam.to(self.dev).eval()
        self.pred = SamPredictor(self.sam)
        self.det = MTCNN()

    def _mask_from_face(self, rgb):
        det = self.det.detect_faces(rgb)
        if not det:
            raise ValueError("No face detected")
        x, y, w, h = det[0]["box"]

        pad_w, pad_up, pad_down = 1.20, 0.05, 0.20
        x0 = max(0, int(x - pad_w * w))
        y0 = max(0, int(y - pad_up * h))
        x1 = min(rgb.shape[1], int(x + w * (1 + pad_w)))
        y1 = min(rgb.shape[0], int(y + h * (1 + pad_down)))
        box = np.array([x0, y0, x1, y1], dtype=np.int32)

        self.pred.set_image(rgb)
        m, _, _ = self.pred.predict(box=box, multimask_output=False)
        m = (m[0] > 0.1).astype(np.uint8)

        _, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if stats.shape[0] > 1:
            keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            m = (lbl == keep).astype(np.uint8)

        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k5, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=1)
        
        return m

    def _post_edit(self, bgr, sharp, contrast):
        if sharp:
            k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            bgr = cv2.filter2D(bgr, -1, k)
        if contrast != 1.0:
            bgr = cv2.convertScaleAbs(bgr, alpha=contrast, beta=0)
        return bgr

    def remove_background(self, img_path, out_path=None, sharp=0, contrast=1.0):
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = self._mask_from_face(rgb)

        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.5)
        res = bgr.copy()
        res[a < 0.40] = 255
        res = self._post_edit(res, sharp, contrast)

        if out_path:
            cv2.imwrite(out_path, res)
        return res

def RemoveBackground(inp, out=None, ckpt=None):
    return BackgroundRemover(ckpt).remove_background(inp, out)
