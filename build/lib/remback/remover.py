import cv2, torch, numpy as np
from remback.sam_models.predictor import SamPredictor
from remback.sam_models.build_sam import sam_model_registry
from mtcnn import MTCNN
from remback.utils import get_checkpoint_path

class BackgroundRemover:
    def __init__(self, checkpoint: str | None = None):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model_registry["vit_b"]()
        if checkpoint is None:
            checkpoint = get_checkpoint_path()
        self.sam.load_state_dict(torch.load(checkpoint, map_location=self.dev))
        self.sam.to(self.dev).eval()

        self.pred = SamPredictor(self.sam)
        self.det = MTCNN()

    def _soft_alpha_from_face(self, rgb: np.ndarray) -> np.ndarray:
        # shrunk a little so hair/shoulders still included zzz
        dets = self.det.detect_faces(rgb)
        if not dets:
            raise ValueError("no face detected")

        x, y, w, h = dets[0]["box"]
        shrink = 0.10
        x += int(w * shrink)
        y += int(h * shrink)
        w = int(w * (1 - 2 * shrink))
        h = int(h * (1 - 2 * shrink))
        box = np.array([x, y, x + w, y + h])

        self.pred.set_image(rgb)
        prob, *_ = self.pred.predict(box=box, multimask_output=False)
        mask = (prob[0] > 0.85).astype(np.uint8)

        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n > 1:
            keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (lab == keep).astype(np.uint8)

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)

        dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        alpha = np.clip(1 - dist / 8.0, 0, 1).astype(np.float32)
        return alpha

    def _post_edit(self, bgr: np.ndarray,
                   sharp: bool = False,
                   contrast: float = 1.0) -> np.ndarray:
        if sharp:
            k = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], np.float32)
            bgr = cv2.filter2D(bgr, -1, k)
        if contrast != 1.0:
            bgr = cv2.convertScaleAbs(bgr, alpha=contrast, beta=0)
        return bgr

    def remove_background(
        self,
        img_path: str,
        out_path: str | None = None,
        *,
        sharp: bool = False,
        contrast: float = 1.0
    ) -> np.ndarray:
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        alpha = self._soft_alpha_from_face(rgb)
        out = bgr.copy()
        out[alpha < 0.5] = 255
        out = self._post_edit(out, sharp, contrast)

        if out_path:
            cv2.imwrite(out_path, out)
        return out

    def get_mask(self, img_path: str) -> np.ndarray:
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._soft_alpha_from_face(rgb)


def RemoveBackground(inp: str,
                     out: str | None = None,
                     ckpt: str | None = None,
                     *,
                     sharp: bool = False,
                     contrast: float = 1.0):
    return BackgroundRemover(ckpt).remove_background(
        inp, out, sharp=sharp, contrast=contrast
    )
