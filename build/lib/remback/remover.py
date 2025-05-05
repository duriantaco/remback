import cv2
import torch
import numpy as np
from remback.sam_models.predictor import SamPredictor
from remback.sam_models.build_sam import sam_model_registry
from mtcnn import MTCNN

class BackgroundRemover:
    def __init__(self, fine_tuned_checkpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sam = sam_model_registry["vit_b"]()
        
        if fine_tuned_checkpoint:
            self.sam.load_state_dict(torch.load(fine_tuned_checkpoint, map_location=self.device))
        
        self.sam.to(self.device)
        self.sam.eval()
        
        self.predictor = SamPredictor(self.sam)
        self.detector = MTCNN()

    def remove_background(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = self.detector.detect_faces(image_rgb)
        if not faces:
            raise ValueError("No face detected in the image")
        
        x, y, w, h = faces[0]['box']
        prompt = np.array([x, y, x + w, y + h])
        
        self.predictor.set_image(image_rgb)
        
        masks, _, _ = self.predictor.predict(box=prompt, multimask_output=False)
        mask = masks[0]
        
        result = image.copy()
        result[~mask] = [255, 255, 255]
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Image saved to {output_path}")
        else:
            return result

    def get_mask(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = self.detector.detect_faces(image_rgb)
        if not faces:
            raise ValueError("No face detected in the image")
        
        x, y, w, h = faces[0]['box']
        prompt = np.array([x, y, x + w, y + h])
        
        self.predictor.set_image(image_rgb)
        
        masks, _, _ = self.predictor.predict(box=prompt, multimask_output=False)
        return masks[0]
    
def RemoveBackground(image_path, output_path=None, 
                    fine_tuned_checkpoint="train/checkpoints/best_fine_tuned_sam.pth"):
    remover = BackgroundRemover(fine_tuned_checkpoint)
    return remover.remove_background(image_path, output_path)

#### eg use 
if __name__ == "__main__":
    RemoveBackground("test.jpg", "output.jpg")