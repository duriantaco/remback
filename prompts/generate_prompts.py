import cv2
import os
import json
from mtcnn import MTCNN
from tqdm import tqdm

detector = MTCNN()

splits = ['train', 'val']

for split in splits:
    image_dir = f'data/{split}/images'
    prompt_dir = f'data/{split}/prompts'
    os.makedirs(prompt_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images in {split} split to process...")
    
    for img_name in tqdm(image_files, desc=f"Processing {split} images"):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error: Could not load image at {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        
        prompt = []
        if faces:
            x, y, w, h = faces[0]['box']
            prompt = [x, y, x + w, y + h]
            print(f"Success: Face detected in {img_name} - Bounding box: {prompt}")
        else:
            print(f"Warning: No face detected in {img_name}")
        
        prompt_path = os.path.join(prompt_dir, img_name.replace('.jpg', '.json'))
        try:
            with open(prompt_path, 'w') as f:
                json.dump(prompt, f)
            print(f"Success: Saved prompt for {img_name} to {prompt_path}")
        except Exception as e:
            print(f"Error: Failed to save prompt for {img_name} - {str(e)}")

print("Processing complete!")