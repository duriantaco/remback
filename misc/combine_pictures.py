from PIL import Image
import os

def crop_to_square(img):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))

def combine_pairs(input_dir, output_path, num_pairs=12, cols=3, margin=10, border_margin=10, image_size=(300, 300), margin_between_images=5):
    processed_images = {}
    
    for i in range(1, num_pairs + 1):
        test_img = crop_to_square(Image.open(os.path.join(input_dir, f'test{i}.jpg')))
        test_img = test_img.resize(image_size, Image.LANCZOS)
        
        output_img = crop_to_square(Image.open(os.path.join(input_dir, f'output{i}.jpg')))
        output_img = output_img.resize(image_size, Image.LANCZOS)
        
        processed_images[f'test{i}'] = test_img
        processed_images[f'output{i}'] = output_img
    
    W, H = image_size 
    pair_width = 2 * W + margin_between_images
    rows = (num_pairs + cols - 1) // cols
    grid_width = cols * pair_width + (cols - 1) * margin
    grid_height = rows * H + (rows - 1) * margin
    final_width = grid_width + 2 * border_margin
    final_height = grid_height + 2 * border_margin
    
    final_image = Image.new('RGB', (final_width, final_height), (255, 255, 255))
    
    for idx, pair_num in enumerate(range(1, num_pairs + 1)):
        test_img = processed_images[f'test{pair_num}']
        output_img = processed_images[f'output{pair_num}']
        
        pair_img = Image.new('RGB', (pair_width, H), (255, 255, 255))
        pair_img.paste(test_img, (0, 0))
        pair_img.paste(output_img, (W + margin_between_images, 0))
        
        row = idx // cols
        col = idx % cols
        x = border_margin + col * (pair_width + margin)
        y = border_margin + row * (H + margin)
        
        final_image.paste(pair_img, (x, y))
    
    final_image.save(output_path)

if __name__ == "__main__":
    input_dir = '/path/to/your/input/directory' 
    output_path = '/path/to/output/remback/combined_grid.jpg'
    combine_pairs(input_dir, output_path, num_pairs=12, cols=3, margin=10, border_margin=10, image_size=(300, 300), margin_between_images=5)