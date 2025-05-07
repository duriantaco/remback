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

def combine_pairs(test_dir,
                  result_dir,
                  output_path,
                  num_pairs     = 12,
                  cols          = 3,
                  margin        = 10,
                  border_margin = 10,
                  image_size    = (300, 300),
                  gap           = 5):
    """
    test_dir    : folder that holds test1.jpg, test2.jpg, …
    result_dir  : folder that holds output1.jpg, output2.jpg, …
    output_path : where the grid will be written
    """

    def load_sq(path):
        img = Image.open(path)
        w, h = img.size
        d = min(w, h)
        l = (w - d) // 2
        t = (h - d) // 2
        return img.crop((l, t, l + d, t + d)).resize(image_size, Image.LANCZOS)

    W, H      = image_size
    pair_w    = 2 * W + gap
    rows      = (num_pairs + cols - 1) // cols
    grid_w    = cols * pair_w + (cols - 1) * margin
    grid_h    = rows * H      + (rows - 1) * margin
    canvas_w  = grid_w + 2 * border_margin
    canvas_h  = grid_h + 2 * border_margin
    canvas    = Image.new("RGB", (canvas_w, canvas_h), "white")

    for i in range(1, num_pairs + 1):
        test_img   = load_sq(os.path.join(test_dir,   f"test{i}.jpg"))
        output_img = load_sq(os.path.join(result_dir, f"output{i}.jpg"))

        pair = Image.new("RGB", (pair_w, H), "white")
        pair.paste(test_img,   (0, 0))
        pair.paste(output_img, (W + gap, 0))

        r = (i - 1) // cols
        c = (i - 1) % cols
        x = border_margin + c * (pair_w + margin)
        y = border_margin + r * (H      + margin)
        canvas.paste(pair, (x, y))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print("saved to", output_path)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    combine_pairs(
        test_dir   = "/home/aaron_monarch/remback/assets/test_images",
        result_dir = "/home/aaron_monarch/remback/assets/output_results",
        output_path= "/home/aaron_monarch/remback/assets/combined_images/combined_grid.jpg",
        num_pairs  = 12,
        cols       = 3
    )
