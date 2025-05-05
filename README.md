# Remback

A Python package for removing backgrounds from images using a fine-tuned Segment Anything Model (SAM).

## Installation
`pip install remback`

## Usage

### Command-Line Interface (CLI)

Remove the background from an image:

```bash
remback --image_path /path/to/input.jpg --output_path /path/to/output.jpg
```

`--image_path`: Path to the input image (required).
`--output_path`: Path to save the output image (default: output.jpg).

### Python API

Use it in your Python scripts:

```python
from remback.remover import BackgroundRemover

remover = BackgroundRemover()
remover.remove_background("input.jpg", "output.jpg")
```

### Requirements

Python 3.8+
Dependencies (installed automatically): torch, opencv-python, numpy, mtcnn, segment-anything.

## Benchmark Results

### Remback 

![SAM Result](combined_grid.jpg)

### Segment Anything

![SAM Result](combined_grid_sam.jpg)

We tested Remback against other methods. Here’s the table with mIoU and Accuracy (higher is better lah):

| Method          | mIoU   | Accuracy |
|-----------------|--------|----------|
| Fine-tuned SAM  | 0.9535 | 0.9661   |
| Original SAM    | 0.3934 | 0.5706   |
| MTCNN           | 0.3164 | 0.4730   |
| Rembg           | 0.8468 | 0.8841   |

### Notes

The fine-tuned model is included in the package.
If no face is detected, it will raise an error.

