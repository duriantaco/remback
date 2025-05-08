# Remback

A Python package for removing backgrounds from profile pictures using a fine-tuned Segment Anything Model (SAM).

### Requirements

1. Python 3.9 - 3.11
2. Dependencies (installed automatically): torch, opencv-python, numpy, mtcnn, segment-anything.

**Note**: Do make sure your environment is using `python 3.9 to 3.11`. You can set up your env as such, `python3.11 -m venv venv311`.
`Python 3.13` will **NOT** work because tensorflow does not support it yet. 

## Installation

### Installation of package
`pip install remback`

### Manual retrieval of checkpoint
`https://huggingface.co/duriantaco/remback/tree/main`

Note: It should automatically download the checkpoint when you run this `remback --image_path /path/to/input.jpg --output_path /path/to/output.jpg`, but if you do run into a `SSL` error, just manually download it from the path above.

## Usage

### Command-Line Interface (CLI)

Remove the background from an image:

```bash
remback --image_path /path/to/input.jpg --output_path /path/to/output.jpg --checkpoint /path/to/checkpoint.pth
```

* `--image_path`: Path to the input image (required).
* `--output_path`: Path to save the output image (default: output.jpg).

#### CLI flags

| Flag            | Default | Meaning                                     |
|-----------------|---------|---------------------------------------------|
| `--sharp`       | 0       | Unsharp‑mask strength (0 = off)             |
| `--contrast`    | 1.0     | Multiply pixel values after cut‑out         |
| `--debug_mask`  | —       | Path to save the binary mask for inspection |

### Python API

Use it in your Python scripts:

```python
from remback.remover import BackgroundRemover

remover = BackgroundRemover()
remover.remove_background("input.jpg", "output.jpg")
```

## Fine‑Tuning

Remback starts from Meta’s `sam_vit_b` checkpoint and fine‑tunes it **exclusively for portrait / upper‑body shots**.

| Component                | Status                |
|--------------------------|-----------------------|
| Image encoder blocks 0‑8 | **Frozen**            |
| Image encoder blocks 9‑11| **Trainable**         |
| Prompt encoder           | **Trainable**         |
| Mask decoder             | **Trainable**         |

* **Loss mix**

| Loss                | Weight |
|---------------------|-------:|
| Binary‑cross‑entropy| 0.35   |
| Dice                | 0.35   |
| BoundaryLoss*       | 0.30   |

\* `BoundaryLoss` drives sharper transitions by comparing Sobel edges of the logits and ground truth.

* **Optimiser / schedule**

* AdamW (`lr 3 e‑5`, `weight‑decay 1 e‑4`)
* mixed‑precision + GradScaler
* early‑stop on val mIoU (patience = 2 epochs)

## Post‑Processing Pipeline

1. **Prompt box expansion**  
   MTCNN face box is padded  
   `+120 %` left/right, `+5 %` up, `+20 %` down → hair & shoulders included.

2. **Raw mask threshold**  
   `logits > 0.10` → binary mask.

3. **Largest‑component keep**  
   Removes spurious blobs outside the subject.

4. **Morphology**  
   * open (5×5 ellipse, 1 iter) – clears pepper noise  
   * close (5×5 ellipse, 1 iter) – seals pin‑holes

5. **Alpha matt­ing**  
   Gaussian blur (σ ≈ 0.5) then apply:  
   ```python
   res[alpha < 0.40] = 255
   ```

## Comparison to Other Tools

Unlike general-purpose tools like rembg, Remback is optimized for images with faces:

1. Uses `MTCNN` for face detection to guide segmentation.
2. Employs custom `BoundaryLoss` for sharper edges around complex areas like stray hair etc.



## Benchmark Results

### Remback 

![SAM Result](assets/combined_images/combined_grid.jpg)

### Rembg

![SAM Result](assets/combined_images/combined_grid_rembg.jpg)

We tested Remback against other methods. Here’s the table with mIoU and Accuracy (higher is better lah):

| Method          | mIoU   | Accuracy |
|-----------------|--------|----------|
| Remback         | 0.9584 | 0.9696   |
| Original SAM    | 0.3864 | 0.5757   |
| MTCNN           | 0.3164 | 0.4730   |
| Rembg           | 0.8468 | 0.8841   |

### Notes

The fine-tuned model is included in the package.
If no face is detected, it will raise an error.

## Acknowledgments & Licenses

| Dependency | License | Notes |
|------------|---------|-------|
| **Segment Anything (SAM)** © Meta AI | Apache 2.0 | https://github.com/facebookresearch/segment-anything |
| **MTCNN face detector** | MIT | https://github.com/ipazc/mtcnn |
| **PyTorch** | BSD‑style | https://pytorch.org |
| **OpenCV** | Apache 2.0 | https://opencv.org |

Remback only redistributes weights you fine‑tuned yourself; the original SAM
checkpoint is downloaded from the official Meta repository under Apache 2.0.

## To Do

- [] Parameterise thresholds (mask_thresh, alpha_cut) via CLI/‑‑config
- [] Batch mode: accept a folder / glob, stream results
- [] Dynamic quant‑int8 checkpoint & flag --cpu_fast
- [] ONNX export script (remback.export_onnx) + doc
- [] Add hair‑refiner head (1‑layer UNet on top of SAM logits)

## Citation
```
@misc{remback2025,
  title  = {Remback: Face‑aware background removal with a fine‑tuned SAM},
  author = {oha},
  year   = {2025},
  note   = {https://pypi.org/project/remback}
}
```