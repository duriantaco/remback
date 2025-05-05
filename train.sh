python fine_tune_sam_v2.py \
       --task infer \
       --sam_ckpt checkpoints/sam_vit_b_01ec64.pth \
       --ref_ckpt checkpoints/refiner.pth \
       --img path/to/photo.jpg
