import argparse
from remback.remover import BackgroundRemover
import sys
import logging

logger = logging.getLogger(__name__)

if not (3, 10) <= sys.version_info[:2] < (3, 12):
    logger.error("This script requires Python 3.10 or 3.11.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", default="output.jpg", help="Path to save the output image")
    parser.add_argument("--checkpoint", help="Path to the fine-tuned checkpoint")
    parser.add_argument("--sharpen", type=int, default=0, help="0/1 or strength")
    parser.add_argument("--contrast", type=float, default=1.0, help=">1.0 = more contrast")

    args = parser.parse_args()

    remover = BackgroundRemover(checkpoint=args.checkpoint)
    remover.remove_background(args.image_path, args.output_path)

if __name__ == "__main__":
    main()