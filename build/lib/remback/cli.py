import argparse
from remback.remover import BackgroundRemover

def main():
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", default="output.jpg", help="Path to save the output image")
    args = parser.parse_args()

    remover = BackgroundRemover()
    remover.remove_background(args.image_path, args.output_path)

if __name__ == "__main__":
    main()