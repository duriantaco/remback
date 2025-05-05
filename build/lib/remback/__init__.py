from .remover import BackgroundRemover

def remove_background(image_path, output_path=None, fine_tuned_checkpoint=None):
    """Convenience function to remove background from an image."""
    remover = BackgroundRemover(fine_tuned_checkpoint)
    return remover.remove_background(image_path, output_path)