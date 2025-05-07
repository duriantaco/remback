from huggingface_hub import hf_hub_download
import os

def get_checkpoint_path():
    """Retrieve or download the fine-tuned checkpoint."""
    checkpoint_dir = os.path.join(os.path.expanduser("~"), ".remback")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "remback_v4.pth")
    if not os.path.exists(checkpoint_path):
        hf_hub_download(
            repo_id="duriantaco/remback",
            filename="remback_v4.pth",
            local_dir=checkpoint_dir
        )
    return checkpoint_path