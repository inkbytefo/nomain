from pathlib import Path
from typing import Tuple
import numpy as np

def load_mnist(data_dir: str = "~/.psinet_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST train images and labels using torchvision with local caching.

    Args:
        data_dir: Base directory to cache datasets. Will create ~/.psinet_data/mnist by default.

    Returns:
        images: np.ndarray of shape (N, 28, 28), dtype uint8
        labels: np.ndarray of shape (N,), dtype uint8/int64
    """
    data_path = Path(data_dir).expanduser().resolve() / 'mnist'
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        import torchvision
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError(
            "torchvision is required for load_mnist. Please ensure it is installed."
        ) from e

    transform = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.MNIST(root=str(data_path), train=True, download=True, transform=transform)

    # Convert to numpy (uint8 images in 0..255)
    images = (ds.data.numpy()).astype(np.uint8)
    labels = ds.targets.numpy()
    return images, labels
