"""Utility functions for DocSAF."""

import torch
import numpy as np
from PIL import Image
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Union, Tuple, Optional
import random

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML config: {e}")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_spec: str = "auto") -> str:
    """Get appropriate device for computation.
    
    Args:
        device_spec: Device specification ("auto", "cuda", "cpu", "cuda:0", etc.)
        
    Returns:
        Device string
    """
    if device_spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_spec.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        return "cpu"
    else:
        return device_spec


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor (3, H, W) or (1, 3, H, W) in [0, 1]
        
    Returns:
        PIL Image in RGB mode
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) tensor, got shape: {tensor.shape}")
    
    # Convert to [0, 255] uint8
    array = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode='RGB')


def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Convert PIL Image to tensor.
    
    Args:
        image: PIL Image
        device: Target device
        
    Returns:
        Image tensor (1, 3, H, W) in [0, 1]
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_image_tensor(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    """Save image tensor to file.
    
    Args:
        tensor: Image tensor to save
        path: Output file path
    """
    pil_image = tensor_to_pil(tensor)
    pil_image.save(path)
    logger.info(f"Saved image to: {path}")


def load_image_tensor(path: Union[str, Path], device: str = "cpu") -> torch.Tensor:
    """Load image from file as tensor.
    
    Args:
        path: Image file path
        device: Target device
        
    Returns:
        Image tensor (1, 3, H, W) in [0, 1]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    image = Image.open(path).convert('RGB')
    return pil_to_tensor(image, device)


def compute_lpips_score(
    image1: torch.Tensor, 
    image2: torch.Tensor,
    net: str = "vgg",
    device: str = "cuda"
) -> float:
    """Compute LPIPS perceptual distance.
    
    Args:
        image1: First image tensor (1, 3, H, W) in [0, 1]
        image2: Second image tensor (1, 3, H, W) in [0, 1]
        net: LPIPS network ("vgg", "alex", "squeeze")
        device: Computation device
        
    Returns:
        LPIPS distance as float
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("LPIPS not installed. Run: pip install lpips")
    
    # Initialize LPIPS model (cached)
    if not hasattr(compute_lpips_score, "_lpips_models"):
        compute_lpips_score._lpips_models = {}
    
    model_key = f"{net}_{device}"
    if model_key not in compute_lpips_score._lpips_models:
        compute_lpips_score._lpips_models[model_key] = lpips.LPIPS(net=net).to(device)
    
    lpips_model = compute_lpips_score._lpips_models[model_key]
    
    # Compute LPIPS
    with torch.no_grad():
        # LPIPS expects [-1, 1] range
        img1_norm = 2 * image1 - 1
        img2_norm = 2 * image2 - 1
        distance = lpips_model(img1_norm.to(device), img2_norm.to(device))
    
    return float(distance.item())


def batch_lpips_scores(
    original_batch: torch.Tensor,
    adversarial_batch: torch.Tensor,
    net: str = "vgg",
    device: str = "cuda"
) -> np.ndarray:
    """Compute LPIPS scores for a batch of image pairs.
    
    Args:
        original_batch: Original images (B, 3, H, W)
        adversarial_batch: Adversarial images (B, 3, H, W)
        net: LPIPS network
        device: Computation device
        
    Returns:
        LPIPS scores array of shape (B,)
    """
    batch_size = original_batch.shape[0]
    scores = []
    
    for i in range(batch_size):
        score = compute_lpips_score(
            original_batch[i:i+1], 
            adversarial_batch[i:i+1],
            net=net,
            device=device
        )
        scores.append(score)
    
    return np.array(scores)


def print_tensor_stats(tensor: torch.Tensor, name: str = "Tensor") -> None:
    """Print tensor statistics for debugging.
    
    Args:
        tensor: Input tensor
        name: Tensor name for display
    """
    stats = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "requires_grad": tensor.requires_grad,
    }
    
    print(f"{name} Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(directory: Union[str, Path], extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.pdf')) -> list:
    """Get list of image/PDF files in directory.
    
    Args:
        directory: Directory to search
        extensions: File extensions to include
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(files)
