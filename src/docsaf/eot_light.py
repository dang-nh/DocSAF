"""Light EOT (Expectation Over Transformations) for DocSAF."""

import torch
import numpy as np
from typing import Tuple, Optional
import cv2
import logging

logger = logging.getLogger(__name__)


def eot_light(
    x: np.ndarray,
    jpeg_q_min: int = 50,
    jpeg_q_max: int = 90,
    resize_min: float = 0.9,
    resize_max: float = 1.1,
    eot_prob: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """Apply light EOT transformations: JPEG compression + resize jitter.
    
    Args:
        x: Input image array (H, W, 3) in uint8 [0, 255]
        jpeg_q_min: Minimum JPEG quality
        jpeg_q_max: Maximum JPEG quality
        resize_min: Minimum resize scale factor
        resize_max: Maximum resize scale factor
        eot_prob: Probability of applying each transform
        seed: Random seed for reproducibility
        
    Returns:
        Transformed image array (H, W, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = x.shape[:2]
    result = x.copy()
    
    # Apply JPEG compression
    if np.random.rand() < eot_prob:
        quality = np.random.randint(jpeg_q_min, jpeg_q_max + 1)
        result = apply_jpeg_compression(result, quality)
    
    # Apply resize jitter
    if np.random.rand() < eot_prob:
        scale = np.random.uniform(resize_min, resize_max)
        result = apply_resize_jitter(result, scale)
    
    return result


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression to image.
    
    Args:
        image: Input image (H, W, 3) uint8
        quality: JPEG quality (0-100)
        
    Returns:
        Compressed image
    """
    try:
        # Encode to JPEG bytes
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        
        # Decode back to image
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return decoded_img
    except Exception as e:
        logger.warning(f"JPEG compression failed: {e}. Returning original image.")
        return image


def apply_resize_jitter(image: np.ndarray, scale: float) -> np.ndarray:
    """Apply resize jitter (scale then back to original size).
    
    Args:
        image: Input image (H, W, 3)
        scale: Scale factor
        
    Returns:
        Resized image with original dimensions
    """
    try:
        h, w = image.shape[:2]
        
        # Scale to new size
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Scale back to original size
        result = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return result
    except Exception as e:
        logger.warning(f"Resize jitter failed: {e}. Returning original image.")
        return image


def eot_light_tensor(
    x: torch.Tensor,
    jpeg_q_min: int = 50,
    jpeg_q_max: int = 90,
    resize_min: float = 0.9,
    resize_max: float = 1.1,
    eot_prob: float = 0.5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Tensor wrapper for EOT-light transformations.
    
    Args:
        x: Input tensor (B, 3, H, W) in [0, 1]
        jpeg_q_min: Minimum JPEG quality
        jpeg_q_max: Maximum JPEG quality  
        resize_min: Minimum resize scale
        resize_max: Maximum resize scale
        eot_prob: Transform probability
        seed: Random seed
        
    Returns:
        Transformed tensor (B, 3, H, W)
    """
    device = x.device
    batch_size = x.shape[0]
    
    results = []
    
    for i in range(batch_size):
        # Convert tensor to numpy uint8
        img_tensor = x[i].cpu()  # (3, H, W)
        img_array = (img_tensor.permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()
        
        # Apply EOT transformations
        img_eot = eot_light(
            img_array,
            jpeg_q_min=jpeg_q_min,
            jpeg_q_max=jpeg_q_max,
            resize_min=resize_min,
            resize_max=resize_max,
            eot_prob=eot_prob,
            seed=seed + i if seed is not None else None
        )
        
        # Convert back to tensor
        img_tensor_eot = torch.from_numpy(img_eot).float() / 255.0
        img_tensor_eot = img_tensor_eot.permute(2, 0, 1)  # (3, H, W)
        
        results.append(img_tensor_eot)
    
    return torch.stack(results).to(device)


def eot_light_albumentations(
    x: np.ndarray,
    jpeg_q_min: int = 50,
    jpeg_q_max: int = 90,
    resize_min: float = 0.9,
    resize_max: float = 1.1,
    eot_prob: float = 0.5
) -> np.ndarray:
    """Alternative implementation using Albumentations (if available).
    
    Args:
        x: Input image array (H, W, 3)
        jpeg_q_min: Minimum JPEG quality
        jpeg_q_max: Maximum JPEG quality
        resize_min: Minimum resize scale
        resize_max: Maximum resize scale
        eot_prob: Transform probability
        
    Returns:
        Transformed image array
    """
    try:
        import albumentations as A
    except ImportError:
        logger.warning("Albumentations not available, falling back to OpenCV implementation")
        return eot_light(x, jpeg_q_min, jpeg_q_max, resize_min, resize_max, eot_prob)
    
    h, w = x.shape[:2]
    
    # Create augmentation pipeline
    transforms = A.Compose([
        A.ImageCompression(
            quality_lower=jpeg_q_min,
            quality_upper=jpeg_q_max,
            p=eot_prob
        ),
        A.RandomScale(
            scale_limit=(resize_min - 1.0, resize_max - 1.0),
            p=eot_prob
        ),
        A.Resize(height=h, width=w, p=1.0)  # Ensure output size matches input
    ])
    
    try:
        result = transforms(image=x)["image"]
        return result
    except Exception as e:
        logger.warning(f"Albumentations transform failed: {e}. Returning original image.")
        return x


def compute_robustness_score(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    num_trials: int = 10,
    eot_config: dict = None
) -> float:
    """Compute robustness score against EOT transformations.
    
    Args:
        original: Original image tensor (1, 3, H, W)
        adversarial: Adversarial image tensor (1, 3, H, W)
        num_trials: Number of EOT trials
        eot_config: EOT configuration dict
        
    Returns:
        Average robustness score (lower = more robust)
    """
    if eot_config is None:
        eot_config = {
            "jpeg_q_min": 50,
            "jpeg_q_max": 90,
            "resize_min": 0.9,
            "resize_max": 1.1,
            "eot_prob": 0.8
        }
    
    scores = []
    
    for trial in range(num_trials):
        # Apply EOT to adversarial image
        adv_eot = eot_light_tensor(adversarial, seed=trial, **eot_config)
        
        # Compute difference (L2 norm as proxy for robustness)
        diff = torch.norm(adv_eot - adversarial, p=2).item()
        scores.append(diff)
    
    return np.mean(scores)


def eot_stats(original: np.ndarray, transformed: np.ndarray) -> dict:
    """Compute statistics of EOT transformation.
    
    Args:
        original: Original image array
        transformed: Transformed image array
        
    Returns:
        Statistics dictionary
    """
    diff = transformed.astype(np.float32) - original.astype(np.float32)
    
    return {
        "l2_norm": float(np.linalg.norm(diff)),
        "linf_norm": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "psnr": compute_psnr(original, transformed),
        "ssim": compute_ssim(original, transformed),
    }


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (simplified)."""
    try:
        from skimage.metrics import structural_similarity
        if len(img1.shape) == 3:
            return structural_similarity(img1, img2, multichannel=True, channel_axis=2, data_range=255)
        else:
            return structural_similarity(img1, img2, data_range=255)
    except ImportError:
        logger.warning("scikit-image not available for SSIM computation")
        return 0.0
