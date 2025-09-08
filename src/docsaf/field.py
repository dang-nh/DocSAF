"""Attenuation field implementation for DocSAF."""

import torch
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def apply_field(
    x: torch.Tensor, 
    S: torch.Tensor, 
    alpha: float, 
    radius: float
) -> torch.Tensor:
    """Apply attenuation field to input image.
    
    The field performs local defocus blending:
    A_θ(x) = σ(alpha * (Gaussian(radius) * S(x)))
    x' = (1 - A) ⊙ x + A ⊙ blur(x)
    
    Args:
        x: Input image tensor (B, 3, H, W) in [0, 1]
        S: Saliency map (B, 1, H, W) in [0, 1]  
        alpha: Field strength parameter
        radius: Gaussian blur radius (pixels)
        
    Returns:
        Adversarial image (B, 3, H, W) with same range as input
        
    Raises:
        ValueError: If input dimensions are invalid
    """
    if x.dim() != 4 or S.dim() != 4:
        raise ValueError(f"Expected 4D tensors, got x: {x.shape}, S: {S.shape}")
    
    if x.shape[0] != S.shape[0] or x.shape[2:] != S.shape[2:]:
        raise ValueError(f"Batch/spatial dimensions mismatch: x: {x.shape}, S: {S.shape}")
    
    try:
        import kornia.filters
    except ImportError:
        raise ImportError("Kornia not installed. Run: pip install kornia")
    
    # Compute blur parameters
    kernel_size = int(max(3, 2 * int(radius) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    sigma = max(0.5, radius / 3.0)
    
    # Apply Gaussian blur
    x_blur = kornia.filters.gaussian_blur2d(
        x, 
        kernel_size=(kernel_size, kernel_size), 
        sigma=(sigma, sigma)
    )
    
    # Compute attenuation mask (gamma = 1.0 fixed)
    A = torch.sigmoid(alpha * S)  # (B, 1, H, W)
    
    # Blend original and blurred images
    x_adv = (1 - A) * x + A * x_blur
    
    return x_adv


def field_with_smoothing(
    x: torch.Tensor,
    S: torch.Tensor, 
    alpha: float,
    radius: float,
    smooth_radius: float = 2.0
) -> torch.Tensor:
    """Apply field with additional saliency smoothing.
    
    Args:
        x: Input image (B, 3, H, W)
        S: Saliency map (B, 1, H, W)
        alpha: Field strength
        radius: Blur radius for defocus
        smooth_radius: Additional smoothing radius for saliency
        
    Returns:
        Adversarial image with smoothed saliency field
    """
    try:
        import kornia.filters
    except ImportError:
        raise ImportError("Kornia not installed. Run: pip install kornia")
    
    # Smooth saliency map first
    smooth_kernel = int(max(3, 2 * int(smooth_radius) + 1))
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
        
    S_smooth = kornia.filters.gaussian_blur2d(
        S,
        kernel_size=(smooth_kernel, smooth_kernel),
        sigma=(smooth_radius, smooth_radius)
    )
    
    # Apply field with smoothed saliency
    return apply_field(x, S_smooth, alpha, radius)


def compute_tv_loss(field_mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute total variation loss for field regularization.
    
    Args:
        field_mask: Attenuation mask (B, 1, H, W)
        reduction: Loss reduction ("mean", "sum", "none")
        
    Returns:
        TV loss scalar or per-batch tensor
    """
    if field_mask.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape: {field_mask.shape}")
    
    # Compute TV via finite differences
    diff_h = torch.abs(field_mask[:, :, 1:, :] - field_mask[:, :, :-1, :])
    diff_w = torch.abs(field_mask[:, :, :, 1:] - field_mask[:, :, :, :-1])
    
    tv_loss = diff_h.mean() + diff_w.mean()
    
    if reduction == "mean":
        return tv_loss
    elif reduction == "sum":
        return tv_loss * field_mask.shape[0]
    elif reduction == "none":
        return tv_loss.expand(field_mask.shape[0])
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def field_stats(x_orig: torch.Tensor, x_adv: torch.Tensor) -> dict:
    """Compute statistics of field application.
    
    Args:
        x_orig: Original image (B, 3, H, W)
        x_adv: Adversarial image (B, 3, H, W)
        
    Returns:
        Dictionary with perturbation statistics
    """
    diff = (x_adv - x_orig).detach().cpu()
    
    return {
        "l2_norm": float(torch.norm(diff, p=2)),
        "linf_norm": float(torch.norm(diff, p=float('inf'))),
        "mean_abs_diff": float(diff.abs().mean()),
        "max_abs_diff": float(diff.abs().max()),
        "affected_pixels": float((diff.abs() > 1e-6).float().mean()),
    }


def validate_field_params(alpha: float, radius: float) -> None:
    """Validate field parameters.
    
    Args:
        alpha: Field strength
        radius: Blur radius
        
    Raises:
        ValueError: If parameters are out of valid range
    """
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"Alpha must be numeric, got: {type(alpha)}")
    
    if not isinstance(radius, (int, float)):
        raise ValueError(f"Radius must be numeric, got: {type(radius)}")
    
    if alpha < 0:
        raise ValueError(f"Alpha must be non-negative, got: {alpha}")
    
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got: {radius}")
    
    if alpha > 10:
        logger.warning(f"Large alpha value ({alpha}) may cause severe distortion")
    
    if radius > 20:
        logger.warning(f"Large radius value ({radius}) may cause excessive blur")


def apply_field_safe(
    x: torch.Tensor,
    S: torch.Tensor, 
    alpha: float,
    radius: float
) -> torch.Tensor:
    """Safe wrapper for apply_field with parameter validation.
    
    Args:
        x: Input image tensor
        S: Saliency map
        alpha: Field strength 
        radius: Blur radius
        
    Returns:
        Adversarial image tensor
        
    Raises:
        ValueError: If parameters or inputs are invalid
    """
    validate_field_params(alpha, radius)
    return apply_field(x, S, alpha, radius)
