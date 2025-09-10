"""Attenuation field implementation for DocSAF."""

import torch
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def _gauss1d(sigma, K=21):
    if sigma.ndim == 0:
        sigma = sigma[None]
    B = sigma.shape[0]
    x = torch.arange(K, device=sigma.device, dtype=sigma.dtype) - (K - 1) / 2
    s = torch.clamp(sigma, min=0.5, max=10.0)
    g = torch.exp(-0.5 * (x[None, :] / s[:, None]) ** 2)  # (B, K)
    g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
    return g


def _gauss2d_kernel(sigma, K=21, C=1):
    g1 = _gauss1d(sigma, K)      # (B, K)
    g2 = g1[:, :, None] * g1[:, None, :]  # (B, K, K)
    kernel = g2[:, None, :, :].repeat(1, C, 1, 1)  # (B, C, K, K)
    return kernel


def _depthwise_conv2d(x, kernel):
    B, C, H, W = x.shape
    K = kernel.shape[-1]
    pad = K // 2
    weight = kernel.view(B * C, 1, K, K)
    y = F.conv2d(x.view(1, B * C, H, W), weight=weight, padding=pad, groups=B * C)
    return y.view(B, C, H, W)


def apply_field(images, saliency_maps, alpha, radius_sigma, K=21):
    """
    images: (B,3,H,W), saliency_maps: (B,1,H,W) in [0,1]
    alpha, radius_sigma: scalar tensors (requires_grad=True)
    returns: x_adv, A
    """
    B, C, H, W = images.shape
    sigma = (radius_sigma.reshape(1) + 1e-3).expand(B)  # keep differentiable

    # Smooth saliency
    k_s = _gauss2d_kernel(sigma, K=K, C=1)
    S_smooth = _depthwise_conv2d(saliency_maps, k_s)  # (B, 1, H, W)

    # Center & scale to avoid sigmoid saturation
    S_centered = S_smooth - S_smooth.mean(dim=(2, 3), keepdim=True)
    S_norm = S_centered / (S_centered.std(dim=(2, 3), keepdim=True) + 1e-6)

    # Mask (depends on alpha)
    A = torch.sigmoid(alpha * S_norm)  # (B, 1, H, W)

    # Blur image (depends on sigma)
    k_x = _gauss2d_kernel(sigma, K=K, C=C)
    x_blur = _depthwise_conv2d(images, k_x)  # (B, 3, H, W)

    # Compose
    x_adv = (1 - A) * images + A * x_blur
    return x_adv, A


def compute_tv_loss(x, reduction="mean"):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    if reduction == "mean":
        return dx.abs().mean() + dy.abs().mean()
    return dx.abs().sum() + dy.abs().sum()


def field_with_smoothing(
    x: torch.Tensor,
    S: torch.Tensor,
    alpha: float,
    radius: float,
    smooth_radius: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply field with additional saliency smoothing.

    Args:
        x: Input image (B, 3, H, W)
        S: Saliency map (B, 1, H, W)
        alpha: Field strength
        radius: Blur radius for defocus
        smooth_radius: Additional smoothing radius for saliency

    Returns:
        Tuple of (adversarial image, mask A)
    """
    try:
        import kornia.filters
    except ImportError:
        raise ImportError("Kornia not installed. Run: pip install kornia")

    # Smooth saliency map first with bounds checking
    H, W = S.shape[-2:]
    max_kernel_size = min(H, W) // 2 * 2 - 1
    max_kernel_size = max(3, max_kernel_size)

    smooth_kernel = int(max(3, 2 * int(smooth_radius) + 1))
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1

    smooth_kernel = min(smooth_kernel, max_kernel_size)

    S_smooth = kornia.filters.gaussian_blur2d(
        S,
        kernel_size=(smooth_kernel, smooth_kernel),
        sigma=(smooth_radius, smooth_radius),
    )

    # Apply field with smoothed saliency
    return apply_field(x, S_smooth, alpha, radius)


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
        "linf_norm": float(torch.norm(diff, p=float("inf"))),
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
    if not isinstance(alpha, (int, float, torch.Tensor)):
        raise ValueError(f"Alpha must be numeric or tensor, got: {type(alpha)}")

    if not isinstance(radius, (int, float, torch.Tensor)):
        raise ValueError(f"Radius must be numeric or tensor, got: {type(radius)}")

    if isinstance(alpha, torch.Tensor):
        alpha_val = alpha.item() if alpha.numel() == 1 else alpha.min().item()
    else:
        alpha_val = alpha

    if isinstance(radius, torch.Tensor):
        radius_val = radius.item() if radius.numel() == 1 else radius.min().item()
    else:
        radius_val = radius

    if alpha_val < 0:
        raise ValueError(f"Alpha must be non-negative, got: {alpha_val}")

    if radius_val <= 0:
        raise ValueError(f"Radius must be positive, got: {radius_val}")

    if alpha_val > 10:
        logger.warning(f"Large alpha value ({alpha_val}) may cause severe distortion")

    if radius_val > 20:
        logger.warning(f"Large radius value ({radius_val}) may cause excessive blur")


def apply_field_safe(
    x: torch.Tensor, S: torch.Tensor, alpha: torch.Tensor, radius: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safe wrapper for apply_field with parameter validation.

    Args:
        x: Input image tensor
        S: Saliency map
        alpha: Field strength
        radius: Blur radius

    Returns:
        Tuple of (adversarial image, mask A)

    Raises:
        ValueError: If parameters or inputs are invalid
    """
    validate_field_params(alpha, radius)
    return apply_field(x, S, alpha, radius)