"""Attenuation field implementation for DocSAF (robust v2)."""

from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F

try:
    import kornia
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False


def _gauss1d(sigma: torch.Tensor, K: int = 21) -> torch.Tensor:
    # sigma: scalar or (B,)
    if sigma.ndim == 0:
        sigma = sigma[None]
    B = sigma.shape[0]
    device, dtype = sigma.device, sigma.dtype
    x = torch.arange(K, device=device, dtype=dtype) - (K - 1) / 2
    s = torch.clamp(sigma, min=0.5, max=10.0)
    g = torch.exp(-0.5 * (x[None, :] / s[:, None]) ** 2)
    g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
    return g  # (B,K)


def _gauss2d_kernel(sigma: torch.Tensor, K: int = 21, C: int = 1) -> torch.Tensor:
    g1 = _gauss1d(sigma, K)                       # (B,K)
    g2 = g1[:, :, None] * g1[:, None, :]          # (B,K,K)
    ker = g2[:, None, :, :].repeat(1, C, 1, 1)    # (B,C,K,K)
    return ker


def _depthwise_conv2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W), kernel: (B,C,K,K)
    B, C, H, W = x.shape
    K = kernel.shape[-1]
    pad = K // 2
    w = kernel.view(B * C, 1, K, K)
    y = F.conv2d(x.view(1, B * C, H, W), weight=w, padding=pad, groups=B * C)
    return y.view(B, C, H, W)


def _robust_scale(S_smooth: torch.Tensor) -> torch.Tensor:
    """
    Robust quantile scaling avoids sigmoid saturation.
    Input:  (B,1,H,W) -> Output in [-1,1]
    """
    B = S_smooth.shape[0]
    flat = S_smooth.view(B, -1)
    p10 = torch.quantile(flat, 0.10, dim=1, keepdim=True).view(B, 1, 1, 1)
    p90 = torch.quantile(flat, 0.90, dim=1, keepdim=True).view(B, 1, 1, 1)
    S_q = (S_smooth - p10) / (p90 - p10 + 1e-6)      # ~[0,1]
    S_tilde = torch.clamp(2.0 * S_q - 1.0, -1.0, 1.0)  # [-1,1]
    return S_tilde


def _build_protect_mask(
    x: torch.Tensor,
    ocr_bboxes: Optional[List[List[Tuple[int, int, int, int]]]] = None,
    edge_thr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns mask for glyph strokes to avoid blurring characters.
    x: (B,3,H,W) in [0,1] or [0,255]
    ocr_bboxes: per-batch list of (x1,y1,x2,y2) in pixel coords (optional)
    """
    if not _HAS_KORNIA:
        # Fallback: if kornia not available, no protection (returns zeros)
        return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)

    B, _, H, W = x.shape
    x_norm = x if x.max() <= 1.1 else x / 255.0
    gray = kornia.color.rgb_to_grayscale(x_norm)                   # (B,1,H,W)
    edges = kornia.filters.sobel(gray).abs().sum(1, keepdim=True)  # (B,1,H,W)

    if edge_thr is None:
        edge_thr = edges.mean(dim=(2, 3), keepdim=True) + 0.5 * edges.std(dim=(2, 3), keepdim=True)
    E = (edges > edge_thr).float()

    M = torch.zeros_like(E)
    if ocr_bboxes is not None:
        # fill OCR boxes
        for b in range(B):
            for (x1, y1, x2, y2) in ocr_bboxes[b]:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W, int(x2)), min(H, int(y2))
                M[b, 0, y1:y2, x1:x2] = 1.0
        # dilate a bit
        if _HAS_KORNIA:
            M = kornia.morphology.dilation(M, torch.ones(1, 1, 5, 5, device=M.device))

    return torch.clamp(M * E, 0, 1)


def apply_field(
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    alpha: torch.Tensor,
    radius_sigma: torch.Tensor,
    K: int = 21,
    ocr_bboxes: Optional[List[List[Tuple[int, int, int, int]]]] = None,
):
    """
    Edge-preserving, glyph-protecting attenuation field.
    Returns: x_adv, A
    """
    B, C, H, W = images.shape
    sigma = (radius_sigma.reshape(1) + 1e-3).expand(B)

    # Smooth saliency
    k_s = _gauss2d_kernel(sigma, K=K, C=1)
    S_smooth = _depthwise_conv2d(saliency_maps, k_s)

    # Robust scaling + sigmoid (keeps gradients alive)
    S_tilde = _robust_scale(S_smooth)              # [-1,1]
    A = torch.sigmoid(alpha * S_tilde)             # (B,1,H,W)

    # Protect glyph strokes inside OCR boxes
    M_protect = _build_protect_mask(images, ocr_bboxes)
    A = A * (1 - M_protect) + 0.05 * M_protect     # keep ~0.05 on strokes

    # Edge-preserving "blur" for background to avoid perceptual artifacts
    if _HAS_KORNIA:
        x_norm = images if images.max() <= 1.1 else images / 255.0
        x_bg = kornia.filters.bilateral_blur(
            x_norm, kernel_size=(K, K), sigma_color=0.1, sigma_space=sigma
        )
        if images.max() > 1.1:
            x_bg = x_bg * 255.0
    else:
        # Fallback to Gaussian if kornia missing
        k_x = _gauss2d_kernel(sigma, K=K, C=C)
        x_bg = _depthwise_conv2d(images, k_x)

    x_adv = (1 - A) * images + A * x_bg
    return x_adv, A


def compute_tv_loss(A: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Total-variation on the final mask A (B,1,H,W)."""
    dx = A[..., :, 1:] - A[..., :, :-1]
    dy = A[..., 1:, :] - A[..., :-1, :]
    if reduction == "mean":
        return dx.abs().mean() + dy.abs().mean()
    return dx.abs().sum() + dy.abs().sum()


def field_stats(x_orig: torch.Tensor, x_adv: torch.Tensor) -> dict:
    diff = (x_adv - x_orig).detach().cpu()
    return {
        "l2_norm": float(torch.norm(diff, p=2)),
        "linf_norm": float(torch.norm(diff, p=float("inf"))),
        "mean_abs_diff": float(diff.abs().mean()),
        "max_abs_diff": float(diff.abs().max()),
        "affected_pixels": float((diff.abs() > 1e-6).float().mean()),
    }