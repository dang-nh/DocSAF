"""Cross-modal saliency computation for DocSAF."""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional
import logging

from .surrogates import ImageTextEmbedder, CLIPEmbedder

logger = logging.getLogger(__name__)


def compute_gradient_saliency(
    embedder: ImageTextEmbedder, 
    image: torch.Tensor, 
    text: str,
    normalize: bool = True
) -> Tuple[float, torch.Tensor]:
    """Compute gradient-based cross-modal saliency.
    
    Args:
        embedder: Image-text embedder (must support gradients)
        image: Input tensor (B, 3, H, W) requiring gradients
        text: Text string for alignment
        normalize: Whether to normalize saliency to [0, 1]
        
    Returns:
        (alignment_score, saliency_map) where:
        - alignment_score: Cosine similarity scalar
        - saliency_map: (B, 1, H, W) saliency in [0, 1] if normalized
        
    Raises:
        ValueError: If image doesn't require gradients
    """
    if not image.requires_grad:
        raise ValueError("Input image must require gradients for saliency computation")
    
    # Get embeddings
    img_emb = embedder.image_embed(image)  # (B, D)
    txt_emb = embedder.text_embed([text])  # (1, D)
    
    # Ensure normalized (cosine similarity)
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    
    # Compute alignment (cosine similarity)
    alignment = (img_emb * txt_emb).sum(dim=-1).mean()  # Scalar for backprop
    
    # Backpropagate to get gradients
    alignment.backward(retain_graph=True)
    
    # Extract gradients and compute saliency
    grads = image.grad.detach()  # (B, 3, H, W)
    saliency = grads.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    if normalize:
        # Normalize to [0, 1] per batch item
        for i in range(saliency.shape[0]):
            s = saliency[i]
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                saliency[i] = (s - s_min) / (s_max - s_min)
            else:
                saliency[i] = torch.zeros_like(s)
    
    return float(alignment.item()), saliency


def clip_alignment_and_saliency(
    model: CLIPEmbedder,
    pil_image: Image.Image, 
    text: str, 
    device: str = "cuda"
) -> Tuple[float, torch.Tensor]:
    """Convenient wrapper for CLIP gradient saliency.
    
    Args:
        model: CLIPEmbedder instance
        pil_image: PIL Image
        text: Text string
        device: Target device
        
    Returns:
        (alignment_score, saliency_map)
    """
    # Preprocess image and enable gradients
    x = model.preprocess_image(pil_image).to(device)
    x.requires_grad_(True)
    
    return compute_gradient_saliency(model, x, text)


def gradcam_saliency(
    embedder: ImageTextEmbedder,
    image: torch.Tensor,
    text: str,
    target_layer: Optional[str] = None
) -> Tuple[float, torch.Tensor]:
    """Grad-CAM based saliency (optional alternative).
    
    Args:
        embedder: Image-text embedder with accessible layers
        image: Input tensor (B, 3, H, W)
        text: Text string
        target_layer: Layer name for CAM (model-specific)
        
    Returns:
        (alignment_score, saliency_map)
        
    Note:
        This is a placeholder for future Grad-CAM integration.
        Current implementation falls back to gradient saliency.
    """
    logger.warning("Grad-CAM saliency not fully implemented. Using gradient saliency.")
    
    if not image.requires_grad:
        image = image.requires_grad_(True)
    
    return compute_gradient_saliency(embedder, image, text)


def multi_scale_saliency(
    embedder: ImageTextEmbedder,
    image: torch.Tensor,
    text: str,
    scales: Tuple[float, ...] = (0.8, 1.0, 1.2)
) -> Tuple[float, torch.Tensor]:
    """Multi-scale gradient saliency for robustness.
    
    Args:
        embedder: Image-text embedder
        image: Input tensor (B, 3, H, W)
        text: Text string
        scales: Scale factors for multi-scale computation
        
    Returns:
        (avg_alignment, aggregated_saliency)
    """
    B, C, H, W = image.shape
    device = image.device
    
    alignments = []
    saliencies = []
    
    for scale in scales:
        # Resize image
        new_h, new_w = int(H * scale), int(W * scale)
        scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        scaled_img = scaled_img.requires_grad_(True)
        
        # Compute saliency
        align, sal = compute_gradient_saliency(embedder, scaled_img, text)
        
        # Resize saliency back to original size
        sal_resized = F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=False)
        
        alignments.append(align)
        saliencies.append(sal_resized)
    
    # Average alignments and saliencies
    avg_alignment = sum(alignments) / len(alignments)
    avg_saliency = torch.stack(saliencies).mean(dim=0)
    
    return avg_alignment, avg_saliency


def saliency_stats(saliency_map: torch.Tensor) -> dict:
    """Compute statistics of saliency map for debugging.
    
    Args:
        saliency_map: (B, 1, H, W) tensor
        
    Returns:
        Dictionary with min, max, mean, std statistics
    """
    sal = saliency_map.detach().cpu()
    
    return {
        "min": float(sal.min()),
        "max": float(sal.max()),
        "mean": float(sal.mean()),
        "std": float(sal.std()),
        "shape": tuple(sal.shape)
    }