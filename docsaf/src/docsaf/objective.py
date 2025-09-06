"""Objective function for DocSAF universal parameter training."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging

from .surrogates import ImageTextEmbedder, load_embedder
from .saliency import compute_gradient_saliency
from .field import apply_field, compute_tv_loss
from .eot_light import eot_light_tensor

logger = logging.getLogger(__name__)


def alignment_collapse_loss(
    embedders: List[ImageTextEmbedder],
    x_adv: torch.Tensor,
    texts: List[str],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute alignment collapse loss across multiple embedders.

    Self-supervised objective to minimize cross-modal alignment between
    adversarial images and their OCR text.

    Args:
        embedders: List of image-text embedders for ensemble attack
        x_adv: Adversarial image batch (B, 3, H, W)
        texts: List of text strings (length B)
        reduction: Loss reduction ("mean", "sum", "none")

    Returns:
        Alignment loss tensor
    """
    if len(texts) != x_adv.shape[0]:
        raise ValueError(
            f"Batch size mismatch: {len(texts)} texts, {x_adv.shape[0]} images"
        )

    total_alignment = 0.0

    for embedder in embedders:
        # Get embeddings
        img_emb = embedder.image_embed(x_adv)  # (B, D)
        txt_emb = embedder.text_embed(texts)  # (B, D)

        # Ensure normalized (cosine similarity)
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        # Compute per-sample cosine similarity
        alignment = (img_emb * txt_emb).sum(dim=-1)  # (B,)

        # Accumulate across embedders
        total_alignment = total_alignment + alignment

    # Average across embedders
    avg_alignment = total_alignment / len(embedders)

    # Apply reduction
    if reduction == "mean":
        return avg_alignment.mean()
    elif reduction == "sum":
        return avg_alignment.sum()
    elif reduction == "none":
        return avg_alignment
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def targeted_alignment_loss(
    embedders: List[ImageTextEmbedder],
    x_adv: torch.Tensor,
    original_texts: List[str],
    target_texts: List[str],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute targeted alignment loss.

    Objective: minimize alignment with original text while maximizing
    alignment with target text.

    Args:
        embedders: List of embedders
        x_adv: Adversarial images (B, 3, H, W)
        original_texts: Original OCR texts
        target_texts: Target texts for adversarial alignment
        reduction: Loss reduction

    Returns:
        Targeted loss tensor
    """
    if (
        len(original_texts) != len(target_texts)
        or len(original_texts) != x_adv.shape[0]
    ):
        raise ValueError("Text lists and batch size must match")

    total_orig_align = 0.0
    total_target_align = 0.0

    for embedder in embedders:
        img_emb = embedder.image_embed(x_adv)

        # Original text alignment (minimize)
        orig_txt_emb = embedder.text_embed(original_texts)
        orig_align = (
            F.normalize(img_emb, dim=-1) * F.normalize(orig_txt_emb, dim=-1)
        ).sum(dim=-1)
        total_orig_align = total_orig_align + orig_align

        # Target text alignment (maximize, so negate)
        target_txt_emb = embedder.text_embed(target_texts)
        target_align = (
            F.normalize(img_emb, dim=-1) * F.normalize(target_txt_emb, dim=-1)
        ).sum(dim=-1)
        total_target_align = total_target_align + target_align

    # Average across embedders
    avg_orig_align = total_orig_align / len(embedders)
    avg_target_align = total_target_align / len(embedders)

    # Combined loss: minimize original, maximize target
    loss = avg_orig_align - avg_target_align

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def docsaf_objective(
    alpha: torch.Tensor,
    radius: torch.Tensor,
    images: torch.Tensor,
    texts: List[str],
    embedders: List[ImageTextEmbedder],
    saliency_maps: torch.Tensor,
    tv_lambda: float = 1e-3,
    eot_prob: float = 0.0,
    eot_config: Optional[Dict] = None,
    targeted_texts: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Main DocSAF objective function.

    Combines alignment collapse loss with TV regularization.

    Args:
        alpha: Field strength parameter (learnable)
        radius: Gaussian radius parameter (learnable, positive via softplus)
        images: Batch of original images (B, 3, H, W)
        texts: List of OCR text strings
        embedders: List of surrogate embedders
        saliency_maps: Pre-computed saliency maps (B, 1, H, W)
        tv_lambda: TV regularization weight
        eot_prob: Probability of applying EOT transforms
        eot_config: EOT configuration
        targeted_texts: Optional target texts for targeted attack

    Returns:
        (total_loss, loss_components_dict)
    """
    batch_size = images.shape[0]
    device = images.device

    # Ensure positive radius via softplus
    radius_pos = F.softplus(radius)

    # Apply attenuation field
    x_adv = apply_field(images, saliency_maps, alpha, radius_pos)

    # Apply EOT transforms if specified
    if eot_prob > 0 and eot_config is not None:
        x_adv_eot = eot_light_tensor(x_adv, eot_prob=eot_prob, **eot_config)
    else:
        x_adv_eot = x_adv

    # Compute alignment loss
    if targeted_texts is not None:
        alignment_loss = targeted_alignment_loss(
            embedders, x_adv_eot, texts, targeted_texts
        )
    else:
        alignment_loss = alignment_collapse_loss(embedders, x_adv_eot, texts)

    # Compute TV regularization on attenuation mask
    attenuation_mask = torch.sigmoid(alpha * saliency_maps)
    tv_loss = compute_tv_loss(attenuation_mask, reduction="mean")

    # Total loss
    total_loss = alignment_loss + tv_lambda * tv_loss

    # Detailed loss components
    loss_components = {
        "total_loss": total_loss,
        "alignment_loss": alignment_loss,
        "tv_loss": tv_loss,
        "alpha": alpha,
        "radius_pos": radius_pos,
    }

    return total_loss, loss_components


class DocSAFObjective:
    """Wrapper class for DocSAF objective with state management."""

    def __init__(
        self, embedder_specs: List[str], tv_lambda: float = 1e-3, device: str = "cuda"
    ):
        """Initialize objective.

        Args:
            embedder_specs: List of embedder specification strings
            tv_lambda: TV regularization weight
            device: Computation device
        """
        self.tv_lambda = tv_lambda
        self.device = device

        # Load embedders
        self.embedders = []
        for spec in embedder_specs:
            try:
                embedder = load_embedder(spec, device)
                self.embedders.append(embedder)
                logger.info(f"Loaded embedder: {spec}")
            except Exception as e:
                logger.warning(f"Failed to load embedder {spec}: {e}")

        if not self.embedders:
            raise ValueError("No embedders loaded successfully")

    def compute_loss(
        self,
        alpha: torch.Tensor,
        radius: torch.Tensor,
        images: torch.Tensor,
        texts: List[str],
        saliency_maps: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute objective loss.

        Args:
            alpha: Field strength parameter
            radius: Gaussian radius parameter
            images: Input images
            texts: OCR texts
            saliency_maps: Saliency maps
            **kwargs: Additional arguments for docsaf_objective

        Returns:
            (loss, components_dict)
        """
        return docsaf_objective(
            alpha=alpha,
            radius=radius,
            images=images,
            texts=texts,
            embedders=self.embedders,
            saliency_maps=saliency_maps,
            tv_lambda=self.tv_lambda,
            **kwargs,
        )

    def evaluate_alignment(
        self, images: torch.Tensor, texts: List[str]
    ) -> Dict[str, float]:
        """Evaluate alignment scores for images and texts.

        Args:
            images: Image batch (B, 3, H, W)
            texts: Text list

        Returns:
            Dictionary with alignment statistics
        """
        alignments = []

        with torch.no_grad():
            for embedder in self.embedders:
                img_emb = F.normalize(embedder.image_embed(images), dim=-1)
                txt_emb = F.normalize(embedder.text_embed(texts), dim=-1)
                alignment = (img_emb * txt_emb).sum(dim=-1)
                alignments.append(alignment.cpu().numpy())

        import numpy as np

        alignments = np.array(alignments)  # (num_embedders, batch_size)

        return {
            "mean_alignment": float(alignments.mean()),
            "std_alignment": float(alignments.std()),
            "min_alignment": float(alignments.min()),
            "max_alignment": float(alignments.max()),
            "per_embedder_mean": alignments.mean(axis=1).tolist(),
        }


def validate_objective_inputs(
    alpha: torch.Tensor,
    radius: torch.Tensor,
    images: torch.Tensor,
    texts: List[str],
    saliency_maps: torch.Tensor,
) -> None:
    """Validate inputs to objective function.

    Args:
        alpha: Field strength
        radius: Blur radius
        images: Image batch
        texts: Text list
        saliency_maps: Saliency maps

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(alpha, torch.Tensor) or alpha.numel() != 1:
        raise ValueError(f"Alpha must be scalar tensor, got: {alpha}")

    if not isinstance(radius, torch.Tensor) or radius.numel() != 1:
        raise ValueError(f"Radius must be scalar tensor, got: {radius}")

    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"Images must be (B,3,H,W), got: {images.shape}")

    if len(texts) != images.shape[0]:
        raise ValueError(f"Text count ({len(texts)}) != batch size ({images.shape[0]})")

    if saliency_maps.shape != (images.shape[0], 1, images.shape[2], images.shape[3]):
        raise ValueError(
            f"Saliency shape mismatch: expected {(images.shape[0], 1, images.shape[2], images.shape[3])}, got {saliency_maps.shape}"
        )

    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"Empty/invalid text at index {i}: '{text}'")


def compute_alignment_drop(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    texts: List[str],
    embedders: List[ImageTextEmbedder],
) -> float:
    """Compute average alignment drop from original to adversarial.

    Args:
        original_images: Original images (B, 3, H, W)
        adversarial_images: Adversarial images (B, 3, H, W)
        texts: Text strings
        embedders: List of embedders

    Returns:
        Average alignment drop (positive = successful attack)
    """
    with torch.no_grad():
        orig_alignment = 0.0
        adv_alignment = 0.0

        for embedder in embedders:
            # Original alignment
            orig_img_emb = F.normalize(embedder.image_embed(original_images), dim=-1)
            txt_emb = F.normalize(embedder.text_embed(texts), dim=-1)
            orig_align = (orig_img_emb * txt_emb).sum(dim=-1).mean()
            orig_alignment += orig_align

            # Adversarial alignment
            adv_img_emb = F.normalize(embedder.image_embed(adversarial_images), dim=-1)
            adv_align = (adv_img_emb * txt_emb).sum(dim=-1).mean()
            adv_alignment += adv_align

        # Average across embedders
        orig_alignment /= len(embedders)
        adv_alignment /= len(embedders)

        return float(orig_alignment - adv_alignment)
