"""Single-image adversarial generation for DocSAF."""

import torch
import argparse
from pathlib import Path
from PIL import Image
import logging

from .utils import (
    load_config,
    setup_logging,
    get_device,
    tensor_to_pil,
    pil_to_tensor,
    save_image_tensor,
    compute_lpips_score,
)
from .surrogates import load_embedder
from .ocr import ocr_read
from .saliency import compute_gradient_saliency
from .field import apply_field_safe
from .pdf_io import pdf_to_pil, is_pdf_file

logger = logging.getLogger(__name__)


def load_universal_params(params_path: str) -> dict:
    """Load universal parameters from saved checkpoint.

    Args:
        params_path: Path to universal.pt file

    Returns:
        Dictionary with alpha and radius parameters
    """
    if not Path(params_path).exists():
        logger.warning(
            f"Universal params not found: {params_path}. Using default values."
        )
        return {"alpha": 1.2, "radius": 7.0}

    try:
        params = torch.load(params_path, map_location="cpu")
        if isinstance(params, dict) and "alpha" in params and "radius" in params:
            return params
        else:
            logger.warning("Invalid params format. Using defaults.")
            return {"alpha": 1.2, "radius": 7.0}
    except Exception as e:
        logger.warning(f"Failed to load params: {e}. Using defaults.")
        return {"alpha": 1.2, "radius": 7.0}


def infer_single_image(
    image_path: str,
    config: dict,
    params: dict,
    target_text: str = None,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Generate adversarial image for single input.

    Args:
        image_path: Path to input image or PDF
        config: Configuration dictionary
        params: Universal parameters (alpha, radius)
        target_text: Optional target text (if None, uses OCR)
        device: Computation device

    Returns:
        Tuple of (original_tensor, adversarial_tensor, stats_dict)
    """
    alpha = params["alpha"]
    radius = params["radius"]

    logger.info(f"Processing: {image_path}")
    logger.info(f"Using alpha={alpha:.3f}, radius={radius:.3f}")

    # Load image
    if is_pdf_file(image_path):
        logger.info("Converting PDF to image...")
        pil_image = pdf_to_pil(image_path, page=0, zoom=2.0)
    else:
        pil_image = Image.open(image_path).convert("RGB")

    # Convert to tensor
    x_orig = pil_to_tensor(pil_image, device)

    # Extract OCR text
    if target_text is None:
        import numpy as np

        img_array = np.array(pil_image)
        ocr_backend = config.get("ocr", "easyocr")
        text = ocr_read(img_array, backend=ocr_backend)
        logger.info(f"Extracted text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    else:
        text = target_text
        logger.info(f"Using target text: '{text}'")

    if not text.strip():
        logger.warning("No text extracted/provided. Using fallback text.")
        text = "document text content"

    # Load embedder
    surrogate_specs = config.get("surrogates", ["openclip:ViT-L-14@336"])
    embedder_spec = (
        surrogate_specs[0] if isinstance(surrogate_specs, list) else surrogate_specs
    )
    embedder = load_embedder(embedder_spec, device)

    # Compute saliency
    logger.info("Computing cross-modal saliency...")
    x_input = x_orig.clone().requires_grad_(True)

    alignment_score, saliency_map = compute_gradient_saliency(
        embedder, x_input, text, normalize=True
    )

    logger.info(f"Original alignment score: {alignment_score:.4f}")

    # Apply attenuation field
    logger.info("Applying attenuation field...")
    x_adv = apply_field_safe(x_orig, saliency_map, alpha, radius)

    # Compute adversarial alignment (no gradients needed for evaluation)
    with torch.no_grad():
        # Get embeddings directly since we don't need gradients for evaluation
        adv_img_emb = embedder.image_embed(x_adv)
        adv_txt_emb = embedder.text_embed([text])
        adv_alignment = float((adv_img_emb * adv_txt_emb).sum(dim=-1).mean())
    logger.info(f"Adversarial alignment score: {adv_alignment:.4f}")

    # Compute LPIPS
    try:
        lpips_score = compute_lpips_score(x_orig, x_adv, device=device)
        logger.info(f"LPIPS perceptual distance: {lpips_score:.4f}")
    except ImportError:
        logger.warning("LPIPS not available")
        lpips_score = None

    # Gather stats
    stats = {
        "original_alignment": alignment_score,
        "adversarial_alignment": adv_alignment,
        "alignment_drop": alignment_score - adv_alignment,
        "lpips_distance": lpips_score,
        "saliency_stats": {
            "min": float(saliency_map.min()),
            "max": float(saliency_map.max()),
            "mean": float(saliency_map.mean()),
        },
    }

    return x_orig, x_adv, stats


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="DocSAF single-image inference")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to input image or PDF"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="runs/universal.pt",
        help="Path to universal parameters",
    )
    parser.add_argument(
        "--output", type=str, help="Output path (auto-generated if not provided)"
    )
    parser.add_argument(
        "--target-text", type=str, help="Target text (uses OCR if not provided)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cuda/cpu)"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load config and params
    config = load_config(args.config)
    params = load_universal_params(args.params)

    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.image)
        output_path = input_path.parent / f"{input_path.stem}_adv{input_path.suffix}"
        if input_path.suffix.lower() == ".pdf":
            output_path = input_path.parent / f"{input_path.stem}_adv.png"
    else:
        output_path = Path(args.output)

    try:
        # Run inference
        x_orig, x_adv, stats = infer_single_image(
            args.image, config, params, args.target_text, device
        )

        # Save adversarial image
        save_image_tensor(x_adv, output_path)

        # Print results
        print(f"\n=== DocSAF Inference Results ===")
        print(f"Input: {args.image}")
        print(f"Output: {output_path}")
        print(f"Original alignment: {stats['original_alignment']:.4f}")
        print(f"Adversarial alignment: {stats['adversarial_alignment']:.4f}")
        print(f"Alignment drop: {stats['alignment_drop']:.4f}")
        if stats["lpips_distance"] is not None:
            print(f"LPIPS distance: {stats['lpips_distance']:.4f}")
        print(f"Success: Adversarial image saved!")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
