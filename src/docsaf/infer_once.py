"""Single-image adversarial generation for DocSAF."""

import torch
import typer
from pathlib import Path
from PIL import Image
import logging
from typing import Optional

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


app = typer.Typer(
    name="docsaf-infer",
    help="DocSAF single-image adversarial generation using attenuation fields with TWO KNOBS ONLY: alpha (field strength) and radius (blur kernel).",
    epilog="""
Examples:
  docsaf-infer --image demo/invoice.png
  docsaf-infer --image document.pdf --page 0 --output out/adv.png
  docsaf-infer --image form.jpg --alpha 1.5 --radius 10
  
The DocSAF attack uses ONLY TWO tunable parameters to maintain simplicity:
- alpha: Field strength (default from config)  
- radius: Gaussian blur radius (default from config)
"""
)


@app.command()
def main(
    image: str = typer.Option(..., "--image", help="Path to input image or PDF"),
    config: str = typer.Option("configs/default.yaml", "--config", help="Path to config file"),
    params: str = typer.Option("runs/universal.pt", "--params", help="Path to universal parameters (alpha, radius)"),
    output: Optional[str] = typer.Option(None, "--output", help="Output path (auto-generated if not provided)"),
    target_text: Optional[str] = typer.Option(None, "--target-text", help="Target text (uses OCR if not provided)"),
    device: str = typer.Option("auto", "--device", help="Device: auto/cuda/cpu"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level: DEBUG/INFO/WARNING/ERROR"),
    alpha: Optional[float] = typer.Option(None, "--alpha", help="Override alpha (field strength) from config"),
    radius: Optional[float] = typer.Option(None, "--radius", help="Override radius (blur kernel) from config"),
):
    """
    Generate adversarial image using DocSAF attenuation fields.
    
    The DocSAF method uses ONLY TWO KNOBS for simplicity:
    - alpha: Controls the strength of the attenuation field  
    - radius: Controls the Gaussian blur kernel size
    
    No additional trainable parameters or loss weights are used.
    """
    # Validate inputs early
    if not Path(image).exists():
        typer.echo(f"❌ Error: Image file not found: {image}", err=True)
        raise typer.Exit(1)
        
    if not Path(config).exists():
        typer.echo(f"❌ Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)
        
    if not Path(params).exists():
        typer.echo(f"❌ Error: Parameters file not found: {params}", err=True)
        typer.echo(f"💡 Hint: Run training first or use default parameters", err=True)
        raise typer.Exit(1)

    # Setup
    try:
        setup_logging(log_level)
        device = get_device(device)
        logger.info(f"Using device: {device}")

        # Load config and params
        config_data = load_config(config)
        params_data = load_universal_params(params)
        
        # Override parameters if provided (enforce two-knob constraint)
        if alpha is not None:
            if alpha < 0:
                typer.echo(f"❌ Error: Alpha must be non-negative, got {alpha}", err=True)
                raise typer.Exit(1)
            params_data["alpha"] = alpha
            typer.echo(f"🔧 Overriding alpha: {alpha}")
            
        if radius is not None:
            if radius <= 0:
                typer.echo(f"❌ Error: Radius must be positive, got {radius}", err=True)
                raise typer.Exit(1)
            params_data["radius"] = radius
            typer.echo(f"🔧 Overriding radius: {radius}")
            
        # Enforce two-knob invariant
        expected_keys = {"alpha", "radius"}
        extra_keys = set(params_data.keys()) - expected_keys
        if extra_keys:
            typer.echo(f"❌ Error: DocSAF uses ONLY two knobs (alpha, radius). Found extra parameters: {extra_keys}", err=True)
            typer.echo(f"💡 This violates the two-knob design constraint.", err=True)
            raise typer.Exit(1)

        # Generate output path if not provided  
        if output is None:
            input_path = Path(image)
            output_path = input_path.parent / f"{input_path.stem}_adv{input_path.suffix}"
            if input_path.suffix.lower() == ".pdf":
                output_path = input_path.parent / f"{input_path.stem}_adv.png"
        else:
            output_path = Path(output)
            
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run inference
        typer.echo(f"🚀 Starting DocSAF inference...")
        typer.echo(f"📄 Input: {image}")
        typer.echo(f"💾 Output: {output_path}")
        typer.echo(f"🎛️  Parameters: alpha={params_data['alpha']:.3f}, radius={params_data['radius']:.3f}")
        
        x_orig, x_adv, stats = infer_single_image(
            image, config_data, params_data, target_text, device
        )

        # Save adversarial image
        save_image_tensor(x_adv, output_path)

        # Print results with nice formatting
        typer.echo("\n" + "="*50)
        typer.echo("🎯 DocSAF Inference Results")
        typer.echo("="*50)
        typer.echo(f"📄 Input:              {image}")
        typer.echo(f"💾 Output:             {output_path}")
        typer.echo(f"🎛️  Alpha (strength):   {params_data['alpha']:.3f}")
        typer.echo(f"🎛️  Radius (blur):      {params_data['radius']:.3f}")
        typer.echo(f"📊 Original alignment:  {stats['original_alignment']:.4f}")
        typer.echo(f"📊 Adversarial align:   {stats['adversarial_alignment']:.4f}")
        typer.echo(f"📉 Alignment drop:      {stats['alignment_drop']:.4f}")
        if stats["lpips_distance"] is not None:
            typer.echo(f"🖼️  LPIPS distance:      {stats['lpips_distance']:.4f}")
            
        if stats['alignment_drop'] > 0:
            typer.echo(f"✅ Success: Attack achieved positive alignment drop!")
        else:
            typer.echo(f"⚠️  Warning: Attack achieved negative alignment drop")
            
        typer.echo(f"💫 Adversarial image saved successfully!")

    except Exception as e:
        typer.echo(f"❌ Inference failed: {e}", err=True)
        logger.error(f"Detailed error: {e}", exc_info=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
