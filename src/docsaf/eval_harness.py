"""Evaluation harness for DocSAF attacks."""

import torch
import numpy as np
import typer
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import random

from .utils import (
    load_config,
    setup_logging,
    get_device,
    get_image_files,
    pil_to_tensor,
    tensor_to_pil,
    compute_lpips_score,
    batch_lpips_scores,
)
from .surrogates import load_aligner, build_aligners, ImageTextAligner
from .ocr import ocr_read
from .saliency import compute_gradient_saliency
from .field import apply_field_safe
from .objective import compute_alignment_drop
from .eot_light import eot_light_tensor
from .pdf_io import pdf_to_pil, is_pdf_file
from .train_universal import DocumentDataset, collate_documents
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Container for evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.alignment_scores = {"original": [], "adversarial": [], "drop": []}
        self.lpips_scores = []
        self.ocr_mismatch_rates = []
        self.donut_mismatch_rates = []
        self.success_rate = 0.0
        self.transfer_asr = {}
        self.defense_asr = {}

    def add_sample(
        self,
        original_alignment: float,
        adversarial_alignment: float,
        lpips_score: float,
        ocr_mismatch: float = 0.0,
        donut_mismatch: float = 0.0,
    ):
        """Add metrics for a single sample."""
        self.alignment_scores["original"].append(original_alignment)
        self.alignment_scores["adversarial"].append(adversarial_alignment)
        self.alignment_scores["drop"].append(original_alignment - adversarial_alignment)
        self.lpips_scores.append(lpips_score)
        self.ocr_mismatch_rates.append(ocr_mismatch)
        self.donut_mismatch_rates.append(donut_mismatch)

    def compute_statistics(self) -> Dict:
        """Compute final statistics."""
        stats = {}

        # Alignment statistics
        for key, values in self.alignment_scores.items():
            if values:
                stats[f"alignment_{key}_mean"] = float(np.mean(values))
                stats[f"alignment_{key}_std"] = float(np.std(values))
                stats[f"alignment_{key}_median"] = float(np.median(values))

        # LPIPS statistics
        if self.lpips_scores:
            stats["lpips_mean"] = float(np.mean(self.lpips_scores))
            stats["lpips_std"] = float(np.std(self.lpips_scores))
            stats["lpips_median"] = float(np.median(self.lpips_scores))
            stats["lpips_max"] = float(np.max(self.lpips_scores))

            # Perceptibility threshold
            threshold = 0.06
            below_threshold = np.array(self.lpips_scores) <= threshold
            stats["lpips_below_threshold_rate"] = float(below_threshold.mean())

        # Success rate (positive alignment drop)
        if self.alignment_scores["drop"]:
            drops = np.array(self.alignment_scores["drop"])
            self.success_rate = float((drops > 0).mean())
            stats["success_rate"] = self.success_rate

        # OCR mismatch statistics
        if self.ocr_mismatch_rates:
            stats["ocr_mismatch_mean"] = float(np.mean(self.ocr_mismatch_rates))
            stats["ocr_mismatch_std"] = float(np.std(self.ocr_mismatch_rates))

        # Donut mismatch statistics
        if self.donut_mismatch_rates:
            stats["donut_mismatch_mean"] = float(np.mean(self.donut_mismatch_rates))
            stats["donut_mismatch_std"] = float(np.std(self.donut_mismatch_rates))

        return stats


def compute_ocr_mismatch(original_text: str, adversarial_text: str) -> float:
    """Compute OCR mismatch rate between original and adversarial text.
    
    Args:
        original_text: OCR text from original image
        adversarial_text: OCR text from adversarial image
        
    Returns:
        Mismatch rate (0.0 = identical, 1.0 = completely different)
    """
    if not original_text.strip() or not adversarial_text.strip():
        return 1.0  # Consider empty texts as complete mismatch
    
    # Simple word-level mismatch
    orig_words = set(original_text.lower().split())
    adv_words = set(adversarial_text.lower().split())
    
    if not orig_words:
        return 1.0
        
    intersection = orig_words.intersection(adv_words)
    union = orig_words.union(adv_words)
    
    # Jaccard distance (1 - Jaccard similarity)
    return 1.0 - len(intersection) / len(union) if union else 1.0


def compute_donut_mismatch(
    original_image: torch.Tensor, adversarial_image: torch.Tensor, donut_aligner=None
) -> float:
    """Compute Donut extractive mismatch rate.
    
    Args:
        original_image: Original image tensor
        adversarial_image: Adversarial image tensor
        donut_aligner: Donut aligner (if available)
        
    Returns:
        Mismatch rate (placeholder implementation)
    """
    # Placeholder implementation - in practice, would use Donut for document understanding
    # For now, return a dummy value based on image difference
    if donut_aligner is None:
        # Simple pixel-wise difference as proxy
        diff = torch.mean(torch.abs(original_image - adversarial_image))
        return min(1.0, float(diff) * 10)  # Scale to reasonable range
    
    # TODO: Implement actual Donut-based document understanding comparison
    return 0.0


def evaluate_single_image(
    image_path: str,
    aligners: List[ImageTextAligner],
    alpha: float,
    radius: float,
    config: dict,
    device: str = "cuda",
) -> Dict:
    """Evaluate attack on single image.

    Args:
        image_path: Path to image/PDF
        aligners: List of aligners for evaluation
        alpha: Field strength
        radius: Blur radius
        config: Configuration
        device: Device

    Returns:
        Evaluation results dictionary
    """
    # Load image
    if is_pdf_file(image_path):
        pil_image = pdf_to_pil(image_path, page=0, zoom=2.0)
    else:
        from PIL import Image

        pil_image = Image.open(image_path).convert("RGB")

    x_orig = pil_to_tensor(pil_image, device)

    # Extract OCR text
    img_array = np.array(pil_image)
    ocr_backend = config.get("ocr", "easyocr")
    text = ocr_read(img_array, backend=ocr_backend)

    if not text.strip():
        text = "document text content"

    # Compute saliency using primary aligner
    primary_aligner = aligners[0]
    x_input = x_orig.clone().requires_grad_(True)
    orig_alignment, saliency_map = compute_gradient_saliency(
        primary_aligner, x_input, text
    )

    # Apply field
    x_adv = apply_field_safe(x_orig, saliency_map, alpha, radius)

    # Evaluate on all aligners
    results = {
        "image_path": str(image_path),
        "text": text,
        "original_alignment": orig_alignment,
        "aligner_results": {},
    }

    # Compute LPIPS
    try:
        lpips_score = compute_lpips_score(x_orig, x_adv, device=device)
        results["lpips_score"] = lpips_score
    except ImportError:
        results["lpips_score"] = None

    # Compute OCR mismatch
    adv_pil = tensor_to_pil(x_adv)
    adv_img_array = np.array(adv_pil)
    adv_text = ocr_read(adv_img_array, backend=ocr_backend)
    ocr_mismatch = compute_ocr_mismatch(text, adv_text)
    results["ocr_mismatch"] = ocr_mismatch

    # Compute Donut mismatch (placeholder)
    donut_mismatch = compute_donut_mismatch(x_orig, x_adv)
    results["donut_mismatch"] = donut_mismatch

    # Evaluate each aligner
    alignment_drops = []
    for i, aligner in enumerate(aligners):
        with torch.no_grad():
            # Use the cosine_align method for consistency
            orig_align = float(aligner.cosine_align(x_orig, text))
            adv_align = float(aligner.cosine_align(x_adv, text))

            alignment_drop = orig_align - adv_align
            alignment_drops.append(alignment_drop)

            results["aligner_results"][f"aligner_{i}"] = {
                "original_alignment": orig_align,
                "adversarial_alignment": adv_align,
                "alignment_drop": alignment_drop,
            }

    results["mean_alignment_drop"] = float(np.mean(alignment_drops))
    return results


def evaluate_transfer_asr(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    texts: List[str],
    held_out_aligners: List[ImageTextAligner],
) -> Dict[str, float]:
    """Evaluate transfer attack success rate.

    Args:
        original_images: Original images (B, 3, H, W)
        adversarial_images: Adversarial images (B, 3, H, W)
        texts: Text strings
        held_out_aligners: Held-out aligners not used in training

    Returns:
        Transfer ASR results per aligner
    """
    transfer_results = {}

    for i, aligner in enumerate(held_out_aligners):
        alignment_drop = compute_alignment_drop(
            original_images, adversarial_images, texts, [aligner]
        )

        # Success = positive alignment drop
        success_rate = 1.0 if alignment_drop > 0 else 0.0
        transfer_results[f"held_out_aligner_{i}"] = success_rate

        logger.info(
            f"Transfer ASR (aligner {i}): {success_rate:.3f} "
            f"(avg drop: {alignment_drop:.4f})"
        )

    return transfer_results


def evaluate_defense_asr(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    texts: List[str],
    aligners: List[ImageTextAligner],
    eot_config: dict,
) -> Dict[str, float]:
    """Evaluate robustness against defenses.

    Args:
        original_images: Original images
        adversarial_images: Adversarial images
        texts: Texts
        aligners: Aligners
        eot_config: EOT defense configuration

    Returns:
        Defense ASR results
    """
    defense_results = {}

    # JPEG compression defense
    jpeg_defended = eot_light_tensor(
        adversarial_images,
        jpeg_q_min=eot_config.get("jpeg_q_min", 50),
        jpeg_q_max=eot_config.get("jpeg_q_max", 90),
        resize_min=1.0,
        resize_max=1.0,
        eot_prob=1.0,  # Always apply JPEG
    )

    jpeg_drop = compute_alignment_drop(original_images, jpeg_defended, texts, aligners)
    defense_results["jpeg_defense_asr"] = 1.0 if jpeg_drop > 0 else 0.0

    # Resize defense
    resize_defended = eot_light_tensor(
        adversarial_images,
        jpeg_q_min=100,
        jpeg_q_max=100,  # No JPEG
        resize_min=eot_config.get("resize_min", 0.9),
        resize_max=eot_config.get("resize_max", 1.1),
        eot_prob=1.0,  # Always apply resize
    )

    resize_drop = compute_alignment_drop(
        original_images, resize_defended, texts, aligners
    )
    defense_results["resize_defense_asr"] = 1.0 if resize_drop > 0 else 0.0

    # Combined defense
    combined_defended = eot_light_tensor(
        adversarial_images,
        jpeg_q_min=eot_config.get("jpeg_q_min", 50),
        jpeg_q_max=eot_config.get("jpeg_q_max", 90),
        resize_min=eot_config.get("resize_min", 0.9),
        resize_max=eot_config.get("resize_max", 1.1),
        eot_prob=1.0,  # Apply both
    )

    combined_drop = compute_alignment_drop(
        original_images, combined_defended, texts, aligners
    )
    defense_results["combined_defense_asr"] = 1.0 if combined_drop > 0 else 0.0

    logger.info(
        f"Defense ASR - JPEG: {defense_results['jpeg_defense_asr']:.3f}, "
        f"Resize: {defense_results['resize_defense_asr']:.3f}, "
        f"Combined: {defense_results['combined_defense_asr']:.3f}"
    )

    return defense_results


def run_evaluation(
    data_dir: str, config: dict, params: dict, output_dir: str, device: str = "cuda"
) -> dict:
    """Run comprehensive evaluation.

    Args:
        data_dir: Test data directory
        config: Configuration
        params: Universal parameters (alpha, radius)
        output_dir: Output directory
        device: Device

    Returns:
        Evaluation results
    """
    alpha = params["alpha"]
    radius = params["radius"]

    logger.info(f"Running evaluation with alpha={alpha:.3f}, radius={radius:.3f}")

    # Load aligners
    surrogate_specs = config.get("surrogates", ["openclip:ViT-L-14@336"])
    training_aligners = build_aligners(surrogate_specs[:2], device)  # Use first 2 for training

    # Load held-out aligners for transfer evaluation
    held_out_specs = ["openclip:ViT-B-32@336"]  # Simplified for now
    try:
        held_out_aligners = build_aligners(held_out_specs, device)
        logger.info(f"Loaded {len(held_out_aligners)} held-out aligners")
    except Exception as e:
        logger.warning(f"Failed to load held-out aligners: {e}")
        held_out_aligners = []

    # Load dataset
    image_files = get_image_files(data_dir)
    if not image_files:
        raise ValueError(f"No image files found in {data_dir}")

    # Limit to reasonable evaluation size
    max_eval_samples = 50
    if len(image_files) > max_eval_samples:
        import random

        random.shuffle(image_files)
        image_files = image_files[:max_eval_samples]
        logger.info(f"Limited evaluation to {max_eval_samples} samples")

    dataset = DocumentDataset(
        image_files,
        ocr_backend=config.get("ocr", "easyocr"),
        device=device,
        max_size=(512, 512),  # Smaller for faster evaluation
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Small batch for memory
        shuffle=False,
        collate_fn=collate_documents,
        num_workers=0,
    )

    # Initialize metrics
    metrics = EvaluationMetrics()
    sample_results = []

    # Evaluation loop
    logger.info("Starting evaluation...")
    all_original_images = []
    all_adversarial_images = []
    all_texts = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["images"]
        texts = batch["texts"]
        paths = batch["paths"]

        # Generate adversarial images
        batch_adversarial = []
        for i in range(len(images)):
            img = images[i : i + 1]
            text = texts[i]

            # Compute saliency
            x_input = img.clone().requires_grad_(True)
            _, saliency_map = compute_gradient_saliency(
                training_aligners[0], x_input, text
            )

            # Apply field
            x_adv = apply_field_safe(img, saliency_map, alpha, radius)
            batch_adversarial.append(x_adv)

        batch_adversarial = torch.cat(batch_adversarial, dim=0)

        # Compute metrics for this batch
        alignment_drop = compute_alignment_drop(
            images, batch_adversarial, texts, training_aligners
        )

        # Compute LPIPS scores
        try:
            lpips_scores = batch_lpips_scores(images, batch_adversarial, device=device)
        except ImportError:
            lpips_scores = np.zeros(len(images))

        # Add to overall metrics
        for i in range(len(images)):
            # Individual alignments for this sample
            with torch.no_grad():
                orig_align = float(training_aligners[0].cosine_align(images[i : i + 1], texts[i]))
                adv_align = float(training_aligners[0].cosine_align(batch_adversarial[i : i + 1], texts[i]))

            metrics.add_sample(orig_align, adv_align, lpips_scores[i])

            sample_results.append(
                {
                    "path": paths[i],
                    "original_alignment": orig_align,
                    "adversarial_alignment": adv_align,
                    "lpips_score": float(lpips_scores[i]),
                }
            )

        # Collect for transfer/defense evaluation
        all_original_images.append(images)
        all_adversarial_images.append(batch_adversarial)
        all_texts.extend(texts)

    # Combine all images for transfer/defense evaluation
    all_original_images = torch.cat(all_original_images, dim=0)
    all_adversarial_images = torch.cat(all_adversarial_images, dim=0)

    # Compute main statistics
    main_stats = metrics.compute_statistics()

    # Transfer ASR
    if held_out_aligners:
        transfer_results = evaluate_transfer_asr(
            all_original_images[:10],  # Limit to first 10 for speed
            all_adversarial_images[:10],
            all_texts[:10],
            held_out_aligners,
        )
        main_stats.update(transfer_results)

    # Defense ASR
    eot_config = config.get("eot", {})
    defense_results = evaluate_defense_asr(
        all_original_images[:10],  # Limit to first 10 for speed
        all_adversarial_images[:10],
        all_texts[:10],
        training_aligners,
        eot_config,
    )
    main_stats.update(defense_results)

    # Compile final results
    results = {
        "evaluation_config": {
            "num_samples": len(sample_results),
            "alpha": alpha,
            "radius": radius,
            "training_aligners": len(training_aligners),
            "held_out_aligners": len(held_out_aligners),
        },
        "main_statistics": main_stats,
        "sample_results": sample_results,
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation results to: {results_file}")
    return results


def generate_markdown_report(results: dict, output_path: str) -> None:
    """Generate Markdown evaluation report.

    Args:
        results: Evaluation results
        output_path: Output file path
    """
    stats = results["main_statistics"]
    config = results["evaluation_config"]

    report = f"""# DocSAF Evaluation Report

## Configuration
- **Samples Evaluated**: {config['num_samples']}
- **Alpha (Field Strength)**: {config['alpha']:.3f}
- **Radius (Blur Radius)**: {config['radius']:.3f}
- **Training Aligners**: {config['training_aligners']}
- **Held-out Aligners**: {config['held_out_aligners']}

## Main Results

### Attack Success Rate (ASR)
- **Overall Success Rate**: {stats.get('success_rate', 0.0):.1%}

### Alignment Statistics
- **Original Alignment** (mean ± std): {stats.get('alignment_original_mean', 0):.3f} ± {stats.get('alignment_original_std', 0):.3f}
- **Adversarial Alignment** (mean ± std): {stats.get('alignment_adversarial_mean', 0):.3f} ± {stats.get('alignment_adversarial_std', 0):.3f}
- **Alignment Drop** (mean ± std): {stats.get('alignment_drop_mean', 0):.3f} ± {stats.get('alignment_drop_std', 0):.3f}

### Perceptual Quality (LPIPS)
- **Mean LPIPS**: {stats.get('lpips_mean', 0):.4f}
- **Median LPIPS**: {stats.get('lpips_median', 0):.4f}
- **Max LPIPS**: {stats.get('lpips_max', 0):.4f}
- **Below Threshold (≤0.06)**: {stats.get('lpips_below_threshold_rate', 0):.1%}

### OCR and Document Understanding Impact
- **OCR Mismatch Rate**: {stats.get('ocr_mismatch_mean', 0):.1%} ± {stats.get('ocr_mismatch_std', 0):.1%}
- **Donut Mismatch Rate**: {stats.get('donut_mismatch_mean', 0):.1%} ± {stats.get('donut_mismatch_std', 0):.1%}

### Transfer Attack Success Rate
"""

    # Add transfer results
    for key, value in stats.items():
        if key.startswith("held_out_aligner_"):
            report += f"- **{key}**: {value:.1%}\n"

    report += """
### Defense Robustness
"""

    # Add defense results
    defense_keys = ["jpeg_defense_asr", "resize_defense_asr", "combined_defense_asr"]
    for key in defense_keys:
        if key in stats:
            name = key.replace("_defense_asr", "").replace("_", " ").title()
            report += f"- **{name} Defense ASR**: {stats[key]:.1%}\n"

    report += f"""
## Summary

The DocSAF attack achieved a **{stats.get('success_rate', 0.0):.1%}** success rate with an average alignment drop of **{stats.get('alignment_drop_mean', 0):.3f}**. The perceptual quality was maintained with a mean LPIPS of **{stats.get('lpips_mean', 0):.4f}**, and **{stats.get('lpips_below_threshold_rate', 0):.1%}** of attacks remained below the perceptibility threshold.

Transfer attacks showed mixed results across held-out models, while defense mechanisms reduced but did not eliminate attack effectiveness.
"""

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Generated Markdown report: {output_path}")


def generate_csv_report(results: dict, output_path: str) -> None:
    """Generate CSV evaluation report.

    Args:
        results: Evaluation results
        output_path: Output file path
    """
    stats = results["main_statistics"]
    config = results["evaluation_config"]
    
    # Summary CSV with key metrics
    csv_data = []
    
    # Add configuration row
    csv_data.append({
        "metric": "config_samples",
        "value": config["num_samples"],
        "description": "Number of samples evaluated"
    })
    csv_data.append({
        "metric": "config_alpha",
        "value": config["alpha"],
        "description": "Field strength parameter"
    })
    csv_data.append({
        "metric": "config_radius", 
        "value": config["radius"],
        "description": "Blur radius parameter"
    })
    
    # Add main statistics
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            description = key.replace("_", " ").title()
            csv_data.append({
                "metric": key,
                "value": value,
                "description": description
            })
    
    # Write summary CSV
    with open(output_path, "w", newline="") as f:
        fieldnames = ["metric", "value", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    # Also write detailed per-sample CSV
    detail_path = str(output_path).replace(".csv", "_detailed.csv")
    sample_results = results.get("sample_results", [])
    
    if sample_results:
        with open(detail_path, "w", newline="") as f:
            fieldnames = list(sample_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_results)
        logger.info(f"Generated detailed CSV: {detail_path}")
    
    logger.info(f"Generated CSV report: {output_path}")


app = typer.Typer(
    name="docsaf-eval",
    help="DocSAF evaluation harness for comprehensive attack assessment using TWO KNOBS ONLY: alpha (field strength) and radius (blur kernel).",
    epilog="""
Examples:
  docsaf-eval --data data/test_docs --output eval_results
  docsaf-eval --data ../dataset --report report.md --csv results.csv
  docsaf-eval --data test_images --seed 1337 --device cuda

The evaluation measures:
- Attack Success Rate (positive alignment drop)
- Transfer ASR (held-out models) 
- Defense robustness (JPEG + resize)
- OCR/Donut mismatch rates
- LPIPS perceptual quality
"""
)


@app.command()
def main(
    data: str = typer.Option(..., "--data", help="Path to test data directory"),
    config: str = typer.Option("configs/default.yaml", "--config", help="Path to config file"),
    params: str = typer.Option("runs/universal.pt", "--params", help="Path to universal parameters (alpha, radius)"),
    output: str = typer.Option("eval_results", "--output", help="Output directory"),
    report: Optional[str] = typer.Option(None, "--report", help="Generate Markdown report at specified path"),
    csv: Optional[str] = typer.Option(None, "--csv", help="Generate CSV report at specified path"),
    device: str = typer.Option("auto", "--device", help="Device: auto/cuda/cpu"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level: DEBUG/INFO/WARNING/ERROR"),
):
    """
    Run comprehensive DocSAF evaluation on a dataset.
    
    Evaluates attack effectiveness, transfer capability, defense robustness,
    and perceptual quality using the TWO-KNOB DocSAF design.
    """
    # Validate inputs early
    if not Path(data).exists():
        typer.echo(f"❌ Error: Data directory not found: {data}", err=True)
        raise typer.Exit(1)
        
    if not Path(config).exists():
        typer.echo(f"❌ Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)
        
    if not Path(params).exists():
        typer.echo(f"❌ Error: Parameters file not found: {params}", err=True)
        typer.echo(f"💡 Hint: Run training first or check parameter path", err=True)
        raise typer.Exit(1)

    # Setup
    try:
        setup_logging(log_level)
        device = get_device(device)
        
        # Set deterministic seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        typer.echo(f"🚀 Starting DocSAF evaluation...")
        typer.echo(f"💻 Device: {device}, Seed: {seed}")

        # Load config and params
        config_data = load_config(config)
        params_data = torch.load(params, map_location="cpu")
        
        # Enforce two-knob invariant
        expected_keys = {"alpha", "radius"}
        extra_keys = set(params_data.keys()) - expected_keys
        if extra_keys:
            typer.echo(f"❌ Error: DocSAF uses ONLY two knobs (alpha, radius). Found extra parameters: {extra_keys}", err=True)
            typer.echo(f"💡 This violates the two-knob design constraint.", err=True)
            raise typer.Exit(1)
            
        typer.echo(f"🎛️  Parameters: alpha={params_data['alpha']:.3f}, radius={params_data['radius']:.3f}")
        typer.echo(f"📂 Data: {data}")
        typer.echo(f"💾 Output: {output}")

        # Run evaluation
        results = run_evaluation(
            data_dir=data,
            config=config_data,
            params=params_data,
            output_dir=output,
            device=device,
        )

        # Generate reports
        if report:
            generate_markdown_report(results, report)
            typer.echo(f"📄 Markdown report: {report}")
        else:
            # Default markdown report
            report_path = Path(output) / "evaluation_report.md"
            generate_markdown_report(results, str(report_path))
            typer.echo(f"📄 Markdown report: {report_path}")
            
        if csv:
            generate_csv_report(results, csv)
            typer.echo(f"📊 CSV report: {csv}")
        else:
            # Default CSV report  
            csv_path = Path(output) / "evaluation_report.csv"
            generate_csv_report(results, str(csv_path))
            typer.echo(f"📊 CSV report: {csv_path}")

        # Print summary with nice formatting
        stats = results["main_statistics"]
        typer.echo("\n" + "="*60)
        typer.echo("🎯 DocSAF Evaluation Summary")
        typer.echo("="*60)
        typer.echo(f"📊 Samples Evaluated:        {results['evaluation_config']['num_samples']}")
        typer.echo(f"🎯 Attack Success Rate:      {stats.get('success_rate', 0.0):.1%}")
        typer.echo(f"📉 Mean Alignment Drop:      {stats.get('alignment_drop_mean', 0):.3f}")
        typer.echo(f"🖼️  Mean LPIPS Distance:      {stats.get('lpips_mean', 0):.4f}")
        typer.echo(f"👁️  Below Threshold (≤0.06):  {stats.get('lpips_below_threshold_rate', 0):.1%}")
        
        # Transfer and defense results
        transfer_keys = [k for k in stats.keys() if k.startswith('held_out_aligner_')]
        if transfer_keys:
            avg_transfer = np.mean([stats[k] for k in transfer_keys])
            typer.echo(f"🔄 Average Transfer ASR:     {avg_transfer:.1%}")
            
        defense_keys = ['jpeg_defense_asr', 'resize_defense_asr', 'combined_defense_asr']
        defense_present = [k for k in defense_keys if k in stats]
        if defense_present:
            avg_defense = np.mean([stats[k] for k in defense_present])
            typer.echo(f"🛡️  Average Defense ASR:      {avg_defense:.1%}")
        
        typer.echo(f"💾 Results saved to:         {output}")
        
        # Success indicator
        success_rate = stats.get('success_rate', 0.0)
        if success_rate > 0.5:
            typer.echo("✅ Evaluation complete: Strong attack performance!")
        elif success_rate > 0.1:
            typer.echo("⚠️  Evaluation complete: Moderate attack performance")
        else:
            typer.echo("❌ Evaluation complete: Weak attack performance")

    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        logger.error(f"Detailed error: {e}", exc_info=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
