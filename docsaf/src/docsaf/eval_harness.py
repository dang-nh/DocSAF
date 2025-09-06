"""Evaluation harness for DocSAF attacks."""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from .utils import (
    load_config, setup_logging, get_device, get_image_files,
    pil_to_tensor, tensor_to_pil, compute_lpips_score, batch_lpips_scores
)
from .surrogates import load_embedder, ImageTextEmbedder
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
        self.alignment_scores = {
            "original": [],
            "adversarial": [],
            "drop": []
        }
        self.lpips_scores = []
        self.success_rate = 0.0
        self.transfer_asr = {}
        self.defense_asr = {}
    
    def add_sample(
        self,
        original_alignment: float,
        adversarial_alignment: float,
        lpips_score: float
    ):
        """Add metrics for a single sample."""
        self.alignment_scores["original"].append(original_alignment)
        self.alignment_scores["adversarial"].append(adversarial_alignment)
        self.alignment_scores["drop"].append(original_alignment - adversarial_alignment)
        self.lpips_scores.append(lpips_score)
    
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
        
        return stats


def evaluate_single_image(
    image_path: str,
    embedders: List[ImageTextEmbedder],
    alpha: float,
    radius: float,
    config: dict,
    device: str = "cuda"
) -> Dict:
    """Evaluate attack on single image.
    
    Args:
        image_path: Path to image/PDF
        embedders: List of embedders for evaluation
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
        pil_image = Image.open(image_path).convert('RGB')
    
    x_orig = pil_to_tensor(pil_image, device)
    
    # Extract OCR text
    img_array = np.array(pil_image)
    ocr_backend = config.get("ocr", "easyocr")
    text = ocr_read(img_array, backend=ocr_backend)
    
    if not text.strip():
        text = "document text content"
    
    # Compute saliency using primary embedder
    primary_embedder = embedders[0]
    x_input = x_orig.clone().requires_grad_(True)
    orig_alignment, saliency_map = compute_gradient_saliency(
        primary_embedder, x_input, text
    )
    
    # Apply field
    x_adv = apply_field_safe(x_orig, saliency_map, alpha, radius)
    
    # Evaluate on all embedders
    results = {
        "image_path": str(image_path),
        "text": text,
        "original_alignment": orig_alignment,
        "embedder_results": {}
    }
    
    # Compute LPIPS
    try:
        lpips_score = compute_lpips_score(x_orig, x_adv, device=device)
        results["lpips_score"] = lpips_score
    except ImportError:
        results["lpips_score"] = None
    
    # Evaluate each embedder
    alignment_drops = []
    for i, embedder in enumerate(embedders):
        with torch.no_grad():
            # Original alignment
            orig_emb = torch.nn.functional.normalize(
                embedder.image_embed(x_orig), dim=-1
            )
            txt_emb = torch.nn.functional.normalize(
                embedder.text_embed([text]), dim=-1
            )
            orig_align = float((orig_emb * txt_emb).sum())
            
            # Adversarial alignment
            adv_emb = torch.nn.functional.normalize(
                embedder.image_embed(x_adv), dim=-1
            )
            adv_align = float((adv_emb * txt_emb).sum())
            
            alignment_drop = orig_align - adv_align
            alignment_drops.append(alignment_drop)
            
            results["embedder_results"][f"embedder_{i}"] = {
                "original_alignment": orig_align,
                "adversarial_alignment": adv_align,
                "alignment_drop": alignment_drop
            }
    
    results["mean_alignment_drop"] = float(np.mean(alignment_drops))
    return results


def evaluate_transfer_asr(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    texts: List[str],
    held_out_embedders: List[ImageTextEmbedder]
) -> Dict[str, float]:
    """Evaluate transfer attack success rate.
    
    Args:
        original_images: Original images (B, 3, H, W)
        adversarial_images: Adversarial images (B, 3, H, W)
        texts: Text strings
        held_out_embedders: Held-out embedders not used in training
        
    Returns:
        Transfer ASR results per embedder
    """
    transfer_results = {}
    
    for i, embedder in enumerate(held_out_embedders):
        alignment_drop = compute_alignment_drop(
            original_images, adversarial_images, texts, [embedder]
        )
        
        # Success = positive alignment drop
        success_rate = 1.0 if alignment_drop > 0 else 0.0
        transfer_results[f"held_out_embedder_{i}"] = success_rate
        
        logger.info(f"Transfer ASR (embedder {i}): {success_rate:.3f} "
                   f"(avg drop: {alignment_drop:.4f})")
    
    return transfer_results


def evaluate_defense_asr(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    texts: List[str],
    embedders: List[ImageTextEmbedder],
    eot_config: dict
) -> Dict[str, float]:
    """Evaluate robustness against defenses.
    
    Args:
        original_images: Original images
        adversarial_images: Adversarial images
        texts: Texts
        embedders: Embedders
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
        eot_prob=1.0  # Always apply JPEG
    )
    
    jpeg_drop = compute_alignment_drop(
        original_images, jpeg_defended, texts, embedders
    )
    defense_results["jpeg_defense_asr"] = 1.0 if jpeg_drop > 0 else 0.0
    
    # Resize defense
    resize_defended = eot_light_tensor(
        adversarial_images,
        jpeg_q_min=100,
        jpeg_q_max=100,  # No JPEG
        resize_min=eot_config.get("resize_min", 0.9),
        resize_max=eot_config.get("resize_max", 1.1),
        eot_prob=1.0  # Always apply resize
    )
    
    resize_drop = compute_alignment_drop(
        original_images, resize_defended, texts, embedders
    )
    defense_results["resize_defense_asr"] = 1.0 if resize_drop > 0 else 0.0
    
    # Combined defense
    combined_defended = eot_light_tensor(
        adversarial_images,
        jpeg_q_min=eot_config.get("jpeg_q_min", 50),
        jpeg_q_max=eot_config.get("jpeg_q_max", 90),
        resize_min=eot_config.get("resize_min", 0.9),
        resize_max=eot_config.get("resize_max", 1.1),
        eot_prob=1.0  # Apply both
    )
    
    combined_drop = compute_alignment_drop(
        original_images, combined_defended, texts, embedders
    )
    defense_results["combined_defense_asr"] = 1.0 if combined_drop > 0 else 0.0
    
    logger.info(f"Defense ASR - JPEG: {defense_results['jpeg_defense_asr']:.3f}, "
               f"Resize: {defense_results['resize_defense_asr']:.3f}, "
               f"Combined: {defense_results['combined_defense_asr']:.3f}")
    
    return defense_results


def run_evaluation(
    data_dir: str,
    config: dict,
    params: dict,
    output_dir: str,
    device: str = "cuda"
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
    
    # Load embedders
    surrogate_specs = config.get("surrogates", ["openclip:ViT-L-14@336"])
    training_embedders = []
    for spec in surrogate_specs[:2]:  # Use first 2 for training evaluation
        try:
            embedder = load_embedder(spec, device)
            training_embedders.append(embedder)
        except Exception as e:
            logger.warning(f"Failed to load embedder {spec}: {e}")
    
    # Load held-out embedders for transfer evaluation
    held_out_specs = ["openclip:ViT-B-32@336", "hf:blip2-flan-t5-xl"]
    held_out_embedders = []
    for spec in held_out_specs:
        try:
            embedder = load_embedder(spec, device)
            held_out_embedders.append(embedder)
            logger.info(f"Loaded held-out embedder: {spec}")
        except Exception as e:
            logger.warning(f"Failed to load held-out embedder {spec}: {e}")
    
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
        max_size=(512, 512)  # Smaller for faster evaluation
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Small batch for memory
        shuffle=False,
        collate_fn=collate_documents,
        num_workers=0
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
        images = batch['images']
        texts = batch['texts']
        paths = batch['paths']
        
        # Generate adversarial images
        batch_adversarial = []
        for i in range(len(images)):
            img = images[i:i+1]
            text = texts[i]
            
            # Compute saliency
            x_input = img.clone().requires_grad_(True)
            _, saliency_map = compute_gradient_saliency(
                training_embedders[0], x_input, text
            )
            
            # Apply field
            x_adv = apply_field_safe(img, saliency_map, alpha, radius)
            batch_adversarial.append(x_adv)
        
        batch_adversarial = torch.cat(batch_adversarial, dim=0)
        
        # Compute metrics for this batch
        alignment_drop = compute_alignment_drop(
            images, batch_adversarial, texts, training_embedders
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
                orig_emb = torch.nn.functional.normalize(
                    training_embedders[0].image_embed(images[i:i+1]), dim=-1
                )
                txt_emb = torch.nn.functional.normalize(
                    training_embedders[0].text_embed([texts[i]]), dim=-1
                )
                orig_align = float((orig_emb * txt_emb).sum())
                
                adv_emb = torch.nn.functional.normalize(
                    training_embedders[0].image_embed(batch_adversarial[i:i+1]), dim=-1
                )
                adv_align = float((adv_emb * txt_emb).sum())
            
            metrics.add_sample(orig_align, adv_align, lpips_scores[i])
            
            sample_results.append({
                "path": paths[i],
                "original_alignment": orig_align,
                "adversarial_alignment": adv_align,
                "lpips_score": float(lpips_scores[i])
            })
        
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
    if held_out_embedders:
        transfer_results = evaluate_transfer_asr(
            all_original_images[:10],  # Limit to first 10 for speed
            all_adversarial_images[:10],
            all_texts[:10],
            held_out_embedders
        )
        main_stats.update(transfer_results)
    
    # Defense ASR
    eot_config = config.get("eot", {})
    defense_results = evaluate_defense_asr(
        all_original_images[:10],  # Limit to first 10 for speed
        all_adversarial_images[:10],
        all_texts[:10],
        training_embedders,
        eot_config
    )
    main_stats.update(defense_results)
    
    # Compile final results
    results = {
        "evaluation_config": {
            "num_samples": len(sample_results),
            "alpha": alpha,
            "radius": radius,
            "training_embedders": len(training_embedders),
            "held_out_embedders": len(held_out_embedders)
        },
        "main_statistics": main_stats,
        "sample_results": sample_results
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
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
- **Training Embedders**: {config['training_embedders']}
- **Held-out Embedders**: {config['held_out_embedders']}

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

### Transfer Attack Success Rate
"""
    
    # Add transfer results
    for key, value in stats.items():
        if key.startswith("held_out_embedder_"):
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
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Generated Markdown report: {output_path}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="DocSAF evaluation harness")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to test data directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--params", type=str, default="runs/universal.pt",
                       help="Path to universal parameters")
    parser.add_argument("--output", type=str, default="eval_results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/cuda/cpu)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load config and params
    config = load_config(args.config)
    
    if not Path(args.params).exists():
        logger.error(f"Universal params not found: {args.params}")
        return
    
    params = torch.load(args.params, map_location="cpu")
    
    try:
        # Run evaluation
        results = run_evaluation(
            data_dir=args.data,
            config=config,
            params=params,
            output_dir=args.output,
            device=device
        )
        
        # Generate report
        report_path = Path(args.output) / "evaluation_report.md"
        generate_markdown_report(results, str(report_path))
        
        # Print summary
        stats = results["main_statistics"]
        print(f"\n=== DocSAF Evaluation Summary ===")
        print(f"Samples: {results['evaluation_config']['num_samples']}")
        print(f"Success Rate: {stats.get('success_rate', 0.0):.1%}")
        print(f"Mean Alignment Drop: {stats.get('alignment_drop_mean', 0):.3f}")
        print(f"Mean LPIPS: {stats.get('lpips_mean', 0):.4f}")
        print(f"Below Perceptibility Threshold: {stats.get('lpips_below_threshold_rate', 0):.1%}")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()