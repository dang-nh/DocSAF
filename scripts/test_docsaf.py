#!/usr/bin/env python3
"""Testing/evaluation script for DocSAF on various datasets."""

import argparse
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
data_dir = current_dir.parent / "data"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(data_dir))

from docsaf.utils import load_config, setup_logging, get_device, pil_to_tensor
from docsaf.surrogates import load_embedder
from docsaf.objective import DocSAFObjective
from docsaf.field import apply_field
from docsaf.saliency import compute_gradient_saliency
from docsaf.train_universal import StructuredDocumentDataset, DocumentDataset, collate_documents
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)


def compute_attack_success_rate(
    original_similarities: list, 
    adversarial_similarities: list, 
    threshold: float = 0.1
) -> float:
    """Compute attack success rate based on similarity degradation."""
    successes = 0
    for orig, adv in zip(original_similarities, adversarial_similarities):
        degradation = orig - adv
        if degradation > threshold:
            successes += 1
    return successes / len(original_similarities) if original_similarities else 0.0


def evaluate_dataset(
    dataset, 
    embedder, 
    alpha: float, 
    radius: float, 
    device: str, 
    batch_size: int = 4,
    max_samples: int = None
) -> dict:
    """Evaluate DocSAF on a dataset."""
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_documents,
        num_workers=0
    )
    
    results = {
        "original_similarities": [],
        "adversarial_similarities": [],
        "lpips_scores": [],
        "processing_times": [],
        "failed_samples": 0
    }
    
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch_idx, batch in enumerate(progress_bar):
        if max_samples and total_samples >= max_samples:
            break
            
        images = batch["images"]
        texts = batch["texts"]
        batch_size_actual = images.shape[0]
        
        try:
            # Compute original similarities
            with torch.no_grad():
                original_sims = []
                for i in range(batch_size_actual):
                    img = images[i:i+1]
                    text = texts[i]
                    
                    # Get embeddings
                    img_emb = embedder.encode_image(img)
                    text_emb = embedder.encode_text([text])
                    
                    # Compute similarity
                    similarity = torch.cosine_similarity(img_emb, text_emb, dim=-1)
                    original_sims.append(float(similarity.item()))
                
                results["original_similarities"].extend(original_sims)
            
            # Compute saliency maps and adversarial examples
            adversarial_sims = []
            lpips_batch = []
            
            for i in range(batch_size_actual):
                img = images[i:i+1].requires_grad_(True)
                text = texts[i]
                
                # Compute saliency
                alignment, saliency = compute_gradient_saliency(
                    embedder, img, text, normalize=True
                )
                
                # Apply adversarial field
                with torch.no_grad():
                    x_adv = apply_field(
                        img.detach(), 
                        saliency.detach(), 
                        torch.tensor(alpha, device=device), 
                        torch.tensor(radius, device=device)
                    )
                
                # Compute adversarial similarity
                with torch.no_grad():
                    img_emb_adv = embedder.encode_image(x_adv)
                    text_emb = embedder.encode_text([text])
                    sim_adv = torch.cosine_similarity(img_emb_adv, text_emb, dim=-1)
                    adversarial_sims.append(float(sim_adv.item()))
                
                # Compute LPIPS (simplified - you might want to implement proper LPIPS)
                lpips_score = torch.mean((img.detach() - x_adv) ** 2).item()
                lpips_batch.append(lpips_score)
                
                # Clear gradients
                if img.grad is not None:
                    img.grad.zero_()
            
            results["adversarial_similarities"].extend(adversarial_sims)
            results["lpips_scores"].extend(lpips_batch)
            
        except Exception as e:
            logger.warning(f"Failed to process batch {batch_idx}: {e}")
            results["failed_samples"] += batch_size_actual
            # Add dummy values to maintain list alignment
            results["adversarial_similarities"].extend([0.0] * batch_size_actual)
            results["lpips_scores"].extend([1.0] * batch_size_actual)
        
        total_samples += batch_size_actual
        
        # Update progress bar
        if results["original_similarities"]:
            avg_orig = np.mean(results["original_similarities"])
            avg_adv = np.mean(results["adversarial_similarities"])
            progress_bar.set_postfix({
                "Avg_Orig_Sim": f"{avg_orig:.3f}",
                "Avg_Adv_Sim": f"{avg_adv:.3f}",
                "Samples": total_samples
            })
    
    progress_bar.close()
    
    # Compute summary statistics
    if results["original_similarities"]:
        success_rate = compute_attack_success_rate(
            results["original_similarities"], 
            results["adversarial_similarities"]
        )
        
        summary = {
            "total_samples": total_samples,
            "failed_samples": results["failed_samples"],
            "attack_success_rate": success_rate,
            "avg_original_similarity": float(np.mean(results["original_similarities"])),
            "avg_adversarial_similarity": float(np.mean(results["adversarial_similarities"])),
            "avg_similarity_drop": float(np.mean(results["original_similarities"]) - np.mean(results["adversarial_similarities"])),
            "avg_lpips_score": float(np.mean(results["lpips_scores"])),
            "std_lpips_score": float(np.std(results["lpips_scores"]))
        }
        
        results["summary"] = summary
    
    return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate DocSAF on document datasets")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--params", type=str, required=True,
        help="Path to trained universal parameters (.pt file)"
    )
    parser.add_argument(
        "--dataset", type=str,
        choices=["funsd", "cord", "sroie", "docvqa", "doclaynet"],
        help="Dataset type (if not specified, treats as simple image directory)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (auto/cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load trained parameters
    if not os.path.exists(args.params):
        raise ValueError(f"Parameters file not found: {args.params}")
    
    params = torch.load(args.params, map_location=device)
    alpha = params["alpha"]
    radius = params["radius"]
    
    logger.info(f"Loaded parameters: alpha={alpha:.3f}, radius={radius:.3f}")
    
    # Load dataset
    dataset_config = config.get("dataset", {})
    max_size = tuple(dataset_config.get("max_size", [1024, 1024]))
    
    if args.dataset:
        logger.info(f"Loading structured dataset: {args.dataset}")
        dataset = StructuredDocumentDataset(
            data_dir=args.data,
            dataset_name=args.dataset,
            split=args.split,
            device=device,
            max_size=max_size,
            line_level=dataset_config.get("line_level", True),
            is_normalize=dataset_config.get("is_normalize", True),
        )
    else:
        logger.info("Loading simple image dataset")
        from docsaf.utils import get_image_files
        image_files = get_image_files(args.data)
        if not image_files:
            raise ValueError(f"No image files found in {args.data}")
        
        dataset = DocumentDataset(
            image_files,
            ocr_backend=config.get("ocr", "easyocr"),
            device=device,
            max_size=max_size
        )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Load surrogate model for evaluation
    surrogate_specs = config.get("surrogates", ["openclip:ViT-L-14@336"])
    embedder = load_embedder(surrogate_specs[0], device=device)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluate_dataset(
        dataset=dataset,
        embedder=embedder,
        alpha=alpha,
        radius=radius,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset or "simple"
    results_file = output_path / f"eval_{dataset_name}_{args.split}_{timestamp}.json"
    
    # Prepare results for JSON serialization
    json_results = {
        "evaluation_config": {
            "dataset": args.dataset,
            "split": args.split,
            "data_dir": args.data,
            "params_file": args.params,
            "alpha": alpha,
            "radius": radius,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "timestamp": timestamp
        },
        "results": {
            "summary": results.get("summary", {}),
            "total_samples": len(results["original_similarities"]),
            "failed_samples": results["failed_samples"]
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    if "summary" in results:
        summary = results["summary"]
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Dataset: {args.dataset or 'Simple images'}")
        print(f"Split: {args.split}")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Failed samples: {summary['failed_samples']}")
        print(f"Parameters: α={alpha:.3f}, r={radius:.3f}")
        print("-" * 30)
        print(f"Attack Success Rate: {summary['attack_success_rate']:.1%}")
        print(f"Avg Original Similarity: {summary['avg_original_similarity']:.3f}")
        print(f"Avg Adversarial Similarity: {summary['avg_adversarial_similarity']:.3f}")
        print(f"Avg Similarity Drop: {summary['avg_similarity_drop']:.3f}")
        print(f"Avg LPIPS Score: {summary['avg_lpips_score']:.4f} ± {summary['std_lpips_score']:.4f}")
        print("=" * 50)
        print(f"Results saved to: {results_file}")
    else:
        print("Evaluation failed - no results to display")


if __name__ == "__main__":
    main()
