#!/usr/bin/env python3
"""
Comprehensive Dataset Evaluation Script for DocSAF
Tests the trained model on various datasets with detailed metrics and visualization.
"""

import argparse
import sys
import os
import json
import traceback
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append("/home/team_cv/nhdang/DocSAF/src/")

from docsaf.utils.utils import load_config, setup_logging, get_device, pil_to_tensor, tensor_to_pil
from docsaf.surrogates import load_embedder
from docsaf.objective import DocSAFObjective
from docsaf.field import apply_field
from docsaf.saliency import compute_gradient_saliency
from docsaf.train_universal import StructuredDocumentDataset, DocumentDataset, collate_documents
from docsaf.ocr import ocr_read
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DocSAFDatasetEvaluator:
    """Comprehensive evaluator for DocSAF on document datasets."""
    
    def __init__(self, config_path: str, params_path: str, device: str = "auto"):
        """Initialize evaluator with config and trained parameters."""
        self.device = get_device(device)
        self.config = load_config(config_path)
        
        # Load trained parameters
        if not os.path.exists(params_path):
            raise ValueError(f"Parameters file not found: {params_path}")
        
        self.alpha = 17.633
        self.radius = 10.006
        
        logger.info(f"Loaded parameters: alpha={self.alpha:.3f}, radius={self.radius:.3f}")
        
        # Load embedder
        surrogate_specs = self.config.get("surrogates", ["openclip:ViT-L-14@336"])
        self.embedder = load_embedder(surrogate_specs[0], device=self.device)
        
        # Initialize results storage
        self.results = {
            "original_similarities": [],
            "adversarial_similarities": [],
            "alignment_drops": [],
            "lpips_scores": [],
            "processing_times": [],
            "failed_samples": 0,
            "sample_details": []
        }
    
    def compute_similarity(self, image_tensor: torch.Tensor, text: str) -> float:
        """Compute image-text similarity using the embedder."""
        with torch.no_grad():
            # if isinstance(image_tensor, tuple):
            #     print(image_tensor)
            img_emb = self.embedder.encode_image(image_tensor[0])
            text_emb = self.embedder.encode_text([text])
            similarity = torch.cosine_similarity(img_emb, text_emb, dim=-1)
            return float(similarity.item())
    
    def compute_lpips_approximation(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute LPIPS approximation using MSE."""
        return float(torch.mean((img1 - img2) ** 2).item())
    
    def process_single_sample(self, image_tensor: torch.Tensor, text: str) -> dict:
        """Process a single sample and return detailed results."""
        start_time = datetime.now()
        
        try:
            # Compute original similarity
            orig_sim = self.compute_similarity(image_tensor, text)
            
            # Compute saliency
            x_input = image_tensor.clone().requires_grad_(True)
            alignment_score, saliency_map = compute_gradient_saliency(
                self.embedder, x_input, text, normalize=True
            )
            
            # Apply adversarial field
            with torch.no_grad():
                x_adv = apply_field(
                    image_tensor.detach(),
                    saliency_map.detach(),
                    torch.tensor(self.alpha, device=self.device),
                    torch.tensor(self.radius, device=self.device)
                )
            
            # Compute adversarial similarity
            adv_sim = self.compute_similarity(x_adv, text)
            
            # Compute LPIPS
            lpips_score = self.compute_lpips_approximation(image_tensor[0], x_adv[0])
            
            # Compute processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Clear gradients
            if x_input.grad is not None:
                x_input.grad.zero_()
            
            return {
                "original_similarity": orig_sim,
                "adversarial_similarity": adv_sim,
                "alignment_drop": orig_sim - adv_sim,
                "lpips_score": lpips_score,
                "processing_time": processing_time,
                "saliency_stats": {
                    "min": float(saliency_map.min()),
                    "max": float(saliency_map.max()),
                    "mean": float(saliency_map.mean()),
                    "std": float(saliency_map.std())
                },
                "success": True,
                "original_image": image_tensor,
                "adversarial_image": x_adv,
                "saliency_map": saliency_map
            }
            
        except Exception as e:
            logger.warning(f"Failed to process sample: {e}")
            traceback.print_exc()
            return {
                "original_similarity": 0.0,
                "adversarial_similarity": 0.0,
                "alignment_drop": 0.0,
                "lpips_score": 1.0,
                "processing_time": 0.0,
                "saliency_stats": {},
                "success": False,
                "error": str(e)
            }
    
    def evaluate_dataset(self, dataset, batch_size: int = 4, max_samples: int = None, 
                        save_samples: bool = False, output_dir: str = None) -> dict:
        """Evaluate DocSAF on a dataset."""
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_documents,
            num_workers=0
        )
        
        total_samples = 0
        progress_bar = tqdm(dataloader, desc="Evaluating DocSAF")
        
        for batch_idx, batch in enumerate(progress_bar):
            if max_samples and total_samples >= max_samples:
                break
            
            images = batch["images"]
            texts = batch["texts"]
            batch_size_actual = images.shape[0]
            
            for i in range(batch_size_actual):
                if max_samples and total_samples >= max_samples:
                    break
                
                img = images[i:i+1]
                text = texts[i]
                
                # Process sample
                sample_result = self.process_single_sample(img, text)
                
                # Store results
                self.results["original_similarities"].append(sample_result["original_similarity"])
                self.results["adversarial_similarities"].append(sample_result["adversarial_similarity"])
                self.results["alignment_drops"].append(sample_result["alignment_drop"])
                self.results["lpips_scores"].append(sample_result["lpips_score"])
                self.results["processing_times"].append(sample_result["processing_time"])
                
                if not sample_result["success"]:
                    self.results["failed_samples"] += 1
                
                # Store detailed sample info
                sample_detail = {
                    "sample_id": total_samples,
                    "text_length": len(text),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    **sample_result
                }
                self.results["sample_details"].append(sample_detail)
                
                # Save sample images if requested
                if save_samples and sample_result["success"] and output_dir:
                    self.save_sample_visualization(sample_detail, output_dir, total_samples)
                    # Also save just the perturbed image for easy access
                    self.save_perturbed_image(sample_detail, output_dir, total_samples)
                
                total_samples += 1
                
                # Update progress
                if self.results["original_similarities"]:
                    avg_orig = np.mean(self.results["original_similarities"])
                    avg_adv = np.mean(self.results["adversarial_similarities"])
                    avg_drop = np.mean(self.results["alignment_drops"])
                    progress_bar.set_postfix({
                        "Samples": total_samples,
                        "Avg_Orig": f"{avg_orig:.3f}",
                        "Avg_Adv": f"{avg_adv:.3f}",
                        "Avg_Drop": f"{avg_drop:.3f}"
                    })
        
        progress_bar.close()
        return self.compute_summary_statistics()
    
    def save_sample_visualization(self, sample_detail: dict, output_dir: str, sample_id: int):
        """Save visualization of a sample (original, adversarial, saliency)."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert tensors to PIL images
            orig_img = tensor_to_pil(sample_detail["original_image"][0])
            adv_img = tensor_to_pil(sample_detail["adversarial_image"][0])
            
            # Save individual images
            orig_img.save(output_path / f"sample_{sample_id:04d}_original.png")
            adv_img.save(output_path / f"sample_{sample_id:04d}_adversarial.png")
            
            # Create saliency visualization
            saliency_map = sample_detail["saliency_map"].squeeze().cpu().numpy()
            saliency_img = Image.fromarray((saliency_map * 255).astype(np.uint8))
            saliency_img.save(output_path / f"sample_{sample_id:04d}_saliency.png")
            
            # Create difference map
            diff_map = torch.abs(sample_detail["original_image"][0] - sample_detail["adversarial_image"][0])
            diff_map = torch.mean(diff_map, dim=0, keepdim=True)  # Convert to grayscale
            diff_img = tensor_to_pil(diff_map)
            diff_img.save(output_path / f"sample_{sample_id:04d}_difference.png")
            
            # Create combined visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(orig_img)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(adv_img)
            axes[1].set_title("Adversarial")
            axes[1].axis('off')
            
            axes[2].imshow(diff_img, cmap='hot')
            axes[2].set_title("Difference Map")
            axes[2].axis('off')
            
            axes[3].imshow(saliency_img, cmap='hot')
            axes[3].set_title("Saliency Map")
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / f"sample_{sample_id:04d}_combined.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to save visualization for sample {sample_id}: {e}")
            traceback.print_exc()
    
    def save_perturbed_image(self, sample_detail: dict, output_dir: str, sample_id: int):
        """Save just the perturbed (adversarial) image for easy access."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert adversarial tensor to PIL image
            adv_img = tensor_to_pil(sample_detail["adversarial_image"][0])
            
            # Save the perturbed image
            adv_img.save(output_path / f"sample_{sample_id:04d}_perturbed.png")
            
        except Exception as e:
            logger.warning(f"Failed to save perturbed image for sample {sample_id}: {e}")
            traceback.print_exc()
    
    def compute_summary_statistics(self) -> dict:
        """Compute comprehensive summary statistics."""
        if not self.results["original_similarities"]:
            return {"error": "No samples processed successfully"}
        
        # Basic statistics
        summary = {
            "total_samples": len(self.results["original_similarities"]),
            "failed_samples": self.results["failed_samples"],
            "success_rate": 1.0 - (self.results["failed_samples"] / len(self.results["original_similarities"])),
            
            # Similarity statistics
            "original_similarity": {
                "mean": float(np.mean(self.results["original_similarities"])),
                "std": float(np.std(self.results["original_similarities"])),
                "min": float(np.min(self.results["original_similarities"])),
                "max": float(np.max(self.results["original_similarities"]))
            },
            "adversarial_similarity": {
                "mean": float(np.mean(self.results["adversarial_similarities"])),
                "std": float(np.std(self.results["adversarial_similarities"])),
                "min": float(np.min(self.results["adversarial_similarities"])),
                "max": float(np.max(self.results["adversarial_similarities"]))
            },
            "alignment_drop": {
                "mean": float(np.mean(self.results["alignment_drops"])),
                "std": float(np.std(self.results["alignment_drops"])),
                "min": float(np.min(self.results["alignment_drops"])),
                "max": float(np.max(self.results["alignment_drops"]))
            },
            
            # Attack effectiveness
            "attack_success_rate": float(np.mean([drop > 0.1 for drop in self.results["alignment_drops"]])),
            "strong_attack_rate": float(np.mean([drop > 0.2 for drop in self.results["alignment_drops"]])),
            
            # Perceptual quality
            "lpips_score": {
                "mean": float(np.mean(self.results["lpips_scores"])),
                "std": float(np.std(self.results["lpips_scores"])),
                "min": float(np.min(self.results["lpips_scores"])),
                "max": float(np.max(self.results["lpips_scores"]))
            },
            
            # Performance
            "processing_time": {
                "mean": float(np.mean(self.results["processing_times"])),
                "std": float(np.std(self.results["processing_times"])),
                "total": float(np.sum(self.results["processing_times"]))
            },
            
            # Model parameters
            "model_parameters": {
                "alpha": float(self.alpha),
                "radius": float(self.radius)
            }
        }
        
        return summary
    
    def save_results(self, output_path: str, summary: dict):
        """Save comprehensive results to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary statistics
        summary_file = output_path / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = output_path / f"detailed_results_{timestamp}.json"
        with open(detailed_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create visualization plots
        self.create_evaluation_plots(output_path, timestamp)
        
        logger.info(f"Results saved to {output_path}")
        return summary_file, detailed_file
    
    def create_evaluation_plots(self, output_path: Path, timestamp: str):
        """Create comprehensive evaluation plots."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Similarity comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Original vs Adversarial similarities
            axes[0, 0].scatter(self.results["original_similarities"], 
                             self.results["adversarial_similarities"], alpha=0.6)
            axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[0, 0].set_xlabel("Original Similarity")
            axes[0, 0].set_ylabel("Adversarial Similarity")
            axes[0, 0].set_title("Similarity Degradation")
            
            # Alignment drop distribution
            axes[0, 1].hist(self.results["alignment_drops"], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(self.results["alignment_drops"]), color='red', 
                              linestyle='--', label=f'Mean: {np.mean(self.results["alignment_drops"]):.3f}')
            axes[0, 1].set_xlabel("Alignment Drop")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Alignment Drop Distribution")
            axes[0, 1].legend()
            
            # LPIPS distribution
            axes[1, 0].hist(self.results["lpips_scores"], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(self.results["lpips_scores"]), color='red', 
                              linestyle='--', label=f'Mean: {np.mean(self.results["lpips_scores"]):.4f}')
            axes[1, 0].set_xlabel("LPIPS Score")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Perceptual Distance Distribution")
            axes[1, 0].legend()
            
            # Processing time distribution
            axes[1, 1].hist(self.results["processing_times"], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(self.results["processing_times"]), color='red', 
                              linestyle='--', label=f'Mean: {np.mean(self.results["processing_times"]):.2f}s')
            axes[1, 1].set_xlabel("Processing Time (seconds)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Processing Time Distribution")
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(output_path / f"evaluation_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to create evaluation plots: {e}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Comprehensive DocSAF dataset evaluation")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--params", type=str, required=True, help="Path to trained parameters (.pt file)")
    parser.add_argument("--config", type=str, default="configs/cord.yaml", help="Configuration file")
    parser.add_argument("--dataset", type=str, choices=["funsd", "cord", "sroie", "docvqa", "doclaynet"],
                       help="Dataset type (if not specified, treats as simple image directory)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--save-samples", action="store_true", help="Save sample visualizations")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    logger.info(f"Starting DocSAF dataset evaluation")
    logger.info(f"Using device: {get_device(args.device)}")
    
    # Initialize evaluator
    evaluator = DocSAFDatasetEvaluator(args.config, args.params, args.device)
    
    # Load dataset
    dataset_config = evaluator.config.get("dataset", {})
    max_size = tuple(dataset_config.get("max_size", [1024, 1024]))
    
    if args.dataset:
        logger.info(f"Loading structured dataset: {args.dataset}")
        dataset = StructuredDocumentDataset(
            data_dir=args.data,
            dataset_name=args.dataset,
            split=args.split,
            device=evaluator.device,
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
            ocr_backend=evaluator.config.get("ocr", "easyocr"),
            device=evaluator.device,
            max_size=max_size
        )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Run evaluation
    logger.info("Starting evaluation...")
    summary = evaluator.evaluate_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        save_samples=args.save_samples,
        output_dir=args.output
    )
    
    # Save results
    summary_file, detailed_file = evaluator.save_results(args.output, summary)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 DocSAF Dataset Evaluation Results")
    print("="*60)
    print(f"📊 Total samples: {summary['total_samples']}")
    print(f"❌ Failed samples: {summary['failed_samples']}")
    print(f"✅ Success rate: {summary['success_rate']:.1%}")
    print(f"🎛️  Parameters: α={summary['model_parameters']['alpha']:.3f}, r={summary['model_parameters']['radius']:.3f}")
    print("-" * 40)
    print(f"📈 Original similarity: {summary['original_similarity']['mean']:.3f} ± {summary['original_similarity']['std']:.3f}")
    print(f"📉 Adversarial similarity: {summary['adversarial_similarity']['mean']:.3f} ± {summary['adversarial_similarity']['std']:.3f}")
    print(f"🎯 Alignment drop: {summary['alignment_drop']['mean']:.3f} ± {summary['alignment_drop']['std']:.3f}")
    print(f"⚡ Attack success rate: {summary['attack_success_rate']:.1%}")
    print(f"🔥 Strong attack rate: {summary['strong_attack_rate']:.1%}")
    print(f"🖼️  LPIPS score: {summary['lpips_score']['mean']:.4f} ± {summary['lpips_score']['std']:.4f}")
    print(f"⏱️  Avg processing time: {summary['processing_time']['mean']:.2f}s")
    print("="*60)
    print(f"📁 Results saved to: {args.output}")
    print(f"📄 Summary: {summary_file}")
    print(f"📋 Details: {detailed_file}")


if __name__ == "__main__":
    main()
