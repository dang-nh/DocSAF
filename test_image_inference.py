#!/usr/bin/env python3
"""
Single Image Inference and Visualization Script for DocSAF
Tests the trained model on individual images with detailed analysis and visualization.
"""

import argparse
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging
import cv2

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from docsaf.utils import load_config, setup_logging, get_device, pil_to_tensor, tensor_to_pil
from docsaf.surrogates import load_embedder
from docsaf.saliency import compute_gradient_saliency
from docsaf.field import apply_field
from docsaf.ocr import ocr_read
from docsaf.pdf_io import pdf_to_pil, is_pdf_file

logger = logging.getLogger(__name__)


class DocSAFImageTester:
    """Single image tester for DocSAF with comprehensive analysis."""
    
    def __init__(self, config_path: str, params_path: str, device: str = "auto"):
        """Initialize tester with config and trained parameters."""
        self.device = get_device(device)
        self.config = load_config(config_path)
        
        # Load trained parameters
        if not os.path.exists(params_path):
            raise ValueError(f"Parameters file not found: {params_path}")
        
        self.params = torch.load(params_path, map_location=self.device)
        self.alpha = self.params["alpha"]
        self.radius = self.params["radius"]
        
        logger.info(f"Loaded parameters: alpha={self.alpha:.3f}, radius={self.radius:.3f}")
        
        # Load embedder
        surrogate_specs = self.config.get("surrogates", ["openclip:ViT-L-14@336"])
        self.embedder = load_embedder(surrogate_specs[0], device=self.device)
        
        # OCR backend
        self.ocr_backend = self.config.get("ocr", "easyocr")
    
    def load_image(self, image_path: str) -> tuple[torch.Tensor, Image.Image]:
        """Load and preprocess image from file."""
        logger.info(f"Loading image: {image_path}")
        
        if is_pdf_file(image_path):
            logger.info("Converting PDF to image...")
            pil_image = pdf_to_pil(image_path, page=0, zoom=2.0)
        else:
            pil_image = Image.open(image_path).convert("RGB")
        
        # Convert to tensor
        image_tensor = pil_to_tensor(pil_image, self.device)
        
        return image_tensor, pil_image
    
    def extract_text(self, pil_image: Image.Image, custom_text: str = None) -> str:
        """Extract text from image using OCR or use custom text."""
        if custom_text:
            logger.info(f"Using custom text: '{custom_text[:100]}{'...' if len(custom_text) > 100 else ''}'")
            return custom_text
        
        # Convert PIL to numpy for OCR
        img_array = np.array(pil_image)
        text = ocr_read(img_array, backend=self.ocr_backend)
        
        if not text.strip():
            logger.warning("No text extracted. Using fallback text.")
            text = "document text content"
        else:
            logger.info(f"Extracted text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        return text
    
    def compute_similarity(self, image_tensor: torch.Tensor, text: str) -> float:
        """Compute image-text similarity using the embedder."""
        with torch.no_grad():
            img_emb = self.embedder.encode_image(image_tensor)
            text_emb = self.embedder.encode_text([text])
            similarity = torch.cosine_similarity(img_emb, text_emb, dim=-1)
            return float(similarity.item())
    
    def analyze_image(self, image_path: str, custom_text: str = None, 
                     save_analysis: bool = True, output_dir: str = None) -> dict:
        """Comprehensive analysis of a single image."""
        start_time = datetime.now()
        
        # Load image
        image_tensor, pil_image = self.load_image(image_path)
        
        # Extract text
        text = self.extract_text(pil_image, custom_text)
        
        # Compute original similarity
        orig_similarity = self.compute_similarity(image_tensor, text)
        logger.info(f"Original similarity: {orig_similarity:.4f}")
        
        # Compute saliency
        logger.info("Computing cross-modal saliency...")
        x_input = image_tensor.clone().requires_grad_(True)
        alignment_score, saliency_map = compute_gradient_saliency(
            self.embedder, x_input, text, normalize=True
        )
        
        # Apply adversarial field
        logger.info("Applying attenuation field...")
        with torch.no_grad():
            x_adv = apply_field(
                image_tensor.detach(),
                saliency_map.detach(),
                torch.tensor(self.alpha, device=self.device),
                torch.tensor(self.radius, device=self.device)
            )
        
        # Compute adversarial similarity
        adv_similarity = self.compute_similarity(x_adv, text)
        alignment_drop = orig_similarity - adv_similarity
        
        logger.info(f"Adversarial similarity: {adv_similarity:.4f}")
        logger.info(f"Alignment drop: {alignment_drop:.4f}")
        
        # Compute additional metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # LPIPS approximation
        lpips_score = float(torch.mean((image_tensor - x_adv) ** 2).item())
        
        # Saliency statistics
        saliency_stats = {
            "min": float(saliency_map.min()),
            "max": float(saliency_map.max()),
            "mean": float(saliency_map.mean()),
            "std": float(saliency_map.std())
        }
        
        # Create analysis results
        analysis = {
            "image_path": image_path,
            "text": text,
            "text_length": len(text),
            "original_similarity": orig_similarity,
            "adversarial_similarity": adv_similarity,
            "alignment_drop": alignment_drop,
            "lpips_score": lpips_score,
            "processing_time": processing_time,
            "saliency_stats": saliency_stats,
            "model_parameters": {
                "alpha": float(self.alpha),
                "radius": float(self.radius)
            },
            "success": alignment_drop > 0.1,
            "original_image": image_tensor,
            "adversarial_image": x_adv,
            "saliency_map": saliency_map,
            "original_pil": pil_image
        }
        
        # Save analysis if requested
        if save_analysis and output_dir:
            self.save_analysis_visualization(analysis, output_dir)
        
        # Clear gradients
        if x_input.grad is not None:
            x_input.grad.zero_()
        
        return analysis
    
    def save_analysis_visualization(self, analysis: dict, output_dir: str):
        """Save comprehensive visualization of the analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get image name for output files
        image_name = Path(analysis["image_path"]).stem
        
        # Convert tensors to PIL images
        orig_img = tensor_to_pil(analysis["original_image"])
        adv_img = tensor_to_pil(analysis["adversarial_image"])
        
        # Create saliency visualization
        saliency_map = analysis["saliency_map"].squeeze().cpu().numpy()
        saliency_img = Image.fromarray((saliency_map * 255).astype(np.uint8))
        
        # Create difference map
        diff_map = torch.abs(analysis["original_image"] - analysis["adversarial_image"])
        diff_map = torch.mean(diff_map, dim=0, keepdim=True)  # Convert to grayscale
        diff_img = tensor_to_pil(diff_map)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Main comparison
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(orig_img)
        ax1.set_title("Original Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Adversarial image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(adv_img)
        ax2.set_title("Adversarial Image", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Difference map
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(diff_img, cmap='hot')
        ax3.set_title("Difference Map", fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Saliency map
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(saliency_img, cmap='hot')
        ax4.set_title("Saliency Map", fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # Side-by-side comparison
        ax5 = fig.add_subplot(gs[1, :2])
        # Create side-by-side image
        width, height = orig_img.size
        combined = Image.new('RGB', (width * 2, height))
        combined.paste(orig_img, (0, 0))
        combined.paste(adv_img, (width, 0))
        ax5.imshow(combined)
        ax5.set_title("Side-by-Side Comparison", fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # Metrics visualization
        ax6 = fig.add_subplot(gs[1, 2:])
        metrics = {
            'Original Similarity': analysis['original_similarity'],
            'Adversarial Similarity': analysis['adversarial_similarity'],
            'Alignment Drop': analysis['alignment_drop'],
            'LPIPS Score': analysis['lpips_score']
        }
        
        bars = ax6.bar(metrics.keys(), metrics.values(), 
                      color=['blue', 'red', 'green', 'orange'])
        ax6.set_title("Performance Metrics", fontsize=14, fontweight='bold')
        ax6.set_ylabel("Score")
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Text analysis
        ax7 = fig.add_subplot(gs[2, :])
        text_info = f"""
        Extracted Text: "{analysis['text'][:200]}{'...' if len(analysis['text']) > 200 else ''}"
        
        Text Length: {analysis['text_length']} characters
        
        Model Parameters: α={analysis['model_parameters']['alpha']:.3f}, r={analysis['model_parameters']['radius']:.3f}
        
        Processing Time: {analysis['processing_time']:.2f} seconds
        
        Attack Success: {'✅ YES' if analysis['success'] else '❌ NO'} (drop > 0.1)
        """
        
        ax7.text(0.05, 0.95, text_info, transform=ax7.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax7.axis('off')
        
        # Save the comprehensive visualization
        plt.savefig(output_path / f"{image_name}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual images
        orig_img.save(output_path / f"{image_name}_original.png")
        adv_img.save(output_path / f"{image_name}_adversarial.png")
        saliency_img.save(output_path / f"{image_name}_saliency.png")
        diff_img.save(output_path / f"{image_name}_difference.png")
        
        # Also save the adversarial image with a clean name for easy access
        adv_img.save(output_path / f"{image_name}_perturbed.png")
        
        # Save analysis data
        analysis_data = {k: v for k, v in analysis.items() 
                        if k not in ['original_image', 'adversarial_image', 'saliency_map', 'original_pil']}
        with open(output_path / f"{image_name}_analysis.json", "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"Analysis saved to {output_path}")
    
    def batch_analyze_images(self, image_paths: list, output_dir: str = None) -> list:
        """Analyze multiple images in batch."""
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                analysis = self.analyze_image(
                    image_path, 
                    save_analysis=True, 
                    output_dir=output_dir
                )
                results.append(analysis)
                
                # Print quick summary
                print(f"  ✅ Success: {analysis['success']} | "
                      f"Drop: {analysis['alignment_drop']:.3f} | "
                      f"Time: {analysis['processing_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "success": False
                })
                print(f"  ❌ Failed: {e}")
        
        return results
    
    def create_batch_summary(self, results: list, output_dir: str):
        """Create summary visualization for batch results."""
        if not results:
            return
        
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            logger.warning("No successful analyses to summarize")
            return
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Alignment drops
        drops = [r['alignment_drop'] for r in successful_results]
        axes[0, 0].hist(drops, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(drops), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(drops):.3f}')
        axes[0, 0].set_xlabel("Alignment Drop")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Alignment Drop Distribution")
        axes[0, 0].legend()
        
        # Processing times
        times = [r['processing_time'] for r in successful_results]
        axes[0, 1].hist(times, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(times), color='red', linestyle='--',
                          label=f'Mean: {np.mean(times):.2f}s')
        axes[0, 1].set_xlabel("Processing Time (seconds)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Processing Time Distribution")
        axes[0, 1].legend()
        
        # Similarity comparison
        orig_sims = [r['original_similarity'] for r in successful_results]
        adv_sims = [r['adversarial_similarity'] for r in successful_results]
        axes[1, 0].scatter(orig_sims, adv_sims, alpha=0.6)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 0].set_xlabel("Original Similarity")
        axes[1, 0].set_ylabel("Adversarial Similarity")
        axes[1, 0].set_title("Similarity Degradation")
        
        # Success rate
        success_rate = len(successful_results) / len(results)
        axes[1, 1].pie([len(successful_results), len(results) - len(successful_results)],
                      labels=['Success', 'Failed'], autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title(f"Success Rate: {success_rate:.1%}")
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "batch_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save batch results
        batch_data = {
            "total_images": len(results),
            "successful_images": len(successful_results),
            "success_rate": success_rate,
            "avg_alignment_drop": float(np.mean(drops)) if drops else 0.0,
            "avg_processing_time": float(np.mean(times)) if times else 0.0,
            "results": results
        }
        
        with open(Path(output_dir) / "batch_results.json", "w") as f:
            json.dump(batch_data, f, indent=2, default=str)


def main():
    """Main image testing script."""
    parser = argparse.ArgumentParser(description="DocSAF single image inference and analysis")
    parser.add_argument("--image", type=str, help="Path to single image file")
    parser.add_argument("--images", type=str, nargs="+", help="Paths to multiple image files")
    parser.add_argument("--image-dir", type=str, help="Directory containing images to test")
    parser.add_argument("--params", type=str, required=True, help="Path to trained parameters (.pt file)")
    parser.add_argument("--config", type=str, default="configs/cord.yaml", help="Configuration file")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory")
    parser.add_argument("--custom-text", type=str, help="Custom text to use instead of OCR")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not any([args.image, args.images, args.image_dir]):
        parser.error("Must specify either --image, --images, or --image-dir")
    
    # Setup
    setup_logging(args.log_level)
    logger.info("Starting DocSAF image analysis")
    
    # Initialize tester
    tester = DocSAFImageTester(args.config, args.params, args.device)
    
    # Collect image paths
    image_paths = []
    
    if args.image:
        if not os.path.exists(args.image):
            raise ValueError(f"Image file not found: {args.image}")
        image_paths.append(args.image)
    
    if args.images:
        for img_path in args.images:
            if not os.path.exists(img_path):
                logger.warning(f"Image file not found: {img_path}")
                continue
            image_paths.append(img_path)
    
    if args.image_dir:
        if not os.path.exists(args.image_dir):
            raise ValueError(f"Image directory not found: {args.image_dir}")
        
        from docsaf.utils import get_image_files
        dir_images = get_image_files(args.image_dir)
        if not dir_images:
            raise ValueError(f"No image files found in {args.image_dir}")
        image_paths.extend(dir_images)
    
    if not image_paths:
        raise ValueError("No valid image files found")
    
    logger.info(f"Found {len(image_paths)} image(s) to analyze")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    if len(image_paths) == 1:
        # Single image analysis
        logger.info("Running single image analysis...")
        analysis = tester.analyze_image(
            image_paths[0], 
            custom_text=args.custom_text,
            save_analysis=True, 
            output_dir=args.output
        )
        
        # Print results
        print("\n" + "="*60)
        print("🎯 DocSAF Single Image Analysis Results")
        print("="*60)
        print(f"📄 Image: {analysis['image_path']}")
        print(f"📝 Text length: {analysis['text_length']} characters")
        print(f"🎛️  Parameters: α={analysis['model_parameters']['alpha']:.3f}, r={analysis['model_parameters']['radius']:.3f}")
        print("-" * 40)
        print(f"📊 Original similarity: {analysis['original_similarity']:.4f}")
        print(f"📉 Adversarial similarity: {analysis['adversarial_similarity']:.4f}")
        print(f"🎯 Alignment drop: {analysis['alignment_drop']:.4f}")
        print(f"🖼️  LPIPS score: {analysis['lpips_score']:.4f}")
        print(f"⏱️  Processing time: {analysis['processing_time']:.2f}s")
        print(f"✅ Attack success: {'YES' if analysis['success'] else 'NO'}")
        print("="*60)
        print(f"📁 Results saved to: {args.output}")
        
    else:
        # Batch analysis
        logger.info("Running batch image analysis...")
        results = tester.batch_analyze_images(image_paths, args.output)
        
        # Create batch summary
        tester.create_batch_summary(results, args.output)
        
        # Print summary
        successful = [r for r in results if r.get('success', False)]
        print("\n" + "="*60)
        print("🎯 DocSAF Batch Analysis Results")
        print("="*60)
        print(f"📊 Total images: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(results) - len(successful)}")
        print(f"📈 Success rate: {len(successful)/len(results):.1%}")
        
        if successful:
            avg_drop = np.mean([r['alignment_drop'] for r in successful])
            avg_time = np.mean([r['processing_time'] for r in successful])
            print(f"🎯 Average alignment drop: {avg_drop:.3f}")
            print(f"⏱️  Average processing time: {avg_time:.2f}s")
        
        print("="*60)
        print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
