#!/usr/bin/env python3
"""Quick demo of the trained DocSAF model."""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from docsaf.surrogates import load_embedder
from docsaf.saliency import compute_gradient_saliency
from docsaf.field import apply_field
from docsaf.utils import pil_to_tensor
from docsaf.ocr import ocr_read

def quick_demo():
    """Quick demo with a sample image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load trained parameters
    params_path = "runs/train_cord_20250909_022942/universal.pt"
    params = torch.load(params_path, map_location='cpu')
    alpha = torch.tensor(params['alpha'], requires_grad=False).to(device)
    radius = torch.tensor(params['radius'], requires_grad=False).to(device)
    
    print(f"Loaded trained parameters:")
    print(f"  Alpha: {alpha.item():.4f}")
    print(f"  Radius: {radius.item():.4f}")
    
    # Load embedder
    embedder = load_embedder("openclip:ViT-L-14@336", device)
    
    # Use a sample image from demo folder
    image_path = "demo/receipt.png"
    if not Path(image_path).exists():
        print(f"Demo image not found: {image_path}")
        print("Please provide an image path or use the test script with --image")
        return
    
    # Load and preprocess image
    pil_image = Image.open(image_path).convert("RGB")
    original_np = np.array(pil_image)
    
    # Resize to consistent size for processing
    pil_image_resized = pil_image.resize((1024, 1024), Image.Resampling.LANCZOS)
    image_tensor = pil_to_tensor(pil_image_resized, device)  # (1, 3, H, W)
    
    # Also resize original for visualization
    original_np = np.array(pil_image_resized)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Extract OCR text
    text = ocr_read(original_np, backend="easyocr")
    if not text.strip():
        text = "document text content"
    print(f"Extracted text: {text[:100]}...")
    
    # Compute saliency
    print("Computing saliency...")
    image_sal = image_tensor.requires_grad_(True)  # Already (1, 3, H, W)
    _, saliency_map = compute_gradient_saliency(embedder, image_sal, text, normalize=True)
    saliency_map = saliency_map.detach()
    
    # Apply field
    print("Applying attenuation field...")
    image_batch = image_tensor  # Already (1, 3, H, W)
    radius_pos = F.softplus(radius) + 1e-3
    x_adv, mask_A = apply_field(image_batch, saliency_map, alpha, radius_pos)
    
    # Convert back to PIL for visualization
    adversarial_np = (x_adv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    
    # Compute alignment drop
    print("Computing alignment drop...")
    with torch.no_grad():
        # Original alignment
        orig_emb = F.normalize(embedder.image_embed(image_batch, requires_grad=False), dim=-1)
        txt_emb = F.normalize(embedder.text_embed([text]), dim=-1)
        orig_align = (orig_emb * txt_emb).sum(dim=-1).item()
        
        # Adversarial alignment
        adv_emb = F.normalize(embedder.image_embed(x_adv, requires_grad=False), dim=-1)
        adv_align = (adv_emb * txt_emb).sum(dim=-1).item()
        
        alignment_drop = orig_align - adv_align
    
    print(f"\nResults:")
    print(f"  Original alignment: {orig_align:.4f}")
    print(f"  Adversarial alignment: {adv_align:.4f}")
    print(f"  Alignment drop: {alignment_drop:.4f}")
    print(f"  Attack success: {'Yes' if alignment_drop > 0 else 'No'}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Adversarial image
    axes[0, 1].imshow(adversarial_np)
    axes[0, 1].set_title('Adversarial Image')
    axes[0, 1].axis('off')
    
    # Difference
    diff = np.abs(adversarial_np.astype(float) - original_np.astype(float))
    axes[0, 2].imshow(diff.astype(np.uint8))
    axes[0, 2].set_title('Perturbation (Absolute Difference)')
    axes[0, 2].axis('off')
    
    # Saliency map
    saliency_np = saliency_map.squeeze().cpu().numpy()
    im1 = axes[1, 0].imshow(saliency_np, cmap='hot')
    axes[1, 0].set_title('Saliency Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Attenuation mask A
    mask_np = mask_A.squeeze().cpu().numpy()
    im2 = axes[1, 1].imshow(mask_np, cmap='viridis')
    axes[1, 1].set_title('Attenuation Mask A')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Text and alignment info
    axes[1, 2].text(0.1, 0.8, f'Text: {text[:100]}...', fontsize=10, wrap=True)
    axes[1, 2].text(0.1, 0.6, f'Alignment Drop: {alignment_drop:.4f}', fontsize=12, 
                    color='red' if alignment_drop > 0 else 'green', weight='bold')
    axes[1, 2].text(0.1, 0.4, f'Success: {"Yes" if alignment_drop > 0 else "No"}', fontsize=12,
                    color='red' if alignment_drop > 0 else 'green', weight='bold')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_path = "demo_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    quick_demo()
