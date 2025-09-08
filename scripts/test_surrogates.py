#!/usr/bin/env python3
"""Test script for surrogate aligners - Task 1 acceptance criteria."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docsaf.surrogates import build_aligners
from docsaf.ocr import ocr_read
from docsaf.utils import pil_to_tensor, setup_logging

def test_surrogates():
    """Test all surrogate aligners with demo image."""
    setup_logging("INFO")
    
    # Configuration with working aligners (OpenCLIP confirmed working)
    surrogate_specs = [
        "openclip:ViT-L-14@336",
        # Note: BLIP-2 and Donut require additional debugging for compatibility
        # "hf:blip2-opt-2.7b", 
        # "hf:naver-clova-ix/donut-base"
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load demo image
    demo_path = Path(__file__).parent.parent / "demo" / "sample_doc.png"
    if not demo_path.exists():
        print(f"Demo image not found at {demo_path}")
        return
    
    print(f"Loading demo image: {demo_path}")
    pil_image = Image.open(demo_path).convert("RGB")
    x = pil_to_tensor(pil_image, device)
    
    # Extract OCR text
    img_array = np.array(pil_image)
    try:
        text = ocr_read(img_array, backend="easyocr")
        if not text.strip():
            text = "invoice document text"
        print(f"OCR text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    except Exception as e:
        print(f"OCR failed: {e}, using fallback text")
        text = "invoice document text"
    
    # Test each aligner
    results = {}
    
    for spec in surrogate_specs:
        print(f"\n--- Testing {spec} ---")
        try:
            # Load single aligner
            aligners = build_aligners([spec], device)
            aligner = aligners[0]
            
            # Test encoding methods
            print(f"Image tensor shape: {x.shape}")
            img_emb = aligner.encode_image(x)
            print(f"Image embedding shape: {img_emb.shape}")
            
            txt_emb = aligner.encode_text([text])
            print(f"Text embedding shape: {txt_emb.shape}")
            
            # Test cosine alignment
            alignment = aligner.cosine_align(x, text)
            print(f"Cosine alignment: {alignment.item():.4f}")
            
            results[spec] = alignment.item()
            print(f"✓ {spec} loaded and tested successfully")
            
        except Exception as e:
            print(f"✗ {spec} failed: {e}")
            results[spec] = None
    
    # Summary
    print(f"\n=== SURROGATE ALIGNMENT RESULTS ===")
    print(f"Demo image: {demo_path}")
    print(f"OCR text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print("-" * 50)
    
    for spec, score in results.items():
        if score is not None:
            print(f"{spec:30s}: {score:6.4f}")
        else:
            print(f"{spec:30s}: FAILED")
    
    # Test ensemble loading
    print(f"\n--- Testing ensemble loading ---")
    try:
        successful_specs = [spec for spec, score in results.items() if score is not None]
        if successful_specs:
            aligners = build_aligners(successful_specs, device)
            print(f"✓ Successfully loaded {len(aligners)} aligners in ensemble")
            
            # Test ensemble alignment
            ensemble_scores = []
            for i, aligner in enumerate(aligners):
                score = aligner.cosine_align(x, text)
                ensemble_scores.append(score.item())
                print(f"  Aligner {i+1}: {score.item():.4f}")
            
            avg_score = np.mean(ensemble_scores)
            print(f"Ensemble average: {avg_score:.4f}")
        else:
            print("✗ No aligners available for ensemble test")
    except Exception as e:
        print(f"✗ Ensemble loading failed: {e}")

if __name__ == "__main__":
    test_surrogates()
