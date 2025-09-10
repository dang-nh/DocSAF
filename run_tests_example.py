#!/usr/bin/env python3
"""
Example script showing how to use the DocSAF test scripts
with the trained model from train_cord_20250909_091350.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print("Error:", e.stderr)
        return False

def main():
    """Run example tests with the trained model."""
    
    # Paths
    trained_params = "runs/train_cord_20250909_091350/universal.pt"
    config_file = "configs/cord.yaml"
    demo_dir = "demo"
    output_dir = "test_results"
    
    print("🎯 DocSAF Test Scripts Example")
    print("=" * 50)
    print(f"Using trained parameters: {trained_params}")
    print(f"Using config: {config_file}")
    print(f"Demo images directory: {demo_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if files exist
    if not Path(trained_params).exists():
        print(f"❌ Trained parameters not found: {trained_params}")
        return
    
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        return
    
    if not Path(demo_dir).exists():
        print(f"❌ Demo directory not found: {demo_dir}")
        return
    
    # Test 1: Single image inference
    demo_images = list(Path(demo_dir).glob("*.png"))
    if demo_images:
        test_image = str(demo_images[0])
        
        cmd = [
            sys.executable, "test_image_inference.py",
            "--image", test_image,
            "--params", trained_params,
            "--config", config_file,
            "--output", f"{output_dir}/single_image",
            "--log-level", "INFO"
        ]
        
        success = run_command(cmd, f"Testing single image: {test_image}")
        if not success:
            print("⚠️  Single image test failed, but continuing...")
    
    # Test 2: Batch image inference
    if demo_images:
        cmd = [
            sys.executable, "test_image_inference.py",
            "--image-dir", demo_dir,
            "--params", trained_params,
            "--config", config_file,
            "--output", f"{output_dir}/batch_images",
            "--log-level", "INFO"
        ]
        
        success = run_command(cmd, f"Testing batch images from: {demo_dir}")
        if not success:
            print("⚠️  Batch image test failed, but continuing...")
    
    # Test 3: Dataset evaluation (if CORD dataset is available)
    cord_data_dir = "data/cord"
    if Path(cord_data_dir).exists():
        cmd = [
            sys.executable, "test_dataset_evaluation.py",
            "--data", cord_data_dir,
            "--params", trained_params,
            "--config", config_file,
            "--dataset", "cord",
            "--split", "test",
            "--output", f"{output_dir}/dataset_eval",
            "--max-samples", "10",  # Limit for quick test
            "--log-level", "INFO"
        ]
        
        success = run_command(cmd, f"Testing dataset evaluation on CORD test set")
        if not success:
            print("⚠️  Dataset evaluation test failed, but continuing...")
    else:
        print(f"⚠️  CORD dataset not found at {cord_data_dir}, skipping dataset test")
    
    print("\n" + "=" * 50)
    print("🎉 Test examples completed!")
    print(f"📁 Check results in: {output_dir}/")
    print("\nTo run individual tests:")
    print("1. Single image: python test_image_inference.py --image demo/form.png --params runs/train_cord_20250909_091350/universal.pt")
    print("2. Batch images: python test_image_inference.py --image-dir demo --params runs/train_cord_20250909_091350/universal.pt")
    print("3. Dataset eval: python test_dataset_evaluation.py --data data/cord --dataset cord --params runs/train_cord_20250909_091350/universal.pt")

if __name__ == "__main__":
    main()
