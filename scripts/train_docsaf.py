#!/usr/bin/env python3
"""Training script for DocSAF universal parameters on various datasets."""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from docsaf.train_universal import main as train_main


def create_parser():
    """Create argument parser with predefined dataset configurations."""
    parser = argparse.ArgumentParser(
        description="Train DocSAF universal parameters on document datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on FUNSD dataset
  python scripts/train_docsaf.py --dataset funsd --data data/funsd

  # Train on CORD dataset  
  python scripts/train_docsaf.py --dataset cord --data data/cord

  # Train on DocVQA dataset
  python scripts/train_docsaf.py --dataset docvqa --data data/docvqa

  # Train on SROIE dataset
  python scripts/train_docsaf.py --dataset sroie --data data/sroie

  # Train on simple image directory
  python scripts/train_docsaf.py --data data/test_docs

  # Use custom configuration
  python scripts/train_docsaf.py --dataset funsd --data data/funsd --config configs/funsd.yaml
        """
    )
    
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset", type=str, 
        choices=["funsd", "cord", "sroie", "docvqa", "doclaynet"],
        help="Dataset type (if not specified, treats as simple image directory)"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output", type=str, default="runs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (auto/cuda/cpu)"
    )
    parser.add_argument(
        "--steps", type=int,
        help="Number of training steps (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser


def validate_args(args):
    """Validate command line arguments."""
    # Check if data directory exists
    if not os.path.exists(args.data):
        raise ValueError(f"Data directory does not exist: {args.data}")
    
    # Check if config file exists
    if not os.path.exists(args.config):
        raise ValueError(f"Config file does not exist: {args.config}")
    
    # For structured datasets, check if dataset-specific subdirectories exist
    if args.dataset:
        expected_subdirs = {
            "funsd": ["train", "test"],
            "cord": ["train", "dev", "test"], 
            "sroie": ["train", "test"],
            "docvqa": ["images", "ocrs"],
            "doclaynet": ["test"]
        }
        
        if args.dataset in expected_subdirs:
            dataset_path = Path(args.data)
            required_dirs = expected_subdirs[args.dataset]
            
            # Check if at least one required directory exists
            if not any((dataset_path / subdir).exists() for subdir in required_dirs):
                print(f"Warning: Expected subdirectories {required_dirs} not found in {args.data}")
                print("Proceeding anyway - the dataset loader will handle missing directories.")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Convert args to sys.argv format for train_universal.main()
        sys.argv = ["train_universal.py"]
        sys.argv.extend(["--data", args.data])
        
        if args.dataset:
            sys.argv.extend(["--dataset", args.dataset])
        
        sys.argv.extend(["--split", args.split])
        sys.argv.extend(["--config", args.config])
        sys.argv.extend(["--output", args.output])
        sys.argv.extend(["--device", args.device])
        sys.argv.extend(["--log-level", args.log_level])
        
        if args.seed is not None:
            sys.argv.extend(["--seed", str(args.seed)])
        
        # Print configuration
        print("=== DocSAF Training Configuration ===")
        print(f"Dataset: {args.dataset or 'Simple images'}")
        print(f"Data directory: {args.data}")
        print(f"Split: {args.split}")
        print(f"Config: {args.config}")
        print(f"Output: {args.output}")
        print(f"Device: {args.device}")
        if args.seed:
            print(f"Seed: {args.seed}")
        print("=" * 40)
        
        # Run training
        train_main()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
