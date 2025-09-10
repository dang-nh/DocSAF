#!/usr/bin/env python3
"""
Comprehensive example script demonstrating DocSAF training and evaluation on various datasets.

This script shows how to:
1. Train universal parameters on different datasets
2. Evaluate trained models
3. Compare results across datasets
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import json
from datetime import datetime

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_dataset_availability(data_root):
    """Check which datasets are available."""
    datasets = {}
    
    # Check FUNSD
    funsd_path = Path(data_root) / "funsd"
    if funsd_path.exists() and (funsd_path / "train").exists():
        datasets["funsd"] = str(funsd_path)
    
    # Check CORD
    cord_path = Path(data_root) / "cord"
    if cord_path.exists() and (cord_path / "train").exists():
        datasets["cord"] = str(cord_path)
    
    # Check SROIE
    sroie_path = Path(data_root) / "sroie"
    if sroie_path.exists() and (sroie_path / "train").exists():
        datasets["sroie"] = str(sroie_path)
    
    # Check DocVQA
    docvqa_path = Path(data_root) / "docvqa"
    if docvqa_path.exists() and (docvqa_path / "images").exists():
        datasets["docvqa"] = str(docvqa_path)
    
    # Check DocLayNet
    doclaynet_path = Path(data_root) / "doclaynet"
    if doclaynet_path.exists() and (doclaynet_path / "test").exists():
        datasets["doclaynet"] = str(doclaynet_path)
    
    # Check simple images
    test_docs_path = Path(data_root) / "test_docs"
    if test_docs_path.exists():
        datasets["simple"] = str(test_docs_path)
    
    return datasets


def train_dataset(dataset_name, data_path, output_dir, args):
    """Train DocSAF on a specific dataset."""
    print(f"\n🚀 Training on {dataset_name.upper()} dataset...")
    
    cmd = [
        "python", "scripts/train_docsaf.py",
        "--data", data_path,
        "--output", output_dir,
        "--device", args.device,
        "--log-level", args.log_level
    ]
    
    if dataset_name != "simple":
        cmd.extend(["--dataset", dataset_name])
        cmd.extend(["--config", f"configs/{dataset_name}.yaml"])
    
    if args.quick:
        # Override config for quick testing
        cmd.extend(["--steps", "40"])
        cmd.extend(["--batch-size", "4"])
    
    return run_command(cmd, f"Training on {dataset_name}")


def evaluate_dataset(dataset_name, data_path, params_path, output_dir, args):
    """Evaluate trained model on a dataset."""
    print(f"\n📊 Evaluating on {dataset_name.upper()} dataset...")
    
    cmd = [
        "python", "scripts/test_docsaf.py",
        "--data", data_path,
        "--params", params_path,
        "--output", output_dir,
        "--device", args.device,
        "--batch-size", str(args.eval_batch_size),
        "--log-level", args.log_level
    ]
    
    if dataset_name != "simple":
        cmd.extend(["--dataset", dataset_name])
        cmd.extend(["--split", "test"])
        cmd.extend(["--config", f"configs/{dataset_name}.yaml"])
    
    if args.quick:
        cmd.extend(["--max-samples", "10"])
    
    return run_command(cmd, f"Evaluating on {dataset_name}")


def run_full_pipeline(args):
    """Run the complete training and evaluation pipeline."""
    # Check available datasets
    datasets = check_dataset_availability(args.data_root)
    
    if not datasets:
        print("❌ No datasets found in the specified data root directory!")
        print(f"Data root: {args.data_root}")
        return
    
    print(f"\n📁 Available datasets:")
    for name, path in datasets.items():
        print(f"  - {name}: {path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / f"pipeline_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": timestamp,
        "datasets": datasets,
        "training_results": {},
        "evaluation_results": {}
    }
    
    # Training phase
    print(f"\n🏃‍♂️ TRAINING PHASE")
    print(f"Output directory: {run_dir}")
    
    trained_models = {}
    
    for dataset_name, data_path in datasets.items():
        if args.datasets and dataset_name not in args.datasets:
            print(f"⏭️  Skipping {dataset_name} (not in selected datasets)")
            continue
        
        success = train_dataset(dataset_name, data_path, str(run_dir), args)
        results["training_results"][dataset_name] = success
        
        if success:
            # Find the latest training output directory
            train_dirs = list(run_dir.glob(f"train*{dataset_name}*" if dataset_name != "simple" else "train_*"))
            if train_dirs:
                latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                params_file = latest_dir / "universal.pt"
                if params_file.exists():
                    trained_models[dataset_name] = str(params_file)
                    print(f"✅ Model saved: {params_file}")
    
    # Evaluation phase
    if trained_models and not args.train_only:
        print(f"\n📊 EVALUATION PHASE")
        
        eval_dir = run_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        for dataset_name, params_path in trained_models.items():
            if dataset_name in datasets:
                success = evaluate_dataset(
                    dataset_name, 
                    datasets[dataset_name], 
                    params_path, 
                    str(eval_dir), 
                    args
                )
                results["evaluation_results"][dataset_name] = success
    
    # Save pipeline results summary
    results_file = run_dir / "pipeline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory: {run_dir}")
    print(f"Results file: {results_file}")
    
    if trained_models:
        print(f"\nTrained models:")
        for dataset, path in trained_models.items():
            print(f"  - {dataset}: {path}")
    
    if results["evaluation_results"]:
        print(f"\nEvaluation results:")
        for dataset, success in results["evaluation_results"].items():
            status = "✅" if success else "❌"
            print(f"  - {dataset}: {status}")
    
    print(f"\n🎉 Pipeline completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run DocSAF training and evaluation examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on all available datasets
  python scripts/run_examples.py --data-root data

  # Run only on specific datasets
  python scripts/run_examples.py --data-root data --datasets funsd cord

  # Quick test run (small number of steps)
  python scripts/run_examples.py --data-root data --quick

  # Train only (no evaluation)
  python scripts/run_examples.py --data-root data --train-only

  # List available datasets
  python scripts/run_examples.py --data-root data --list-only
        """
    )
    
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+", 
        choices=["funsd", "cord", "sroie", "docvqa", "doclaynet", "simple"],
        help="Specific datasets to process (default: all available)"
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
        "--quick", action="store_true",
        help="Quick test run with minimal steps"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only run training, skip evaluation"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only list available datasets and exit"
    )
    
    args = parser.parse_args()
    
    # Check if data root exists
    if not os.path.exists(args.data_root):
        print(f"❌ Data root directory not found: {args.data_root}")
        sys.exit(1)
    
    # List datasets and exit if requested
    if args.list_only:
        datasets = check_dataset_availability(args.data_root)
        print(f"Available datasets in {args.data_root}:")
        for name, path in datasets.items():
            print(f"  - {name}: {path}")
        return
    
    # Run the pipeline
    try:
        run_full_pipeline(args)
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
