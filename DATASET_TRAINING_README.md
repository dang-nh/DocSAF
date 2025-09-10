# DocSAF Dataset Training and Evaluation

This document describes how to train and evaluate DocSAF on various document understanding datasets.

## Supported Datasets

- **FUNSD**: Form Understanding in Noisy Scanned Documents
- **CORD**: Consolidated Receipt Dataset  
- **SROIE**: Scanned Receipts OCR and Information Extraction
- **DocVQA**: Document Visual Question Answering
- **DocLayNet**: Document Layout Analysis

## Quick Start

### 1. Training on a Specific Dataset

```bash
# Train on FUNSD dataset
python scripts/train_docsaf.py --dataset funsd --data data/funsd

# Train on CORD dataset
python scripts/train_docsaf.py --dataset cord --data data/cord

# Train on DocVQA dataset
python scripts/train_docsaf.py --dataset docvqa --data data/docvqa
```

### 2. Evaluating Trained Models

```bash
# Evaluate on test set
python scripts/test_docsaf.py --dataset funsd --data data/funsd --params runs/train_funsd_*/universal.pt

# Evaluate with custom parameters
python scripts/test_docsaf.py --dataset cord --data data/cord --params runs/universal.pt --max-samples 100
```

### 3. Running Full Pipeline

```bash
# Run complete training and evaluation pipeline
python scripts/run_examples.py --data-root data

# Quick test run
python scripts/run_examples.py --data-root data --quick

# Process only specific datasets
python scripts/run_examples.py --data-root data --datasets funsd cord
```

## Directory Structure

Your data directory should be organized as follows:

```
data/
├── funsd/
│   ├── train/
│   │   ├── annotations/
│   │   └── images/
│   └── test/
│       ├── annotations/
│       └── images/
├── cord/
│   ├── train/
│   │   ├── image/
│   │   └── json/
│   ├── dev/
│   └── test/
├── sroie/
│   ├── train/
│   │   ├── images/
│   │   ├── tagged/
│   │   └── ocr/
│   └── test/
├── docvqa/
│   ├── images/
│   ├── ocrs/
│   ├── train_v1.0_withQT.json
│   └── val_v1.0_withQT.json
└── doclaynet/
    └── test/
        ├── annotations/
        └── images/
```

## Configuration Files

Dataset-specific configurations are available in the `configs/` directory:

- `configs/funsd.yaml` - Optimized for form understanding
- `configs/cord.yaml` - Optimized for receipt processing
- `configs/sroie.yaml` - Optimized for receipt OCR/IE
- `configs/docvqa.yaml` - Optimized for visual question answering
- `configs/doclaynet.yaml` - Optimized for layout analysis
- `configs/default.yaml` - General purpose configuration

## Training Scripts

### `scripts/train_docsaf.py`

Main training script with dataset support.

**Arguments:**
- `--data`: Path to dataset directory (required)
- `--dataset`: Dataset type (funsd/cord/sroie/docvqa/doclaynet)
- `--split`: Dataset split (train/val/test, default: train)
- `--config`: Configuration file path
- `--output`: Output directory for results
- `--device`: Device to use (auto/cuda/cpu)
- `--seed`: Random seed

**Examples:**
```bash
# Basic training
python scripts/train_docsaf.py --dataset funsd --data data/funsd

# Custom configuration
python scripts/train_docsaf.py --dataset cord --data data/cord --config configs/cord.yaml

# Train on validation split
python scripts/train_docsaf.py --dataset docvqa --data data/docvqa --split val
```

### `scripts/test_docsaf.py`

Evaluation script for trained models.

**Arguments:**
- `--data`: Path to dataset directory (required)
- `--params`: Path to trained parameters file (required)
- `--dataset`: Dataset type
- `--split`: Dataset split (default: test)
- `--config`: Configuration file path
- `--output`: Output directory for results
- `--batch-size`: Evaluation batch size
- `--max-samples`: Maximum samples to evaluate

**Examples:**
```bash
# Basic evaluation
python scripts/test_docsaf.py --dataset funsd --data data/funsd --params runs/universal.pt

# Limited evaluation
python scripts/test_docsaf.py --dataset cord --data data/cord --params runs/train_cord_*/universal.pt --max-samples 50
```

### `scripts/run_examples.py`

Comprehensive pipeline script that runs training and evaluation.

**Arguments:**
- `--data-root`: Root directory containing datasets
- `--datasets`: Specific datasets to process
- `--output`: Output directory
- `--quick`: Quick test run with minimal steps
- `--train-only`: Only run training, skip evaluation
- `--list-only`: List available datasets and exit

**Examples:**
```bash
# Full pipeline
python scripts/run_examples.py --data-root data

# Quick test
python scripts/run_examples.py --data-root data --quick

# Specific datasets only
python scripts/run_examples.py --data-root data --datasets funsd cord

# List available datasets
python scripts/run_examples.py --data-root data --list-only
```

## Output Structure

Training outputs are organized as follows:

```
runs/
├── train_funsd_20240101_120000/
│   ├── universal.pt              # Trained parameters
│   ├── training_report.json      # Training statistics
│   └── config.yaml              # Used configuration
├── train_cord_20240101_130000/
└── evaluations/
    ├── eval_funsd_test_20240101_140000.json
    └── eval_cord_test_20240101_140000.json
```

## Dataset-Specific Notes

### FUNSD
- Focused on form understanding and entity extraction
- Uses line-level bounding boxes
- Smaller dataset, requires fewer training steps

### CORD
- Receipt understanding with structured information extraction
- Robust to image quality variations
- Good for testing EOT transforms

### SROIE
- Similar to CORD but with different annotation format
- OCR and information extraction tasks
- Line-level processing recommended

### DocVQA
- Visual question answering on documents
- Combines questions with document context
- Requires careful handling of text length
- Uses precomputed JSONL files for efficiency

### DocLayNet
- Document layout analysis
- Element-level rather than line-level processing
- Larger images, may require more memory

## Performance Tips

1. **Memory Management**: Reduce batch size if you encounter CUDA out of memory errors
2. **Quick Testing**: Use `--quick` flag for rapid prototyping
3. **Dataset Size**: Start with FUNSD (smaller) before moving to DocVQA (larger)
4. **GPU Utilization**: Monitor GPU memory usage with `nvidia-smi`
5. **Configuration**: Use dataset-specific configs for optimal performance

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Check directory structure matches expected format
   - Verify file permissions
   - Use `--list-only` to check available datasets

2. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 2`
   - Reduce image size in config: `max_size: [512, 512]`
   - Use CPU: `--device cpu`

3. **Import Errors**
   - Ensure data/utils.py is accessible
   - Check Python path configuration
   - Verify all dependencies are installed

4. **Slow Training**
   - Use GPU: `--device cuda`
   - Increase batch size if memory allows
   - Reduce number of steps for testing

### Getting Help

Check the logs for detailed error messages. The scripts provide comprehensive logging at INFO level by default. Use `--log-level DEBUG` for more detailed output.
