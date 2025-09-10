# DocSAF Test Scripts

This directory contains comprehensive test scripts for evaluating the trained DocSAF model on both individual images and datasets.

## 📊 Training Results Summary

Based on the training results from `runs/train_cord_20250909_091350/`:

- **Final Loss**: 0.194
- **Trained Parameters**:
  - `alpha`: 0.681 (field strength)
  - `radius`: 6.177 (blur kernel size)
- **Training Config**:
  - Steps: 2000
  - Batch size: 8
  - Learning rate: 0.001
  - TV lambda: 0.002

## 🧪 Test Scripts

### 1. `test_image_inference.py` - Single Image Analysis

**Purpose**: Comprehensive analysis of individual images with detailed visualization and metrics.

**Features**:
- Single image or batch processing
- OCR text extraction or custom text input
- Cross-modal saliency computation
- Adversarial image generation
- Detailed metrics (similarity, LPIPS, processing time)
- Comprehensive visualizations (original, adversarial, saliency, difference maps)
- JSON output with analysis data

**Usage Examples**:

```bash
# Single image analysis
python test_image_inference.py \
    --image demo/form.png \
    --params runs/train_cord_20250909_091350/universal.pt \
    --config configs/cord.yaml \
    --output test_results/single

# Batch processing from directory
python test_image_inference.py \
    --image-dir demo \
    --params runs/train_cord_20250909_091350/universal.pt \
    --output test_results/batch

# With custom text (skip OCR)
python test_image_inference.py \
    --image demo/receipt.png \
    --custom-text "Invoice #12345 Total: $99.99" \
    --params runs/train_cord_20250909_091350/universal.pt
```

**Output Files**:
- `{image_name}_analysis.png` - Comprehensive visualization
- `{image_name}_original.png` - Original image
- `{image_name}_adversarial.png` - Adversarial image
- `{image_name}_perturbed.png` - Adversarial image (clean name for easy access)
- `{image_name}_saliency.png` - Saliency map
- `{image_name}_difference.png` - Difference map
- `{image_name}_analysis.json` - Detailed metrics
- `batch_summary.png` - Batch results summary
- `batch_results.json` - Batch analysis data

### 2. `test_dataset_evaluation.py` - Dataset Evaluation

**Purpose**: Comprehensive evaluation of DocSAF on document datasets with statistical analysis.

**Features**:
- Support for structured datasets (CORD, FUNSD, SROIE, DocVQA, DocLayNet)
- Batch processing with progress tracking
- Attack success rate computation
- Statistical analysis (mean, std, distributions)
- Sample visualization saving
- Comprehensive evaluation plots
- JSON reports with detailed metrics

**Usage Examples**:

```bash
# CORD dataset evaluation
python test_dataset_evaluation.py \
    --data data/cord \
    --dataset cord \
    --split test \
    --params runs/train_cord_20250909_091350/universal.pt \
    --config configs/cord.yaml \
    --output test_results/cord_eval

# FUNSD dataset evaluation
python test_dataset_evaluation.py \
    --data data/funsd \
    --dataset funsd \
    --split test \
    --params runs/train_cord_20250909_091350/universal.pt \
    --max-samples 50

# Simple image directory evaluation
python test_dataset_evaluation.py \
    --data demo \
    --params runs/train_cord_20250909_091350/universal.pt \
    --output test_results/demo_eval
```

**Output Files**:
- `evaluation_summary_{timestamp}.json` - Summary statistics
- `detailed_results_{timestamp}.json` - Detailed per-sample results
- `evaluation_plots_{timestamp}.png` - Statistical plots
- `sample_{id:04d}_combined.png` - Combined visualization (original, adversarial, difference, saliency)
- `sample_{id:04d}_original.png` - Original image
- `sample_{id:04d}_adversarial.png` - Adversarial image
- `sample_{id:04d}_perturbed.png` - Adversarial image (clean name for easy access)
- `sample_{id:04d}_saliency.png` - Saliency map
- `sample_{id:04d}_difference.png` - Difference map

### 3. `run_tests_example.py` - Quick Test Runner

**Purpose**: Automated example runner that demonstrates all test scripts.

**Usage**:
```bash
python run_tests_example.py
```

This script will:
1. Test single image inference on demo images
2. Test batch image processing
3. Test dataset evaluation (if CORD dataset is available)
4. Generate example outputs in `test_results/`

## 📈 Key Metrics Explained

### Alignment Metrics
- **Original Similarity**: Image-text similarity before attack
- **Adversarial Similarity**: Image-text similarity after attack
- **Alignment Drop**: Difference between original and adversarial similarity
- **Attack Success Rate**: Percentage of samples with alignment drop > 0.1
- **Strong Attack Rate**: Percentage of samples with alignment drop > 0.2

### Perceptual Quality
- **LPIPS Score**: Perceptual distance between original and adversarial images
- **Processing Time**: Time taken to generate adversarial image

### Model Parameters
- **Alpha (α)**: Field strength parameter controlling attenuation intensity
- **Radius (r)**: Gaussian blur kernel size controlling spatial extent

## 🎯 Expected Results

Based on the training results, you should expect:

- **Alignment Drop**: ~0.1-0.3 (depending on image complexity)
- **Attack Success Rate**: 60-80% (samples with drop > 0.1)
- **LPIPS Score**: 0.01-0.05 (low perceptual distortion)
- **Processing Time**: 1-3 seconds per image (depending on hardware)

## 🔧 Configuration

Both scripts use the same configuration system as the main DocSAF pipeline:

- **Config File**: `configs/cord.yaml` (or specify with `--config`)
- **Parameters File**: `runs/train_cord_20250909_091350/universal.pt`
- **Device**: Auto-detection (CUDA if available, else CPU)

## 📁 Output Structure

```
test_results/
├── single_image/
│   ├── form_analysis.png
│   ├── form_original.png
│   ├── form_adversarial.png
│   ├── form_perturbed.png
│   ├── form_saliency.png
│   ├── form_difference.png
│   └── form_analysis.json
├── batch_images/
│   ├── batch_summary.png
│   ├── batch_results.json
│   └── [individual sample files]
└── dataset_eval/
    ├── evaluation_summary_20250909_120000.json
    ├── detailed_results_20250909_120000.json
    ├── evaluation_plots_20250909_120000.png
    ├── sample_0000_combined.png
    ├── sample_0000_original.png
    ├── sample_0000_adversarial.png
    ├── sample_0000_perturbed.png
    ├── sample_0000_saliency.png
    ├── sample_0000_difference.png
    └── [more samples...]
```

## 🚀 Quick Start

1. **Test on demo images**:
   ```bash
   python test_image_inference.py --image-dir demo --params runs/train_cord_20250909_091350/universal.pt
   ```

2. **Evaluate on CORD dataset**:
   ```bash
   python test_dataset_evaluation.py --data data/cord --dataset cord --params runs/train_cord_20250909_091350/universal.pt
   ```

3. **Run all examples**:
   ```bash
   python run_tests_example.py
   ```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
   ```bash
   --device cpu --batch-size 2
   ```

2. **No images found**: Check file paths and supported formats (PNG, JPG, PDF)

3. **OCR fails**: Ensure EasyOCR is installed or use custom text
   ```bash
   --custom-text "Your text here"
   ```

4. **Dataset not found**: Check dataset structure or use simple image directory mode

### Performance Tips

- Use GPU for faster processing: `--device cuda`
- Reduce max samples for quick testing: `--max-samples 10`
- Disable sample saving for large batches: remove `--save-samples`

## 📊 Analysis and Interpretation

### Good Attack Results
- Alignment drop > 0.1
- Low LPIPS score (< 0.05)
- Fast processing time (< 3s)

### Visualization Interpretation
- **Saliency Map**: Shows which regions are most important for image-text alignment
- **Difference Map**: Shows where the adversarial perturbations are applied
- **Side-by-side**: Compare original vs adversarial images

### Statistical Analysis
- **Success Rate**: Overall attack effectiveness
- **Distribution Plots**: Understanding of attack consistency
- **Processing Time**: Performance benchmarking
