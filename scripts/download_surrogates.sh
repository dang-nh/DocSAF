#!/bin/bash

# DocSAF Surrogate Model Download Script
# Downloads and caches OpenCLIP, BLIP-2, and Donut models for offline use

set -e

echo "=== DocSAF Surrogate Model Download ==="
echo "This script will download and cache surrogate models for DocSAF"
echo

# Check if conda environment is active
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Warning: No conda environment detected. Please activate the docsaf environment:"
    echo "conda activate docsaf"
    echo
fi

# Check if Python and required packages are available
echo "Checking dependencies..."
python3 -c "import torch, open_clip, transformers" 2>/dev/null || {
    echo "Error: Missing required packages. Please install DocSAF first:"
    echo "pip install -e ."
    exit 1
}

echo "✓ Dependencies OK"
echo

# Create cache directories
echo "Creating cache directories..."
mkdir -p ~/.cache/huggingface/hub
mkdir -p ~/.cache/open_clip
echo "✓ Cache directories created"
echo

# Download OpenCLIP models
echo "=== Downloading OpenCLIP Models ==="

echo "Downloading ViT-L-14@336 (primary)..."
python3 -c "
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e32')
print('✓ ViT-L-14 downloaded')
"

echo "Downloading ViT-B-32@336 (transfer eval)..."
python3 -c "
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
print('✓ ViT-B-32 downloaded')
"

echo "✓ OpenCLIP models downloaded"
echo

# Download BLIP-2 models (optional, can be large)
echo "=== BLIP-2 Models (Optional) ==="
read -p "Download BLIP-2 models? They are large (~15GB). [y/N]: " download_blip2

if [[ $download_blip2 =~ ^[Yy]$ ]]; then
    echo "Downloading BLIP-2 OPT-2.7B..."
    python3 -c "
from transformers import Blip2Processor, Blip2Model
processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
model = Blip2Model.from_pretrained('Salesforce/blip2-opt-2.7b')
print('✓ BLIP-2 OPT-2.7B downloaded')
"
else
    echo "⚠ BLIP-2 models skipped (can be downloaded later)"
fi
echo

# Download Donut models (optional)
echo "=== Donut Models (Optional) ==="
read -p "Download Donut models? [y/N]: " download_donut

if [[ $download_donut =~ ^[Yy]$ ]]; then
    echo "Downloading Donut base..."
    python3 -c "
from transformers import DonutProcessor, VisionEncoderDecoderModel
processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base')
model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base')
print('✓ Donut base downloaded')
"
else
    echo "⚠ Donut models skipped (can be downloaded later)"
fi
echo

# Download EasyOCR models
echo "=== OCR Models ==="
echo "Downloading EasyOCR models..."
python3 -c "
import easyocr
reader = easyocr.Reader(['en'], gpu=False)
print('✓ EasyOCR English model downloaded')
"

# Download LPIPS models
echo "Downloading LPIPS models..."
python3 -c "
import lpips
loss_fn = lpips.LPIPS(net='vgg')
print('✓ LPIPS VGG model downloaded')
"

echo
echo "=== Download Complete ==="
echo "All requested models have been downloaded and cached."
echo "Models are stored in:"
echo "  - OpenCLIP: ~/.cache/huggingface/hub/"
echo "  - HuggingFace: ~/.cache/huggingface/hub/"
echo "  - EasyOCR: ~/.EasyOCR/"
echo "  - LPIPS: site-packages/lpips/weights/"
echo
echo "You can now run DocSAF offline (if all models were downloaded)."
echo "To test the installation, run:"
echo "  python scripts/test_surrogates.py"
echo
