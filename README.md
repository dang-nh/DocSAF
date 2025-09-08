# DocSAF: Universal Self-Supervised Semantic Attenuation Fields

**Phase 0 Vertical Slice** - Turn document images into adversarial images using the DocSAF attenuation field with only **two knobs**: `alpha` and `radius`.

## 🚀 Quickstart

### Install Dependencies

```bash
pip install -e .
```

### Run Phase 0 Demo

```bash
# Put your document image in demo/
python -m src.docsaf.infer_once --image demo/your_doc.png --config configs/default.yaml
```

**Expected output:**
```
Original alignment: 0.7834
Adversarial alignment: 0.6123  
Alignment drop: 0.1711
Success: Adversarial image saved!
```

The adversarial image will be saved as `demo/your_doc_adv.png` with a measurable cosine similarity drop.

### Interactive Demo

Launch the Jupyter notebook for step-by-step exploration:

```bash
jupyter notebook scripts/demo_notebook.ipynb
```

## 🎯 Two Knobs Only

DocSAF Phase 0 uses exactly **two tunable parameters**:

- **`alpha`** (1.2): Attenuation field strength 
- **`radius`** (7): Gaussian blur kernel radius

These are configured in `configs/default.yaml`. No trainable loss weights needed.

## 🏗️ Architecture

**Pipeline:**
1. **Load** document image or PDF page
2. **OCR** text extraction (EasyOCR)  
3. **Saliency** cross-modal gradients (OpenCLIP)
4. **Field** attenuation: `A = σ(α·S)`, `x' = (1-A)⊙x + A⊙blur(x)`
5. **EOT** light transforms (JPEG + resize)

**Core equation:**
```
x_adv = (1 - sigmoid(alpha * S)) ⊙ x + sigmoid(alpha * S) ⊙ blur(x, radius)
```

## 📁 Project Structure

```
src/docsaf/
├── surrogates.py    # OpenCLIP wrapper
├── ocr.py          # EasyOCR interface  
├── saliency.py     # Gradient saliency
├── field.py        # Attenuation field
├── eot_light.py    # JPEG + resize
├── pdf_io.py       # PDF → RGB
└── infer_once.py   # End-to-end CLI
```

## 🧪 Testing

```bash
pytest -q
```

**Expected:** All tests pass, confirming:
- `alpha=0` produces identity transformation
- LPIPS increases monotonically with `alpha`  
- Saliency decreases in blurred text regions