# DocSAF — Implementation Guide for Cursor (Claude Sonnet)

*A concise, “agent-friendly” plan to build **DocSAF: Universal Self-Supervised Semantic Attenuation Fields** for DocVLM attacks.*

> **Design philosophy:** One mechanism, two knobs. No multi-head complexity.
> **Core hypothesis:** Suppress the few, high-leverage regions that bind a document’s image to its own OCR text → collapse cross-modal grounding → high transfer, low detectability.
> **Tunables:** `alpha` (strength), `radius` (Gaussian blur scale). Everything else fixed.

---

## 0) Quick References (models / libs you’ll call)

* **CLIP / OpenCLIP** for image–text embeddings. ([GitHub][1], [PyPI][2], [Hugging Face][3])
* **BLIP-2** and **Donut** as diverse surrogates (OCR-free). ([Hugging Face][4])
* **Grad-CAM** for saliency (supports similarity and ViTs). ([GitHub][5])
* **OCR**: EasyOCR (light) or PaddleOCR (full). ([GitHub][6], [Jaided AI][7], [PyPI][8], [PaddlePaddle][9])
* **LPIPS** (perceptual distance), optional TorchMetrics LPIPS. ([GitHub][10], [Lightning AI][11])
* **Blur / EOT-light ops**: Kornia Gaussian blur; Albumentations JPEG compression; (optional) PyMuPDF to rasterize PDF test docs. ([kornia.readthedocs.io][12], [kornia.org][13], [explore.albumentations.ai][14], [vfdev-5-albumentations.readthedocs.io][15], [pymupdf.readthedocs.io][16])

---

## 1) Repository Layout (create exactly this)

```
docsaf/
  README.md
  pyproject.toml
  src/docsaf/
    __init__.py
    ocr.py               # EasyOCR / PaddleOCR wrappers
    surrogates.py        # CLIP / BLIP-2 / Donut embedders
    saliency.py          # cross-modal saliency (Grad-CAM / gradients)
    field.py             # A_theta(x): attenuation field and apply()
    objective.py         # alignment collapse + TV regularizer
    train_universal.py   # optimize alpha, radius on a corpus
    infer_once.py        # one-pass attack on a single document
    eval_harness.py      # ASR / Transfer-ASR / Defense-ASR / LPIPS
    eot_light.py         # JPEG/resize transforms (tiny set)
    pdf_io.py            # optional: PDF->image via PyMuPDF
    utils.py
  configs/
    default.yaml
    small.yaml
  tests/
    test_field.py
    test_saliency.py
  scripts/
    download_surrogates.sh
    demo_notebook.ipynb
```

---

## 2) Cursor “Context Engineering” (paste this into Cursor → New Instruction)

**System**
“You are Claude-sonnet writing minimal, production-ready PyTorch. Enforce: two tunables (alpha, radius), typed functions, short modules, testable units. Prefer OpenCLIP + HF Transformers. Cite well-known libs in comments but keep code self-contained.”

**Style**

* Prefer pure functions, no global state.
* Fail closed: raise `ValueError` with actionable messages.
* Write 10–20 line units; factor aggressively.
* Keep default seeds and device management in helpers.

**Definition of Done**

* `train_universal.py` trains **only** `{alpha, radius}` on a small corpus (folder of PNG/JPG/PDF).
* `infer_once.py` runs attack in one pass, writes `*_adv.png`.
* `eval_harness.py` computes ASR, Transfer-ASR, Defense-ASR, LPIPS.
* Unit tests pass: `pytest -q`.

---

## 3) Setup (agent should run these)

```bash
# Python
uv venv && source .venv/bin/activate  # or micromamba/conda

# Core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch transformers accelerate sentence-transformers
pip install easyocr paddleocr "paddleocr[all]"  # choose one at runtime
pip install kornia albumentations pymupdf
pip install lpips torchmetrics grad-cam  # LPIPS + Grad-CAM
pip install pytest rich typer==0.12.3
```

* OpenCLIP for embeddings; HF for BLIP-2/Donut; LPIPS as perceptual metric; Kornia/Albumentations for blur/JPEG; PyMuPDF for PDF raster. ([PyPI][2], [Hugging Face][4], [GitHub][10], [kornia.readthedocs.io][12], [explore.albumentations.ai][14], [pymupdf.readthedocs.io][16])

---

## 4) Minimal Config (`configs/default.yaml`)

```yaml
# two knobs
alpha: 1.2         # field strength
radius: 7          # Gaussian kernel radius (px at ~300dpi)

# saliency
saliency: "gradients"   # ["gradients","gradcam"]
saliency_target: "clip" # CLIP text-image cosine

# surrogates
surrogates:
  - "openclip:ViT-L-14@336"
  - "hf:blip2-opt-2.7b"       # lightweight variant or 2.7b on CPU offload
  - "hf:naver-clova-donut-base"

# ocr
ocr: "easyocr"   # or "paddleocr"

# training
steps: 15000
batch_size: 8
lr: 0.01
tv_lambda: 1.0e-3

# EOT-light
jpeg_q_min: 50
jpeg_q_max: 90
resize_min: 0.9
resize_max: 1.1
eot_prob: 0.5
```

---

## 5) Implementation Steps (agent tasks with acceptance criteria)

### 5.1 Surrogates — `src/docsaf/surrogates.py`

**Goal:** expose a uniform API for image/text embeddings.

```python
class ImageTextEmbedder(Protocol):
    def image_embed(self, image: torch.Tensor) -> torch.Tensor: ...
    def text_embed(self, texts: list[str]) -> torch.Tensor: ...

def load_clip(name: str="ViT-L-14-336") -> ImageTextEmbedder: ...
def load_blip2() -> ImageTextEmbedder: ...
def load_donut_text() -> ImageTextEmbedder: ...  # use Donut tokenizer/encoder for text side if needed
```

* CLIP / OpenCLIP image & text encoders: cosine similarity is our **alignment**. ([GitHub][1], [PyPI][2])
* BLIP-2 & Donut: include for capability diversity (OCR-free). ([Hugging Face][4])

**AC:** `pytest` creates dummy tensors and checks `(img_emb @ txt_emb.T).shape == (B, T)`.

---

### 5.2 OCR wrapper — `src/docsaf/ocr.py`

* Implement `ocr_read(image: np.ndarray, lang="en") -> str`.
* Start with **EasyOCR** for a light footprint; allow PaddleOCR as an alternate backend. ([GitHub][6], [PaddlePaddle][9])

**AC:** Running on a sample invoice image returns non-empty text.

---

### 5.3 Cross-modal saliency — `src/docsaf/saliency.py`

Two routes:

**A) Gradient saliency on CLIP alignment (simple & fast):**

1. Require normalized embeddings.
2. Compute `cos( img_emb(x), txt_emb(ocr_text) )`.
3. Backprop w\.r.t. input; take `abs(grad).mean(dim=1)` → normalize to `[0,1]`.

**B) Grad-CAM variant (if using ViT/CNN backbones):** use `pytorch-grad-cam`’s image-similarity support; average CAMs over surrogates. ([GitHub][5])

**AC:** `S(x)` returns `H×W` map in `[0,1]`; unit test checks monotone scaling under blurred vs sharp text crops.

---

### 5.4 Attenuation field — `src/docsaf/field.py`

**Definition:**

* Build `Aθ(x) = σ( alpha * ( Gaussian(radius) * S(x) ) ** gamma )` with `gamma=1.0`.
* Apply as **local defocus blend**:

  ```
  x_blur = kornia.filters.gaussian_blur2d(x, (k,k), (σ,σ))
  x_adv  = (1 - A) * x + A * x_blur
  ```

Use Kornia for differentiable blur. ([kornia.readthedocs.io][12])

**AC:** Boundary cases `alpha=0` ⇒ `x_adv==x`; increasing `alpha` increases LPIPS monotonically (checked on two images). ([GitHub][10])

---

### 5.5 Objective — `src/docsaf/objective.py`

**Self-supervised alignment collapse** (no labels):

$$
\min_{\alpha,r} \; \frac{1}{|\mathcal{M}|}\textstyle\sum_{m\in \mathcal{M}}
\cos\!\big(g_m^{img}(x'), g_m^{txt}(t)\big) + \lambda_{TV}\,TV(A_\theta)
$$

* `x' = apply_field(x, S(x); alpha, radius)`
* `t = OCR(x)`
* TV loss via finite differences; `tv_lambda = 1e-3` fixed.

**Targeted (optional):**

$$
\min\; \text{Align}(x', t) - \text{Align}(x', t^\star) + \lambda_{TV}TV(A_\theta)
$$

**AC:** Loss decreases on a tiny toy set (5–10 docs).

---

### 5.6 EOT-light — `src/docsaf/eot_light.py`

* Random JPEG compression (Albumentations `ImageCompression`) and mild resize jitter (Torchvision/Albumentations). Keep p=0.5. ([explore.albumentations.ai][14], [vfdev-5-albumentations.readthedocs.io][15])

**AC:** Wrapped `augment(x)` returns same shape tensor; JPEG quality varies in logs.

---

### 5.7 Universal training — `src/docsaf/train_universal.py`

* Optimize only `{alpha, radius}` (treat `radius` as positive via softplus).

* Loop:

  1. Load batch of images (rasterize PDFs via PyMuPDF if needed). ([pymupdf.readthedocs.io][16])
  2. `t = ocr_read(x)`
  3. `S = saliency(x, t)`
  4. `x' = apply_field(x, S; alpha, radius)`
  5. `x'' = eot_light(x')`
  6. Compute ensemble alignment; add TV.
  7. Adam step on `{alpha, radius}`.

* Log LPIPS vs original; cap median LPIPS ≤ 0.06. ([GitHub][10])

**AC:** Saves `universal.pt` with `{"alpha": float, "radius": float}` and a short JSON report of loss curve.

---

### 5.8 One-pass inference — `src/docsaf/infer_once.py`

**Inputs:** image/PDF path, optional `t_star`.
**Steps:** OCR→saliency→apply field with saved `{alpha, radius}`→save PNG.
**AC:** Writes `{name}_adv.png` and prints clean vs adv alignment scores.

---

### 5.9 Evaluation harness — `src/docsaf/eval_harness.py`

* **ASR** (untargeted): fraction of prompts where answer/extraction degrades (proxy: drop in alignment or wrong span from surrogate VQA/KIE; keep simple).
* **Transfer-ASR:** run held-out surrogate(s) (e.g., different BLIP-2/CLIP variant).
* **Defense-ASR:** re-JPEG, resize, re-OCR and re-score.
* **LPIPS** via `lpips.LPIPS(net="vgg")` or TorchMetrics. ([GitHub][10], [Lightning AI][11])

**AC:** Produces a small Markdown report with metrics tables.

---

## 6) Core Snippets (drop into files)

**Alignment + gradient saliency (CLIP/OpenCLIP)**

```python
# src/docsaf/saliency.py
def clip_alignment_and_saliency(model, preprocess, x_img: PIL.Image.Image, text: str, device="cuda"):
    import torch, open_clip
    x = preprocess(x_img).unsqueeze(0).to(device).requires_grad_(True)
    tok = open_clip.tokenize([text]).to(device)

    with torch.no_grad():
        img_f = model.encode_image(x)
        txt_f = model.encode_text(tok)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

    sim = (img_f * txt_f).sum()                       # cosine due to normalization
    sim.backward()                                    # d sim / d x
    g = x.grad.detach().abs().mean(1, keepdim=True)   # (1,1,H,W)
    S = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return sim.item(), S                              # scalar, [0..1]
```

(OpenCLIP / CLIP usage is standard; cosine similarity between normalized image/text embeddings.) ([PyPI][2], [GitHub][1])

**Field application (Kornia)**

```python
# src/docsaf/field.py
def apply_field(x: torch.Tensor, S: torch.Tensor, alpha: float, radius: float) -> torch.Tensor:
    import torch, kornia
    # x: (B,3,H,W) in [0,1], S: (B,1,H,W)
    k = int(max(3, 2*int(radius)+1))
    sigma = max(0.5, radius/3.0)
    x_blur = kornia.filters.gaussian_blur2d(x, (k,k), (sigma, sigma))
    A = torch.sigmoid(alpha * S)                      # gamma=1.0 fixed
    return x*(1-A) + x_blur*A
```

(Kornia’s `gaussian_blur2d` is differentiable and documented.) ([kornia.readthedocs.io][12])

**EOT-light (JPEG + resize)**

```python
# src/docsaf/eot_light.py
def eot_light(x: np.ndarray, jpeg_q_min=50, jpeg_q_max=90, resize_min=0.9, resize_max=1.1, p=0.5):
    import albumentations as A
    import numpy as np
    h, w = x.shape[:2]
    aug = A.Compose([
        A.ImageCompression(quality_lower=jpeg_q_min, quality_upper=jpeg_q_max, p=p),
        A.Resize(int(h*np.random.uniform(resize_min, resize_max)),
                 int(w*np.random.uniform(resize_min, resize_max)), p=p),
        A.Resize(h, w, p=1.0)
    ])
    return aug(image=x)["image"]
```

(Albumentations `ImageCompression` covers JPEG/WebP; we only need JPEG.) ([explore.albumentations.ai][14])

**PDF raster (optional)**

```python
# src/docsaf/pdf_io.py
def pdf_page_to_rgb(path:str, page:int=0, zoom:float=2.0):
    import fitz  # PyMuPDF
    doc = fitz.open(path); p = doc[page]
    pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    import numpy as np
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2]==4: img = img[...,:3]
    return img
```

(PyMuPDF official docs describe page rendering and image extraction.) ([pymupdf.readthedocs.io][16])

---

## 7) Vibe Coding: Step-by-step plan for the agent

1. **Bootstrap**

   * Scaffold repo & files; write `pyproject.toml`.
   * Add `download_surrogates.sh` (HF models cache).
   * Create `default.yaml`.

2. **Thin vertical slice**

   * Implement CLIP embedder only + EasyOCR.
   * Implement gradient saliency + field apply.
   * Hard-code `alpha=1.2`, `radius=7`.
   * Create `infer_once.py` that reads one PNG → `*_adv.png`.

3. **Objective & training**

   * Add BLIP-2 + Donut loaders (CPU offload if needed). ([Hugging Face][4])
   * Implement TV loss and optimizer over `{alpha, radius}`.
   * Train on a tiny folder (`data/sample_docs/`) for 1–2k steps.

4. **Evaluation harness**

   * Implement ASR proxy (drop in cosine alignment; for Donut, exact string mismatch on a small prompt). ([Hugging Face][17])
   * Add Defense-ASR via `eot_light`. ([explore.albumentations.ai][14])
   * Add LPIPS report. ([GitHub][10])

5. **Polish**

   * CLI via `typer` for train/infer/eval.
   * Write tests for saliency monotonicity & field edge cases.

---

## 8) Prompts the agent should keep in memory (Cursor “Notes”)

* “Only two hyperparameters are tunable: `alpha`, `radius`. Do not add more loss weights.”
* “Prefer **gradient saliency** first; enable **Grad-CAM** as optional.” ([GitHub][5])
* “Use **OpenCLIP** by default; add BLIP-2/Donut to diversify surrogates.” ([PyPI][2], [Hugging Face][4])
* “OCR defaults to **EasyOCR**; PaddleOCR is a config switch.” ([GitHub][6], [PaddlePaddle][9])
* “Keep EOT minimal: JPEG + slight resize only.” ([explore.albumentations.ai][14])
* “Report LPIPS and ensure median ≤ 0.06 unless user overrides.” ([GitHub][10])

---

## 9) Example CLI

```bash
# Inference (single image)
python -m src.docsaf.infer_once --image demo/invoice.png --out out/invoice_adv.png --config configs/default.yaml

# Train universal params on a folder
python -m src.docsaf.train_universal --data data/train_docs --out runs/universal.pt --config configs/default.yaml

# Evaluate (ASR / Transfer-ASR / Defense-ASR / LPIPS)
python -m src.docsaf.eval_harness --data data/test_docs --report out/report.md --config configs/default.yaml
```

---

## 10) Test Plan (quick)

* **Unit:**

  * `test_field.py`: `alpha=0` => identical image; monotone LPIPS for alpha ∈ {0.5,1.0,1.5}. ([GitHub][10])
  * `test_saliency.py`: blur a region → saliency decreases there.

* **Smoke:**

  * One pass creates `*_adv.png`; print clean vs adv cosine for CLIP. ([GitHub][1])

* **Regression:**

  * After universal training (≤ 30 min small run), ASR proxy improves ≥ +20% vs identity.

---

## 11) Evaluation Notes (for later large runs)

* **Surrogates**: {OpenCLIP ViT-L/336, BLIP-2 small, Donut-base}. ([PyPI][2], [Hugging Face][4])
* **Held-out**: swap to a different OpenCLIP or larger BLIP-2 checkpoint to measure Transfer-ASR. ([Hugging Face][18])
* **Defenses**: JPEG (Q∈\[50,90]), resize ±10%, re-OCR. ([explore.albumentations.ai][14])
* **Perceptibility**: LPIPS + brief human pairwise check (<5% noticed). ([GitHub][10])

---

## 12) Non-Goals (explicit)

* No multi-head (glyph/layout) edits.
* No heavy EOT stacks.
* No per-document optimization by default (universal is primary path).

---

## 13) Security & Ethics Footnote (keep in README)

* Use only on consented or public benchmark data.
* Disclose vulnerabilities responsibly; never attack production systems.

---

### Appendix A — Short rationale for chosen libs (citations)

* **CLIP / OpenCLIP**: standard image–text encoders; easy cosine alignment surface; many checkpoints. ([GitHub][1], [PyPI][2])
* **BLIP-2 / Donut**: give OCR-free and instruction bridging behavior for better cross-family transfer. ([Hugging Face][4]) ([Hugging Face][17])
* **Grad-CAM**: well-maintained repo supporting similarity and ViTs. ([GitHub][5])
* **EasyOCR / PaddleOCR**: practical, plug-and-play OCR backends. ([GitHub][6], [PaddlePaddle][9])
* **LPIPS**: perceptual distance aligned with human judgments; simple PyTorch API. ([GitHub][10])
* **Kornia**: differentiable Gaussian blur; ideal for our smooth field. ([kornia.readthedocs.io][12])
* **Albumentations**: JPEG compression transform for light EOT. ([explore.albumentations.ai][14])
* **PyMuPDF**: robust PDF rasterization for doc images. ([pymupdf.readthedocs.io][16])