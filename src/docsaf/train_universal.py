"""Universal parameter training for DocSAF."""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm
import os
import sys

from torch.nn import functional as F
# Add data directory to path for importing utils
# current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# data_dir = os.path.join(current_dir, 'data')
# sys.path.insert(0, data_dir)

from .utils.utils import (
    load_config,
    setup_logging,
    get_device,
    set_random_seeds,
    pil_to_tensor,
    get_image_files,
    ensure_dir,
    compute_lpips_score,
)
from .surrogates import load_embedder
from .ocr import ocr_read
from .saliency import compute_gradient_saliency
from .objective import DocSAFObjective, validate_objective_inputs
from .pdf_io import pdf_to_pil, is_pdf_file
from .field import apply_field

# Import dataset readers from data/utils.py
# try:
from .utils.data import read_examples_func, load_image, read_docvqa_jsonl_file
# except ImportError:
#     print("Warning: Could not import dataset utils. Falling back to simple image loading.")

logger = logging.getLogger(__name__)


class StructuredDocumentDataset(Dataset):
    """Dataset for structured document datasets (FUNSD, CORD, SROIE, DocVQA, etc.)."""

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        split: str = "train",
        device: str = "cuda",
        max_size: tuple = (1024, 1024),
        line_level: bool = True,
        is_normalize: bool = True,
        use_precomputed: bool = True,
    ):
        """Initialize structured dataset.

        Args:
            data_dir: Path to dataset directory
            dataset_name: Dataset name (funsd, cord, sroie, docvqa, doclaynet)
            split: Split name (train, val, test)
            device: Target device for tensors
            max_size: Maximum image size (W, H)
            line_level: Use line-level bounding boxes
            is_normalize: Normalize bounding boxes
            use_precomputed: Use precomputed JSONL files for DocVQA
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.device = device
        self.max_size = max_size
        self.line_level = line_level
        self.is_normalize = is_normalize

        # Load dataset examples
        self.examples = self._load_examples(use_precomputed)
        
        logger.info(f"Initialized {dataset_name} dataset with {len(self.examples)} examples")

    def _load_examples(self, use_precomputed: bool = True):
        """Load examples using appropriate dataset reader."""
        try:
            if self.dataset_name == "docvqa" and use_precomputed:
                # Use precomputed JSONL files for DocVQA
                examples = read_docvqa_jsonl_file(
                    self.data_dir, self.split, self.line_level
                )
            elif self.dataset_name in read_examples_func:
                # Use dataset-specific readers
                reader_func = read_examples_func[self.dataset_name]
                if self.dataset_name == "docvqa":
                    examples = reader_func(
                        self.data_dir, self.split, self.line_level, self.is_normalize
                    )
                else:
                    examples, _ = reader_func(
                        self.data_dir, self.split, self.line_level, self.is_normalize
                    )
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load {self.dataset_name} dataset: {e}")
            return []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Get dataset item.

        Returns:
            Dict with 'image' tensor, 'text' string, 'path' string, 'words' list, 'bboxes' list
        """
        try:
            example = self.examples[idx]
            
            # Load image
            if "image" in example and example["image"] is not None:
                # Image already loaded as PIL
                pil_image = example["image"]
            else:
                # Load from path
                image_path = example["image_path"]
                if self.dataset_name == "docvqa":
                    # For DocVQA, load image from path
                    pil_image, _ = load_image(image_path)
                else:
                    pil_image = Image.open(image_path).convert("RGB")

            # Resize to consistent size for batching
            if self.max_size is not None:
                w, h = pil_image.size
                max_w, max_h = self.max_size
                # Always resize to ensure consistent batch dimensions
                scale = min(max_w / w, max_h / h)
                new_size = (int(w * scale), int(h * scale))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Ensure exact size by padding if necessary
                if new_size != self.max_size:
                    # Create a new image with exact max_size
                    new_image = Image.new('RGB', self.max_size, (255, 255, 255))
                    new_image.paste(pil_image, (0, 0))
                    pil_image = new_image

            # Convert to tensor
            image_tensor = pil_to_tensor(pil_image, self.device)

            # Prepare text
            if self.dataset_name == "docvqa":
                # For DocVQA, combine question and context
                question = example.get("question", "")
                words = example.get("words", [])
                context = " ".join(words[:512])  # Limit context length
                text = f"Question: {question} Context: {context}"
            else:
                # For other datasets, use OCR words
                words = example.get("words", [])
                text = " ".join(words) if words else "document content"

            # Fallback if no text
            if not text.strip():
                text = "document text content"
                logger.warning(f"No text found for example {idx}")

            return {
                "image": image_tensor.squeeze(0),  # (3, H, W)
                "text": text,
                "path": example.get("image_path", f"example_{idx}"),
                "words": words,
                "bboxes": example.get("bboxes", []),
                "labels": example.get("ner_tags", example.get("labels", [])),
            }

        except Exception as e:
            logger.error(f"Failed to load example {idx}: {e}")
            # Return dummy data to avoid crashing
            dummy_image = torch.zeros(3, 224, 224, device=self.device)
            return {
                "image": dummy_image,
                "text": "error loading document",
                "path": f"error_example_{idx}",
                "words": [],
                "bboxes": [],
                "labels": [],
            }


class DocumentDataset(Dataset):
    """Simple dataset for document images with OCR text extraction."""

    def __init__(
        self,
        image_paths: list,
        ocr_backend: str = "easyocr",
        device: str = "cuda",
        max_size: tuple = (1024, 1024),
    ):
        """Initialize dataset.

        Args:
            image_paths: List of image/PDF file paths
            ocr_backend: OCR backend to use
            device: Target device for tensors
            max_size: Maximum image size (W, H)
        """
        self.image_paths = image_paths
        self.ocr_backend = ocr_backend
        self.device = device
        self.max_size = max_size

        logger.info(f"Initialized simple dataset with {len(image_paths)} files")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get dataset item."""
        path = self.image_paths[idx]

        try:
            # Load image
            if is_pdf_file(path):
                pil_image = pdf_to_pil(path, page=0, zoom=2.0, max_size=self.max_size)
            else:
                pil_image = Image.open(path).convert("RGB")

                # Resize to consistent size for batching
                if self.max_size is not None:
                    w, h = pil_image.size
                    max_w, max_h = self.max_size
                    # Always resize to ensure consistent batch dimensions
                    scale = min(max_w / w, max_h / h)
                    new_size = (int(w * scale), int(h * scale))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Ensure exact size by padding if necessary
                    if new_size != self.max_size:
                        # Create a new image with exact max_size
                        new_image = Image.new('RGB', self.max_size, (255, 255, 255))
                        new_image.paste(pil_image, (0, 0))
                        pil_image = new_image

            # Convert to tensor
            image_tensor = pil_to_tensor(pil_image, self.device)

            # Extract OCR text
            img_array = np.array(pil_image)
            text = ocr_read(img_array, backend=self.ocr_backend)

            # Fallback if no text extracted
            if not text.strip():
                text = "document text content"
                logger.warning(f"No text extracted from {path}, using fallback")

            return {
                "image": image_tensor.squeeze(0),  # (3, H, W)
                "text": text,
                "path": str(path),
                "words": [],
                "bboxes": [],
                "labels": [],
            }

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            # Return dummy data to avoid crashing
            dummy_image = torch.zeros(3, 224, 224, device=self.device)
            return {
                "image": dummy_image,
                "text": "error loading document",
                "path": str(path),
                "words": [],
                "bboxes": [],
                "labels": [],
            }


def collate_documents(batch: list) -> dict:
    """Collate function for document batch."""
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    paths = [item["path"] for item in batch]
    words = [item["words"] for item in batch]
    bboxes = [item["bboxes"] for item in batch]
    labels = [item["labels"] for item in batch]

    return {
        "images": images, 
        "texts": texts, 
        "paths": paths,
        "words": words,
        "bboxes": bboxes, 
        "labels": labels
    }


def compute_batch_saliency(
    embedder, images: torch.Tensor, texts: list, saliency_method: str = "gradients"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute saliency maps for a batch.

    Args:
        embedder: Primary embedder for saliency computation
        images: Image batch (B, 3, H, W)
        texts: Text list
        saliency_method: Saliency computation method

    Returns:
        (alignments, saliency_maps) tensors
    """
    batch_size = images.shape[0]
    device = images.device

    alignments = torch.zeros(batch_size, device=device)
    saliency_maps = []

    for i in range(batch_size):
        # Single image with gradients enabled
        img = images[i : i + 1].requires_grad_(True)
        text = texts[i]

        # Compute saliency
        alignment, saliency = compute_gradient_saliency(
            embedder, img, text, normalize=True
        )

        alignments[i] = alignment
        saliency_maps.append(saliency.detach())

        # Clear gradients
        if img.grad is not None:
            img.grad.zero_()

    saliency_batch = torch.cat(saliency_maps, dim=0)  # (B, 1, H, W)

    return alignments, saliency_batch


def train_universal_parameters(
    data_dir: str, config: dict, output_dir: str, device: str = "cuda", 
    dataset_name: str = None, split: str = "train"
) -> dict:
    """Train universal alpha and radius parameters.

    Args:
        data_dir: Directory containing training documents
        config: Configuration dictionary
        output_dir: Output directory for results
        device: Computation device
        dataset_name: Name of structured dataset (funsd, cord, sroie, docvqa, doclaynet)
        split: Dataset split to use

    Returns:
        Training results dictionary
    """
    # Setup
    output_path = Path(output_dir)
    ensure_dir(output_path)

    # Training config
    train_config = config.get("training", {})
    steps = train_config.get("steps", 15000)
    batch_size = train_config.get("batch_size", 8)
    lr = train_config.get("lr", 0.01)
    tv_lambda = train_config.get("tv_lambda", 1e-3)

    # Dataset config
    dataset_config = config.get("dataset", {})
    line_level = dataset_config.get("line_level", True)
    is_normalize = dataset_config.get("is_normalize", True)
    max_size = tuple(dataset_config.get("max_size", [1024, 1024]))

    # Initialize parameters
    alpha = torch.tensor(config.get("alpha", 1.2), dtype=torch.float32, device=device, requires_grad=True)
    radius = torch.tensor(config.get("radius", 7.0), dtype=torch.float32, device=device, requires_grad=True)

    # Setup optimizer (only optimize alpha and radius)
    # optimizer = optim.Adam([alpha, radius], lr=lr)
    optimizer = optim.Adam(
                    [{'params':[alpha],  'lr':lr*3},
                    {'params':[radius], 'lr':lr}],
                    betas=(0.9,0.999)
                    )

    # Load dataset
    if dataset_name and dataset_name.lower() in ["funsd", "cord", "sroie", "docvqa", "doclaynet"]:
        # Use structured dataset
        logger.info(f"Loading structured dataset: {dataset_name}")
        dataset = StructuredDocumentDataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            device=device,
            max_size=max_size,
            line_level=line_level,
            is_normalize=is_normalize,
        )
    else:
        # Use simple image dataset
        logger.info("Loading simple image dataset")
        image_files = get_image_files(data_dir)
        if not image_files:
            raise ValueError(f"No image files found in {data_dir}")
        
        dataset = DocumentDataset(
            image_files, 
            ocr_backend=config.get("ocr", "easyocr"), 
            device=device,
            max_size=max_size
        )
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty. Check data_dir: {data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_documents,
        num_workers=0,  # Set to 0 for device tensors
    )

    # Setup objective
    surrogate_specs = config.get("surrogates", ["openclip:ViT-L-14@336"])
    objective = DocSAFObjective(surrogate_specs, tv_lambda=tv_lambda, device=device)

    # Primary embedder for saliency computation
    primary_embedder = objective.embedders[0]

    # EOT configuration
    eot_config = config.get("eot", {})
    eot_prob = eot_config.get("eot_prob", 0.0)
    if eot_prob > 0:
        eot_params = {k: v for k, v in eot_config.items() if k != "eot_prob"}
    else:
        eot_params = None

    # Training loop
    logger.info(f"Starting training for {steps} steps...")
    logger.info(f"Initial alpha={alpha.item():.3f}, radius={radius.item():.3f}")

    history = {
        "steps": [],
        "losses": [],
        "alpha_values": [],
        "radius_values": [],
        "alignment_losses": [],
        "tv_losses": [],
        "lpips_scores": [],
    }

    step = 0
    data_iter = iter(dataloader)

    progress_bar = tqdm(total=steps, desc="Training")

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # --- Saliency phase ---
        images_sal = batch["images"].detach().clone().requires_grad_(True)
        texts = batch["texts"]

        # compute saliency (will raise if graph is broken)
        _, saliency_maps = compute_batch_saliency(primary_embedder, images_sal, texts)
        saliency_maps = saliency_maps.detach()

        # --- Loss phase (no grads on images) ---
        images = batch["images"].detach()

        # OPTIONAL warmup
        warmup_steps = 200
        if step < warmup_steps:
            radius.requires_grad_(False)
        else:
            radius.requires_grad_(True)

        total_loss, loss_components = objective.compute_loss(
            alpha=alpha,
            radius=radius,
            images=images,
            texts=texts,
            saliency_maps=saliency_maps,
            eot_prob=0.0,       # force EOT off in training
            eot_config=None,
        )

        optimizer.zero_grad()
        total_loss.backward()

        # if radius was frozen, wipe accidental grads
        if not radius.requires_grad and radius.grad is not None:
            radius.grad.zero_()

        optimizer.step()

        # Sanity test block (run once after first batch)
        if step == 1:
            images = batch["images"].detach()
            texts = batch["texts"]
            images_sal = images.clone().requires_grad_(True)
            _, sal = compute_gradient_saliency(primary_embedder, images_sal, texts[0], normalize=True)
            sal = sal.detach()
            alpha.grad = None; radius.grad = None
            loss_dbg, _ = objective.compute_loss(alpha, radius, images.detach(), texts, sal, eot_prob=0.0)
            loss_dbg.backward()
            logger.info(f"[SANITY] alpha.grad={alpha.grad} radius.grad={radius.grad}")

        # Diagnostics
        if step % 20 == 0:
            from docsaf.saliency import saliency_stats
            stats = saliency_stats(saliency_maps)
            logger.info(f"saliency stats: min={stats['min']:.4g} max={stats['max']:.4g} std={stats['std']:.4g}")

            with torch.no_grad():
                x_adv_dbg, A_dbg = apply_field(images[:1], saliency_maps[:1], alpha, F.softplus(radius))
                logger.info(f"A stats: min={A_dbg.min().item():.4f} max={A_dbg.max().item():.4f} mean={A_dbg.mean().item():.4f}")

            logger.info(f"|grad alpha|={0.0 if alpha.grad is None else alpha.grad.abs().item():.3e} "
                        f"|grad radius|={0.0 if radius.grad is None else radius.grad.abs().item():.3e}")
                
        # Logging
        if step % 100 == 0:
            # Compute LPIPS on first sample
            try:
                with torch.no_grad():
                    x_adv_sample, A = apply_field(
                        images[:1],
                        saliency_maps[:1],
                        alpha,
                        torch.nn.functional.softplus(radius),
                    )
                    lpips_score = compute_lpips_score(
                        images[:1], x_adv_sample, device=device
                    )
            except:
                lpips_score = 0.0

            logger.info(
                f"Step {step:>5}: Loss={total_loss.item():.4f}, "
                f"Alpha={alpha.item():.3f}, Radius={torch.nn.functional.softplus(radius).item():.3f}, "
                f"LPIPS={lpips_score:.4f}"
            )

        # Record history
        history["steps"].append(step)
        history["losses"].append(float(total_loss.item()))
        history["alpha_values"].append(float(alpha.item()))
        history["radius_values"].append(
            float(torch.nn.functional.softplus(radius).item())
        )
        history["alignment_losses"].append(
            float(loss_components["alignment_loss"].item())
        )
        history["tv_losses"].append(float(loss_components["tv_loss"].item()))

        step += 1
        progress_bar.update(1)

    progress_bar.close()

    # Final parameters
    final_alpha = float(alpha.item())
    final_radius = float(torch.nn.functional.softplus(radius).item())

    logger.info(f"Training completed!")
    logger.info(f"Final parameters: alpha={final_alpha:.3f}, radius={final_radius:.3f}")

    # Save results
    results = {
        "alpha": final_alpha,
        "radius": final_radius,
        "training_config": train_config,
        "final_loss": float(total_loss.item()),
        "history": history,
    }

    # Save universal parameters
    params_path = output_path / "universal.pt"
    torch.save({"alpha": final_alpha, "radius": final_radius}, params_path)
    logger.info(f"Saved universal parameters to: {params_path}")

    # Save training report
    report_path = output_path / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved training report to: {report_path}")

    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="DocSAF universal parameter training")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data directory"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["funsd", "cord", "sroie", "docvqa", "doclaynet"], 
        help="Structured dataset name (if not specified, uses simple image loading)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (train/val/test)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output", type=str, default="runs", help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cuda/cpu)"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load config
    config = load_config(args.config)

    # Set random seed
    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    set_random_seeds(seed)
    logger.info(f"Random seed: {seed}")

    # Create timestamped output directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_suffix = f"_{args.dataset}" if args.dataset else ""
    output_dir = Path(args.output) / f"train{dataset_suffix}_{timestamp}"

    try:
        # Run training
        results = train_universal_parameters(
            data_dir=args.data, 
            config=config, 
            output_dir=str(output_dir), 
            device=device,
            dataset_name=args.dataset,
            split=args.split
        )

        print(f"\n=== Training Completed ===")
        print(f"Output directory: {output_dir}")
        print(f"Dataset: {args.dataset or 'simple images'}")
        print(f"Split: {args.split}")
        print(f"Final alpha: {results['alpha']:.3f}")
        print(f"Final radius: {results['radius']:.3f}")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Universal parameters saved to: {output_dir}/universal.pt")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
