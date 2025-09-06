"""Surrogate model embedders with unified API."""

from typing import Protocol, List
import torch
import torch.nn.functional as F
from PIL import Image


class ImageTextEmbedder(Protocol):
    """Unified interface for image-text embedders."""

    def image_embed(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to embedding vector.

        Args:
            image: (B, 3, H, W) normalized tensor

        Returns:
            (B, D) embedding tensor
        """
        ...

    def text_embed(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings to embedding vectors.

        Args:
            texts: List of text strings

        Returns:
            (len(texts), D) embedding tensor
        """
        ...


class CLIPEmbedder:
    """OpenCLIP wrapper implementing ImageTextEmbedder protocol."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion400m_e32",
        device: str = "cuda",
    ):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def image_embed(self, image: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed image tensor."""
        with torch.no_grad():
            features = self.model.encode_image(image.to(self.device))
            return F.normalize(features, dim=-1)

    def text_embed(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            return F.normalize(features, dim=-1)

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input."""
        return self.preprocess(pil_image).unsqueeze(0)


class BLIP2Embedder:
    """BLIP-2 wrapper for diversity (CPU offload supported)."""

    def __init__(
        self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"
    ):
        from transformers import Blip2Processor, Blip2Model

        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16)

        if device == "cuda" and torch.cuda.is_available():
            self.model.to(device)
        else:
            self.model.to("cpu")

        self.model.eval()

    def image_embed(self, image: torch.Tensor) -> torch.Tensor:
        """Extract image features from BLIP-2."""
        # Convert tensor back to PIL for processor
        if image.dim() == 4:
            image = image.squeeze(0)

        # Tensor to PIL conversion
        import torchvision.transforms as T

        to_pil = T.ToPILImage()
        pil_image = to_pil(image.cpu())

        inputs = self.processor(images=pil_image, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            return F.normalize(image_features, dim=-1)

    def text_embed(self, texts: List[str]) -> torch.Tensor:
        """Extract text features from BLIP-2."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            return F.normalize(text_features, dim=-1)


def load_clip_embedder(
    name: str = "ViT-L-14@336", device: str = "cuda"
) -> CLIPEmbedder:
    """Load OpenCLIP embedder with specified architecture.

    Args:
        name: Model architecture (e.g., "ViT-L-14@336")
        device: Target device

    Returns:
        CLIPEmbedder instance
    """
    # Parse name format "ViT-L-14@336" -> model="ViT-L-14", pretrained="laion400m_e32"
    if "@" in name:
        model_arch = name.split("@")[0]
        # Use standard pretrained weights
        pretrained = "laion400m_e32"
    else:
        model_arch = name
        pretrained = "laion400m_e32"

    return CLIPEmbedder(model_arch, pretrained, device)


def load_blip2_embedder(
    name: str = "blip2-opt-2.7b", device: str = "cuda"
) -> BLIP2Embedder:
    """Load BLIP-2 embedder.

    Args:
        name: BLIP-2 variant
        device: Target device

    Returns:
        BLIP2Embedder instance
    """
    full_name = f"Salesforce/{name}"
    return BLIP2Embedder(full_name, device)


def load_embedder(spec: str, device: str = "cuda") -> ImageTextEmbedder:
    """Load embedder from spec string.

    Args:
        spec: Format "openclip:ViT-L-14@336" or "hf:blip2-opt-2.7b"
        device: Target device

    Returns:
        ImageTextEmbedder instance

    Raises:
        ValueError: If spec format is invalid
    """
    if not ":" in spec:
        raise ValueError(
            f"Invalid embedder spec: {spec}. Use 'openclip:model' or 'hf:model'"
        )

    backend, model_name = spec.split(":", 1)

    if backend == "openclip":
        return load_clip_embedder(model_name, device)
    elif backend == "hf" and "blip2" in model_name:
        return load_blip2_embedder(model_name, device)
    else:
        raise ValueError(
            f"Unsupported embedder backend: {backend} with model: {model_name}"
        )
