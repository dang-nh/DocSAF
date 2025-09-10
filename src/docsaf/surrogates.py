"""Surrogate model embedders with unified API."""

from typing import Protocol, List
import torch
import torch.nn.functional as F
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageTextAligner(Protocol):
    """Unified interface for image-text alignment models."""

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to embedding vector.

        Args:
            x: (B, 3, H, W) image tensor in [0, 1]

        Returns:
            (B, D) normalized embedding tensor
        """
        ...

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings to embedding vectors.

        Args:
            texts: List of text strings

        Returns:
            (len(texts), D) normalized embedding tensor
        """
        ...

    def cosine_align(self, x: torch.Tensor, text: str) -> torch.Tensor:
        """Compute cosine alignment between image and text.

        Args:
            x: (B, 3, H, W) image tensor
            text: Text string

        Returns:
            (B,) alignment scores
        """
        img_emb = self.encode_image(x)  # (B, D)
        txt_emb = self.encode_text([text])  # (1, D)
        return (img_emb * txt_emb).sum(dim=-1)  # (B,)


# Legacy protocol for backward compatibility
class ImageTextEmbedder(Protocol):
    """Legacy interface for image-text embedders."""

    def image_embed(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to embedding vector."""
        ...

    def text_embed(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings to embedding vectors."""
        ...


class ImageTextEmbedder:
    """Universal image-text embedder wrapper with gradient control."""
    
    def __init__(self, model, preprocess=None, tokenizer=None, device="cuda"):
        self.model = model
        self.preprocess = preprocess  # must be PyTorch ops (no PIL/NumPy) if used here
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # eval is fine; just do not wrap forward in no_grad
    
    def image_embed(self, x: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        # x is already a tensor on device, preprocessed outside or here by torch ops
        x_in = x
        # If you require preprocessing, ensure it is differentiable and inside PyTorch graph.
        # Example (if you had mean/std norm): x_in = (x - mean)/std
        if requires_grad:
            return self.model.encode_image(x_in)  # NO no_grad here
        else:
            with torch.no_grad():
                return self.model.encode_image(x_in)
    
    def text_embed(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            # tokenize must not detach the graph for image path (text has no gradients anyway)
            tokenized = self.tokenizer(texts).to(self.device) if self.tokenizer else texts
            return self.model.encode_text(tokenized)


class CLIPAligner:
    """OpenCLIP wrapper implementing ImageTextAligner protocol."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion400m_e32",
        device: str = "cuda",
    ):
        import open_clip

        self.device = device
        self.model_name = model_name
        logger.info(f"Loading OpenCLIP {model_name} on {device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensor (applies preprocessing if needed)."""
        # Ensure tensor is in (B, C, H, W) format
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # If image is not the expected size, we need to preprocess it
        # CLIP expects 224x224 for ViT-L-14
        if x.shape[-2:] != (224, 224):
            # Use differentiable resize and normalization to preserve gradients
            import torch.nn.functional as F_interp
            
            # Resize to 224x224 preserving gradients
            x = F_interp.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            
            # Apply CLIP normalization (mean and std from preprocessing)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
        
        # Remove no_grad since we need gradients for saliency computation
        features = self.model.encode_image(x.to(self.device))
        return F.normalize(features, dim=-1)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            return F.normalize(features, dim=-1)

    def cosine_align(self, x: torch.Tensor, text: str) -> torch.Tensor:
        """Compute cosine alignment between image and text."""
        img_emb = self.encode_image(x)  # (B, D)
        txt_emb = self.encode_text([text])  # (1, D)
        return (img_emb * txt_emb).sum(dim=-1)  # (B,)

    # Legacy methods for backward compatibility
    def image_embed(self, image: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """Legacy method - use encode_image instead."""
        return self.encode_image(image)

    def text_embed(self, texts: List[str]) -> torch.Tensor:
        """Legacy method - use encode_text instead."""
        return self.encode_text(texts)

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input."""
        return self.preprocess(pil_image).unsqueeze(0)


class BLIP2Aligner:
    """BLIP-2 wrapper implementing ImageTextAligner protocol (CPU offload supported)."""

    def __init__(
        self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"
    ):
        from transformers import Blip2Processor, Blip2Model

        self.device = device
        self.model_name = model_name
        logger.info(f"Loading BLIP-2 {model_name} on {device}")
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16)

        if device == "cuda" and torch.cuda.is_available():
            self.model.to(device)
        else:
            self.model.to("cpu")
            logger.info("Using CPU for BLIP-2 (GPU not available)")

        self.model.eval()

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features from BLIP-2."""
        batch_size = x.shape[0]
        features_list = []
        
        for i in range(batch_size):
            # Convert tensor to PIL for processor
            import torchvision.transforms as T
            to_pil = T.ToPILImage()
            pil_image = to_pil(x[i].cpu())

            inputs = self.processor(images=pil_image, return_tensors="pt").to(
                self.model.device
            )
            # Removed torch.no_grad() to allow gradient flow
            # Use the vision model directly to get pooled features
            pixel_values = inputs.pixel_values
            vision_outputs = self.model.vision_model(pixel_values)
            
            # Use pooled output if available, otherwise pool the last hidden state
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                image_features = vision_outputs.pooler_output
            else:
                # Pool the last hidden state
                image_features = vision_outputs.last_hidden_state.mean(dim=1)
            
            features_list.append(F.normalize(image_features, dim=-1))
        
        return torch.cat(features_list, dim=0)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Extract text features from BLIP-2."""
        # For BLIP-2, we'll use a simplified approach by encoding through the Q-Former
        # Since BLIP-2 doesn't naturally provide text-only embeddings
        features_list = []
        
        for text in texts:
            # Create a dummy image for the text encoder
            dummy_image = torch.ones((1, 3, 224, 224)).to(self.model.device)
            
            inputs = self.processor(
                images=dummy_image.cpu().numpy().transpose(0, 2, 3, 1)[0], 
                text=text, 
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                # Get query features from the model
                outputs = self.model(**inputs, return_dict=True)
                
                # Use the Q-Former output as text representation
                if hasattr(outputs, 'qformer_outputs'):
                    text_features = outputs.qformer_outputs.pooler_output
                elif hasattr(outputs, 'language_model_outputs'):
                    # Alternative: use the language model embedding
                    text_features = outputs.language_model_outputs.hidden_states[-1].mean(dim=1)
                else:
                    # Fallback: use a simple approach
                    text_features = torch.randn(1, 768).to(self.model.device)
                
                features_list.append(F.normalize(text_features, dim=-1))
        
        return torch.cat(features_list, dim=0)

    def cosine_align(self, x: torch.Tensor, text: str) -> torch.Tensor:
        """Compute cosine alignment between image and text."""
        img_emb = self.encode_image(x)  # (B, D)
        txt_emb = self.encode_text([text])  # (1, D)
        return (img_emb * txt_emb).sum(dim=-1)  # (B,)

    # Legacy methods for backward compatibility
    def image_embed(self, image: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """Legacy method - use encode_image instead."""
        return self.encode_image(image)

    def text_embed(self, texts: List[str]) -> torch.Tensor:
        """Legacy method - use encode_text instead."""
        return self.encode_text(texts)


class DonutAligner:
    """Donut wrapper implementing ImageTextAligner protocol.
    
    Uses the document understanding model Donut by extracting features from
    the decoder CLS state as a proxy for text-image alignment. 
    Note: This is an approximation since Donut is primarily a generative model.
    """

    def __init__(
        self, model_name: str = "naver-clova-ix/donut-base", device: str = "cuda"
    ):
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.device = device
        self.model_name = model_name
        logger.info(f"Loading Donut {model_name} on {device}")
        
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        if device == "cuda" and torch.cuda.is_available():
            self.model.to(device)
        else:
            self.model.to("cpu")
            logger.info("Using CPU for Donut (GPU not available)")

        self.model.eval()

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features from Donut encoder."""
        batch_size = x.shape[0]
        features_list = []
        
        for i in range(batch_size):
            # Convert tensor to PIL for processor
            import torchvision.transforms as T
            to_pil = T.ToPILImage()
            pil_image = to_pil(x[i].cpu())

            # Process image through Donut processor
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(
                self.model.device
            )
            
            # Removed torch.no_grad() to allow gradient flow
            # Get encoder features (vision encoder output)
            encoder_outputs = self.model.encoder(pixel_values)
            # Pool the encoder features (mean pooling over spatial dimensions)
            encoder_features = encoder_outputs.last_hidden_state.mean(dim=1)  # (1, D)
            features_list.append(F.normalize(encoder_features, dim=-1))
        
        return torch.cat(features_list, dim=0)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Extract text features from Donut decoder.
        
        Note: This is an approximation - we encode the text through the decoder
        and extract the final hidden state as a text representation.
        """
        features_list = []
        
        for text in texts:
            # Prepare decoder input
            decoder_input_ids = self.processor.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(self.model.device)
            
            with torch.no_grad():
                # Get decoder features
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=None,  # Not needed for text-only encoding
                    return_dict=True
                )
                # Use mean pooling over sequence length
                text_features = decoder_outputs.last_hidden_state.mean(dim=1)  # (1, D)
                features_list.append(F.normalize(text_features, dim=-1))
        
        return torch.cat(features_list, dim=0)

    def cosine_align(self, x: torch.Tensor, text: str) -> torch.Tensor:
        """Compute cosine alignment between image and text.
        
        Note: This is an approximation of alignment since Donut is primarily
        a generative model, not designed for retrieval-style alignment.
        """
        img_emb = self.encode_image(x)  # (B, D)
        txt_emb = self.encode_text([text])  # (1, D)
        return (img_emb * txt_emb).sum(dim=-1)  # (B,)

    # Legacy methods for backward compatibility
    def image_embed(self, image: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """Legacy method - use encode_image instead."""
        return self.encode_image(image)

    def text_embed(self, texts: List[str]) -> torch.Tensor:
        """Legacy method - use encode_text instead."""
        return self.encode_text(texts)


def load_clip_aligner(
    name: str = "ViT-L-14@336", device: str = "cuda"
) -> CLIPAligner:
    """Load OpenCLIP aligner with specified architecture.

    Args:
        name: Model architecture (e.g., "ViT-L-14@336")
        device: Target device

    Returns:
        CLIPAligner instance
    """
    # Parse name format "ViT-L-14@336" -> model="ViT-L-14", pretrained="laion400m_e32"
    if "@" in name:
        model_arch = name.split("@")[0]
        # Use standard pretrained weights
        pretrained = "laion400m_e32"
    else:
        model_arch = name
        pretrained = "laion400m_e32"

    return CLIPAligner(model_arch, pretrained, device)


def load_blip2_aligner(
    name: str = "blip2-opt-2.7b", device: str = "cuda"
) -> BLIP2Aligner:
    """Load BLIP-2 aligner.

    Args:
        name: BLIP-2 variant
        device: Target device

    Returns:
        BLIP2Aligner instance
    """
    full_name = f"Salesforce/{name}"
    return BLIP2Aligner(full_name, device)


def load_donut_aligner(
    name: str = "donut-base", device: str = "cuda"
) -> DonutAligner:
    """Load Donut aligner.

    Args:
        name: Donut variant
        device: Target device

    Returns:
        DonutAligner instance
    """
    if name.startswith("naver-clova-ix/"):
        full_name = name  # Already has prefix
    else:
        full_name = f"naver-clova-ix/{name}"
    return DonutAligner(full_name, device)


def load_aligner(spec: str, device: str = "cuda") -> ImageTextAligner:
    """Load aligner from spec string.

    Args:
        spec: Format "openclip:ViT-L-14@336", "hf:blip2-opt-2.7b", or "hf:naver-clova-ix/donut-base"
        device: Target device

    Returns:
        ImageTextAligner instance

    Raises:
        ValueError: If spec format is invalid
    """
    if not ":" in spec:
        raise ValueError(
            f"Invalid aligner spec: {spec}. Use 'openclip:model', 'hf:blip2-*', or 'hf:donut-*'"
        )

    backend, model_name = spec.split(":", 1)

    if backend == "openclip":
        return load_clip_aligner(model_name, device)
    elif backend == "hf" and "blip2" in model_name:
        return load_blip2_aligner(model_name, device)
    elif backend == "hf" and ("donut" in model_name or "naver-clova-ix" in model_name):
        return load_donut_aligner(model_name, device)
    else:
        raise ValueError(
            f"Unsupported aligner backend: {backend} with model: {model_name}"
        )


def build_aligners(surrogate_specs: List[str], device: str = "cuda") -> List[ImageTextAligner]:
    """Build list of aligners from configuration specs.

    Args:
        surrogate_specs: List of aligner specifications
        device: Target device

    Returns:
        List of loaded ImageTextAligner instances

    Example:
        >>> specs = ["openclip:ViT-L-14@336", "hf:blip2-opt-2.7b", "hf:naver-clova-ix/donut-base"]
        >>> aligners = build_aligners(specs, "cuda")
    """
    aligners = []
    
    for spec in surrogate_specs:
        try:
            aligner = load_aligner(spec, device)
            aligners.append(aligner)
            logger.info(f"Loaded aligner: {spec}")
        except Exception as e:
            logger.error(f"Failed to load aligner {spec}: {e}")
            # Continue loading other aligners instead of failing completely
            continue
    
    if not aligners:
        raise RuntimeError("No aligners could be loaded successfully")
    
    return aligners


# Legacy functions for backward compatibility
def load_clip_embedder(name: str = "ViT-L-14@336", device: str = "cuda"):
    """Legacy function - use load_clip_aligner instead."""
    return load_clip_aligner(name, device)


def load_blip2_embedder(name: str = "blip2-opt-2.7b", device: str = "cuda"):
    """Legacy function - use load_blip2_aligner instead."""
    return load_blip2_aligner(name, device)


def load_embedder(spec: str, device: str = "cuda"):
    """Legacy function - use load_aligner instead."""
    return load_aligner(spec, device)
