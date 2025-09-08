"""DocSAF: Universal Self-Supervised Semantic Attenuation Fields for Document VLM Attacks."""

__version__ = "0.1.0"
__author__ = "DocSAF Team"
__email__ = "team@docsaf.ai"

from .field import apply_field
from .saliency import compute_gradient_saliency
from .surrogates import load_clip_embedder

__all__ = [
    "apply_field",
    "compute_gradient_saliency", 
    "load_clip_embedder",
]
