"""DocSAF: Universal Self-Supervised Semantic Attenuation Fields for Document VLM Attacks."""

__version__ = "0.1.0"
__author__ = "DocSAF Team"
__email__ = "team@docsaf.ai"

from .field import apply_field
from .saliency import compute_gradient_saliency
from .surrogates import load_clip_embedder
from .ocr import ocr_read
from .objective import docsaf_objective, DocSAFObjective
from .utils.utils import load_config, setup_logging

__all__ = [
    "apply_field",
    "compute_gradient_saliency",
    "load_clip_embedder",
    "ocr_read",
    "docsaf_objective",
    "DocSAFObjective",
    "load_config",
    "setup_logging",
]
