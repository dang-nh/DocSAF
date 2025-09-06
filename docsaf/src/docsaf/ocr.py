"""OCR wrappers with unified interface."""

import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def ocr_read(image: np.ndarray, lang: str = "en", backend: str = "easyocr") -> str:
    """Extract text from image using specified OCR backend.

    Args:
        image: RGB image array (H, W, 3) in uint8
        lang: Language code(s) for OCR
        backend: OCR backend ("easyocr" or "paddleocr")

    Returns:
        Extracted text string

    Raises:
        ValueError: If backend is unsupported
        ImportError: If required OCR library not installed
    """
    if backend == "easyocr":
        return _easyocr_read(image, lang)
    elif backend == "paddleocr":
        return _paddleocr_read(image, lang)
    else:
        raise ValueError(
            f"Unsupported OCR backend: {backend}. Use 'easyocr' or 'paddleocr'"
        )


def _easyocr_read(image: np.ndarray, lang: str = "en") -> str:
    """EasyOCR implementation."""
    try:
        import easyocr
    except ImportError:
        raise ImportError("EasyOCR not installed. Run: pip install easyocr")

    # Initialize reader (cached by EasyOCR internally)
    reader = easyocr.Reader([lang], gpu=True)

    try:
        results = reader.readtext(image, detail=0)  # detail=0 returns only text
        return " ".join(results)
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}. Returning empty string.")
        return ""


def _paddleocr_read(image: np.ndarray, lang: str = "en") -> str:
    """PaddleOCR implementation."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("PaddleOCR not installed. Run: pip install paddleocr")

    # Language mapping for PaddleOCR
    lang_map = {"en": "en", "ch": "ch", "chinese": "ch"}
    paddle_lang = lang_map.get(lang, "en")

    # Initialize OCR (use_gpu=True if available)
    ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, use_gpu=True)

    try:
        results = ocr.ocr(image, cls=True)

        # Extract text from PaddleOCR results
        text_lines = []
        if results and results[0]:
            for line in results[0]:
                if len(line) >= 2 and line[1]:
                    text_lines.append(line[1][0])  # Extract text string

        return " ".join(text_lines)
    except Exception as e:
        logger.warning(f"PaddleOCR failed: {e}. Returning empty string.")
        return ""


def ocr_confidence(
    image: np.ndarray, lang: str = "en", backend: str = "easyocr"
) -> List[Tuple[str, float]]:
    """Extract text with confidence scores.

    Args:
        image: RGB image array (H, W, 3) in uint8
        lang: Language code
        backend: OCR backend

    Returns:
        List of (text, confidence) tuples
    """
    if backend == "easyocr":
        return _easyocr_confidence(image, lang)
    elif backend == "paddleocr":
        return _paddleocr_confidence(image, lang)
    else:
        raise ValueError(f"Unsupported OCR backend: {backend}")


def _easyocr_confidence(image: np.ndarray, lang: str = "en") -> List[Tuple[str, float]]:
    """EasyOCR with confidence scores."""
    try:
        import easyocr
    except ImportError:
        raise ImportError("EasyOCR not installed. Run: pip install easyocr")

    reader = easyocr.Reader([lang], gpu=True)

    try:
        results = reader.readtext(
            image, detail=1
        )  # detail=1 returns bbox, text, confidence
        return [(item[1], item[2]) for item in results]
    except Exception as e:
        logger.warning(f"EasyOCR confidence extraction failed: {e}")
        return []


def _paddleocr_confidence(
    image: np.ndarray, lang: str = "en"
) -> List[Tuple[str, float]]:
    """PaddleOCR with confidence scores."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("PaddleOCR not installed. Run: pip install paddleocr")

    lang_map = {"en": "en", "ch": "ch", "chinese": "ch"}
    paddle_lang = lang_map.get(lang, "en")

    ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, use_gpu=True)

    try:
        results = ocr.ocr(image, cls=True)

        text_confidence = []
        if results and results[0]:
            for line in results[0]:
                if len(line) >= 2 and line[1]:
                    text, confidence = line[1]
                    text_confidence.append((text, confidence))

        return text_confidence
    except Exception as e:
        logger.warning(f"PaddleOCR confidence extraction failed: {e}")
        return []
