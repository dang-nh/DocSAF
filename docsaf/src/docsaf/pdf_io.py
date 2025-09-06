"""PDF I/O utilities for DocSAF."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def pdf_page_to_rgb(
    path: Union[str, Path], 
    page: int = 0, 
    zoom: float = 2.0,
    max_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Convert PDF page to RGB image array.
    
    Args:
        path: Path to PDF file
        page: Page number (0-indexed)
        zoom: Zoom factor for rendering (higher = better quality)
        max_size: Optional (width, height) limit
        
    Returns:
        RGB image array (H, W, 3) in uint8 [0, 255]
        
    Raises:
        ImportError: If PyMuPDF not installed
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If page number is invalid
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    try:
        # Open PDF document
        doc = fitz.open(str(path))
        
        if page < 0 or page >= len(doc):
            raise ValueError(f"Invalid page number {page}. PDF has {len(doc)} pages.")
        
        # Get page
        pdf_page = doc[page]
        
        # Create transformation matrix
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page as pixmap
        pix = pdf_page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = pix.samples
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        
        # Handle different color channels
        if pix.n == 4:  # RGBA
            img = img[:, :, :3]  # Drop alpha channel
        elif pix.n == 1:  # Grayscale
            img = np.repeat(img, 3, axis=2)  # Convert to RGB
        
        # Apply size limit if specified
        if max_size is not None:
            img = _resize_if_needed(img, max_size)
        
        doc.close()
        return img
        
    except Exception as e:
        logger.error(f"Failed to convert PDF page: {e}")
        raise


def pdf_to_pil(
    path: Union[str, Path], 
    page: int = 0, 
    zoom: float = 2.0,
    max_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """Convert PDF page to PIL Image.
    
    Args:
        path: Path to PDF file
        page: Page number
        zoom: Zoom factor
        max_size: Optional size limit
        
    Returns:
        PIL Image in RGB mode
    """
    img_array = pdf_page_to_rgb(path, page, zoom, max_size)
    return Image.fromarray(img_array, mode='RGB')


def pdf_info(path: Union[str, Path]) -> dict:
    """Get PDF metadata and information.
    
    Args:
        path: Path to PDF file
        
    Returns:
        Dictionary with PDF information
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    try:
        doc = fitz.open(str(path))
        
        info = {
            "page_count": len(doc),
            "metadata": doc.metadata,
            "file_size": path.stat().st_size,
            "file_name": path.name,
        }
        
        # Get page sizes
        page_sizes = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            page_sizes.append((rect.width, rect.height))
        
        info["page_sizes"] = page_sizes
        info["avg_page_size"] = (
            sum(size[0] for size in page_sizes) / len(page_sizes),
            sum(size[1] for size in page_sizes) / len(page_sizes)
        )
        
        doc.close()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
        raise


def extract_pdf_text(path: Union[str, Path], page: int = 0) -> str:
    """Extract text from PDF page using PyMuPDF.
    
    Args:
        path: Path to PDF file
        page: Page number
        
    Returns:
        Extracted text string
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    try:
        doc = fitz.open(str(path))
        
        if page < 0 or page >= len(doc):
            raise ValueError(f"Invalid page number {page}. PDF has {len(doc)} pages.")
        
        pdf_page = doc[page]
        text = pdf_page.get_text()
        
        doc.close()
        return text.strip()
        
    except Exception as e:
        logger.warning(f"Failed to extract PDF text: {e}")
        return ""


def pdf_pages_to_images(
    path: Union[str, Path],
    output_dir: Union[str, Path],
    zoom: float = 2.0,
    max_pages: Optional[int] = None,
    file_format: str = "png"
) -> List[Path]:
    """Convert all PDF pages to image files.
    
    Args:
        path: Path to PDF file
        output_dir: Directory to save images
        zoom: Zoom factor for rendering
        max_pages: Maximum number of pages to convert
        file_format: Output image format ("png", "jpg")
        
    Returns:
        List of output image file paths
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    try:
        doc = fitz.open(str(path))
        num_pages = len(doc)
        
        if max_pages is not None:
            num_pages = min(num_pages, max_pages)
        
        output_paths = []
        
        for page_num in range(num_pages):
            # Convert page to image
            img_array = pdf_page_to_rgb(str(path), page_num, zoom)
            img = Image.fromarray(img_array, mode='RGB')
            
            # Save image
            output_path = output_dir / f"{path.stem}_page_{page_num:03d}.{file_format}"
            img.save(output_path)
            output_paths.append(output_path)
            
            logger.info(f"Converted page {page_num + 1}/{num_pages} -> {output_path}")
        
        doc.close()
        return output_paths
        
    except Exception as e:
        logger.error(f"Failed to convert PDF pages: {e}")
        raise


def _resize_if_needed(img: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    """Resize image if it exceeds maximum size.
    
    Args:
        img: Input image array (H, W, 3)
        max_size: (max_width, max_height)
        
    Returns:
        Resized image array
    """
    h, w = img.shape[:2]
    max_w, max_h = max_size
    
    if w <= max_w and h <= max_h:
        return img
    
    # Calculate scale factor
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h)
    
    # Resize image
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    import cv2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    logger.info(f"Resized image from ({w}, {h}) to ({new_w}, {new_h})")
    return resized


def is_pdf_file(path: Union[str, Path]) -> bool:
    """Check if file is a PDF.
    
    Args:
        path: File path to check
        
    Returns:
        True if file is PDF
    """
    path = Path(path)
    return path.suffix.lower() == '.pdf' and path.exists()


def batch_pdf_to_images(
    pdf_dir: Union[str, Path],
    output_dir: Union[str, Path], 
    zoom: float = 2.0,
    max_pages_per_pdf: Optional[int] = None,
    file_format: str = "png"
) -> List[Path]:
    """Convert all PDFs in directory to images.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Output directory for images
        zoom: Zoom factor
        max_pages_per_pdf: Limit pages per PDF
        file_format: Output format
        
    Returns:
        List of all created image paths
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in: {pdf_dir}")
        return []
    
    all_output_paths = []
    
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            paths = pdf_pages_to_images(
                pdf_path, 
                output_dir, 
                zoom=zoom,
                max_pages=max_pages_per_pdf,
                file_format=file_format
            )
            all_output_paths.extend(paths)
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
    
    logger.info(f"Converted {len(pdf_files)} PDFs to {len(all_output_paths)} images")
    return all_output_paths