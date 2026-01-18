import os
import logging
from .pdf_extraction import extract_text_from_pdf
from .image_ocr import extract_text_from_image_easyocr

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

def load_document(file_path: str) -> str:
    """
    Load a document (PDF, DOCX, image, etc.) and extract its text content.

    This function checks the file extension to decide if the document should
    be processed with image OCR or standard text extraction.

    Parameters:
        file_path (str): The full path to the document.

    Returns:
        str: The extracted text content from the document.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in IMAGE_EXTS:
        return extract_text_from_image_easyocr(file_path)

    # Fallback extraction for PDF, DOCX, etc.
    return extract_text_from_pdf(file_path)