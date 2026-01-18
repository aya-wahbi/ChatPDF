import os
import logging
import easyocr

logger = logging.getLogger(__name__)

# Create the reader once (initialization can be slow)
_reader = easyocr.Reader(["en"], gpu=False)  # Set gpu=True if CUDA is available

def extract_text_from_image_easyocr(image_path: str) -> str:
    """
    Extract text content from an image using EasyOCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text from the image. Returns an empty string if extraction fails.

    Raises:
        FileNotFoundError: If the specified image file does not exist.
    """
    if not os.path.exists(image_path):
        logger.error("File not found: %s", image_path)
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        # detail=0 returns only strings; paragraph=True merges lines for better coherence
        lines = _reader.readtext(image_path, detail=0, paragraph=True)
        return "\n".join(lines).strip()
    except Exception as e:
        logger.error("Error extracting text from image %s: %s", image_path, e, exc_info=True)
        return ""