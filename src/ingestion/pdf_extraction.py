import logging
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using the unstructured library's partition function.

    This function auto-detects the file type and retrieves text-based content if available.

    Parameters:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content, or an empty string if extraction fails.
    """
    try:
        # 'partition' auto-detects file types (PDF, image, Docx, etc.)
        elements = partition(filename=file_path, languages=["eng"])
        # Extract the text attribute from elements if available
        text = "\n".join([elem.text for elem in elements if hasattr(elem, 'text') and elem.text])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}", exc_info=True)
        return ""