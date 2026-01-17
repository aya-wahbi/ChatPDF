import os
from .pdf_extraction import extract_text_from_pdf

def load_document(file_path):
    """
    Load and return text content from a document.
    
    Currently supports PDF files. More formats (e.g., DOCX, TXT) can be added.
    
    Args:
      file_path (str): Path to the document file.
      
    Returns:
      str: Text extracted from the document.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return ""