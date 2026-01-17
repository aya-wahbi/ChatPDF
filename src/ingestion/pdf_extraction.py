from unstructured.partition.pdf import partition_pdf

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using the unstructured library.
    
    Args:
      file_path (str): Path to the PDF file.
      
    Returns:
      str: The concatenated text extracted from the PDF.
    """
    try:
        elements = partition_pdf(file_path)
        text = "\n".join([elem.text for elem in elements if elem.text])
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""