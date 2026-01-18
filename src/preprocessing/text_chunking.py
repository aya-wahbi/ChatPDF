import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, str]]:
    """
    Splits a given text into smaller, overlapping chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk, in characters.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary represents a chunk with its 'text'.
    """
    if not text:
        logger.warning("Received empty text to chunk; returning empty list.")
        return []

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk_content = text[start:end]
        chunks.append({"text": chunk_content})
        # Increase start such that there is a proper overlap.
        # Avoid infinite loop if chunk_size <= chunk_overlap.
        start += (chunk_size - chunk_overlap) if chunk_size > chunk_overlap else 1

    return chunks

def process_and_chunk_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, str]]:
    """
    Loads text from a file and creates overlapping chunks.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap (in characters) between consecutive chunks.

    Returns:
        List[Dict[str, str]]: A list of text chunks.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return chunk_text(content, chunk_size, chunk_overlap)
    except Exception as e:
        logger.error("Error processing and chunking %s: %s", file_path, e, exc_info=True)
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_text = (
        "This is a very long sample text that needs to be chunked. "
        "We want to make sure that the chunks have some overlap so that context is preserved across boundaries. "
        "This is important for maintaining coherence when retrieving information later on. "
        "Let's make this text even longer to properly test the chunking mechanism. "
        "More text means more chunks, and more opportunities to see the overlap in action. "
        "The goal is to simulate a research paper's content."
    )
    
    logging.info("=== Testing chunk_text ===")
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(chunks):
        logging.info("Chunk %d: %s", i + 1, chunk["text"])
        logging.info("-" * 40)