from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, str]]:
    """
    Splits a given text into smaller, overlapping chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk (in characters, roughly).
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary
                              represents a chunk with its 'text' content.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        chunks.append({"text": chunk_content})
        
        # Move the start position for the next chunk, accounting for overlap
        start += chunk_size - chunk_overlap
        
        # Ensure we don't go backward or get stuck if chunk_size <= chunk_overlap
        if chunk_size <= chunk_overlap and start < len(text):
            start += 1 # Just move one character forward to avoid infinite loop if overlap is too large

    return chunks

def process_and_chunk_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, str]]:
    """
    Loads text from a file and chunks it.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap.

    Returns:
        List[Dict[str, str]]: A list of text chunks.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return chunk_text(content, chunk_size, chunk_overlap)
    except Exception as e:
        print(f"Error processing and chunking {file_path}: {e}")
        return []


if __name__ == "__main__":
    sample_text = "This is a very long sample text that needs to be chunked. We want to make sure that the chunks have some overlap so that context is preserved across the boundaries. This is important for maintaining coherence when retrieving information later on. Let's make this text even longer to properly test the chunking mechanism. More text means more chunks, and more opportunities to see the overlap in action. The goal is to simulate a research paper's content."
    
    print("--- Testing chunk_text ---")
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk['text']}")
        print("-" * 20)

