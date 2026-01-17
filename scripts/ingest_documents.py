import os
from src.ingestion.document_loader import load_document
from src.preprocessing.text_chunking import process_and_chunk_document
from src.embedding.vector_index import VectorIndex
import json # To save processed chunks with metadata

def main():
    raw_folder = 'data/raw'
    processed_text_folder = 'data/processed'
    index_folder = 'data/index'
    
    # Ensure folders exist
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(processed_text_folder, exist_ok=True)
    os.makedirs(index_folder, exist_ok=True)

    # Initialize the VectorIndex
    vector_index = VectorIndex(model_name='all-MiniLM-L6-v2', index_type='faiss')

    all_processed_chunks = []

    for file_name in os.listdir(raw_folder):
        file_path = os.path.join(raw_folder, file_name)
        if not os.path.isfile(file_path):
            continue # Skip directories

        print(f"Processing {file_name}...")
        
        # 1. Load and extract text
        content = load_document(file_path)
        
        if content:
            # Save extracted text (optional, but good for debugging)
            text_output_file = os.path.join(processed_text_folder, f"{os.path.splitext(file_name)[0]}.txt")
            with open(text_output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved extracted text to {text_output_file}")

            # 2. Chunk the text
            # Add metadata to chunks, e.g., source file name
            chunks = process_and_chunk_document(text_output_file, chunk_size=1000, chunk_overlap=200)
    
            # Add source metadata to each chunk
            for chunk in chunks:
                chunk['source'] = file_name # You might want to add page numbers here too if available from unstructured

            all_processed_chunks.extend(chunks)
            print(f"Chunked {len(chunks)} segments from {file_name}")
            
        else:
            print(f"Failed to process {file_name}. Skipping chunking and embedding.")

    # 3. Add all chunks to the vector index
    if all_processed_chunks:
        print(f"\nAdding {len(all_processed_chunks)} total chunks to the vector index...")
        vector_index.add_chunks(all_processed_chunks)
        print("All chunks embedded and indexed.")

        # 4. Save the vector index and metadata
        index_file_path = os.path.join(index_folder, "faiss_index.bin")
        metadata_file_path = os.path.join(index_folder, "chunk_metadata.pkl")
        vector_index.save_index(index_file_path, metadata_file_path)
    else:
        print("No chunks were processed. Vector index not created/saved.")

if __name__ == "__main__":
    # Create dummy raw PDF files for testing
    # You'll need to place actual PDFs in 'data/raw' for real testing
    # For now, let's simulate by creating a dummy text file that load_document can handle
    # (assuming you extend load_document to handle .txt or just manually place a PDF)
    
    # Example: Create a dummy PDF file (this won't work as a real PDF,
    # but demonstrates the need for actual PDFs in data/raw)
    # For a real test, you'd put actual PDF files here.
    dummy_pdf_content = """
    This is a research paper on quantum computing. Quantum computers use quantum-mechanical phenomena such as superposition and entanglement to perform computations.
    A key challenge in quantum computing is error correction, as quantum states are very fragile.
    Another section discusses the applications of quantum computing in drug discovery and materials science.
    The conclusion highlights the potential for exponential speedups in certain problems.
    """
    dummy_pdf_path = os.path.join('data/raw', 'dummy_research_paper.pdf')
    # NOTE: This will create a .pdf file, but it's just text.
    # The `unstructured` library might still be able to extract text from it,
    # but for a true test, you'd need a proper PDF.
    # For demonstration, let's save it as a .txt and modify document_loader to handle .txt
    
    # To properly test, you need real PDFs in data/raw.
    # For a quick test of the *pipeline*, you could temporarily modify document_loader.py
    # to handle .txt files and put a .txt file in data/raw.
    # Or, just ensure you have a real PDF in data/raw.

    # Let's assume you have a real PDF named 'example.pdf' in 'data/raw'
    # If you don't, create a simple text file and adjust `document_loader.py` to handle it
    # For testing purposes, I'll create a dummy text file that `load_document` *could* read if extended.
    # For your actual project, ensure `data/raw` contains real PDFs.
    
    # Example of how you would run this script after placing real PDFs in data/raw:
    # python scripts/ingest_documents.py
    
    print("Starting document ingestion and indexing process...")
    main()
    print("\nIngestion and indexing complete.")