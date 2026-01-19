import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)


import os
import logging
from src.ingestion.document_loader import load_document
from src.preprocessing.text_chunking import process_and_chunk_document
from src.embedding.vector_index import VectorIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    raw_folder = 'data/raw'
    processed_text_folder = 'data/processed'
    index_folder = 'data/index'
    
    # Ensure directories exist
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(processed_text_folder, exist_ok=True)
    os.makedirs(index_folder, exist_ok=True)

    # Initialize the VectorIndex for FAISS (using only FAISS now)
    vector_index = VectorIndex(model_name='all-MiniLM-L6-v2')

    all_processed_chunks = []

    # Process each file in the raw data folder
    for file_name in os.listdir(raw_folder):
        file_path = os.path.join(raw_folder, file_name)
        if not os.path.isfile(file_path):
            continue  # Skip directories or non-files

        logging.info("Processing file: %s", file_name)
        
        # 1. Load and extract text from the document
        content = load_document(file_path)
        if content:
            # Optionally save the extracted text for reference or debugging
            text_output_file = os.path.join(processed_text_folder, f"{os.path.splitext(file_name)[0]}.txt")
            with open(text_output_file, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info("Saved extracted text to %s", text_output_file)

            # 2. Chunk the text and add metadata (e.g., source file name)
            chunks = process_and_chunk_document(text_output_file, chunk_size=1000, chunk_overlap=200)
            for chunk in chunks:
                chunk['source'] = file_name  # Additional metadata can be included here
            all_processed_chunks.extend(chunks)
            logging.info("Chunked %d segments from %s", len(chunks), file_name)
        else:
            logging.warning("Failed to process %s. Skipping chunking and embedding.", file_name)

    # 3. Add all chunks to the vector index
    if all_processed_chunks:
        logging.info("Adding %d total chunks to the vector index...", len(all_processed_chunks))
        vector_index.add_chunks(all_processed_chunks)
        logging.info("All chunks embedded and indexed.")

        # 4. Save the vector index and metadata to disk
        index_file_path = os.path.join(index_folder, "faiss_index.bin")
        metadata_file_path = os.path.join(index_folder, "chunk_metadata.pkl")
        vector_index.save_index(index_file_path, metadata_file_path)
    else:
        logging.warning("No chunks were processed. Vector index was not created or saved.")

if __name__ == "__main__":
    logging.info("Starting document ingestion and indexing process...")

    # Create a dummy raw text file for testing if it doesn't already exist.
    # This dummy file contains content focused on RL, ML, classification/regression,
    # neural networks, and deep learning.
    raw_folder = 'data/raw'
    os.makedirs(raw_folder, exist_ok=True)
    dummy_text_file = os.path.join(raw_folder, "dummy_ml_paper.txt")
    if not os.path.exists(dummy_text_file):
        dummy_ml_content = (
            "Reinforcement Learning (RL) is a branch of machine learning where agents learn "
            "through interactions with an environment to maximize cumulative rewards.\n\n"
            "This paper explores various machine learning (ML) techniques, including both classification "
            "and regression methods, which are essential for making predicted outputs based on data.\n\n"
            "Furthermore, the study delves into neural networks – the backbone of deep learning – "
            "examining both traditional multi-layer perceptrons and advanced architectures such as "
            "convolutional and recurrent neural networks.\n\n"
            "Applications in robotics, autonomous driving, and healthcare are highlighted, with a focus "
            "on the practical deployment of RL and ML techniques in real-world scenarios."
        )
        with open(dummy_text_file, "w", encoding="utf-8") as f:
            f.write(dummy_ml_content)
        logging.info("Created dummy file for testing: %s", dummy_text_file)

    main()
    logging.info("Document ingestion and indexing complete.")