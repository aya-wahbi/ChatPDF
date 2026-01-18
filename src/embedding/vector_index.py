import os
import pickle
import logging
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None
    logging.error("FAISS not installed. Install it via 'pip install faiss-cpu' or 'faiss-gpu'.")

logger = logging.getLogger(__name__)

class VectorIndex:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the VectorIndex with a pre-trained SentenceTransformer model and a FAISS vector database.

        Args:
            model_name (str): Name of the SentenceTransformer model to use.
        """
        logger.info("Loading SentenceTransformer model: %s...", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        if faiss is None:
            raise ImportError("FAISS library is required. Please install it.")
        self.vector_db = None  # Will be initialized as a FAISS index (e.g. faiss.IndexFlatL2)
        self.chunk_metadata: List[Dict[str, Any]] = []  # Stores original chunks/metadata
        logger.info("Model loaded successfully.")

    def _initialize_faiss_index(self, embeddings: np.ndarray):
        """Initializes the FAISS index using the provided embeddings."""
        if self.vector_db is None:
            self.vector_db = faiss.IndexFlatL2(self.dimension)
        self.vector_db.add(embeddings)

    def add_chunks(self, chunks_with_metadata: List[Dict[str, Any]]):
        """
        Adds text chunks and their metadata to the FAISS index.

        Args:
            chunks_with_metadata (List[Dict[str, Any]]): List of dictionaries with 'text' and optional metadata.
        """
        if not chunks_with_metadata:
            logger.warning("No chunks provided for indexing.")
            return

        texts = [chunk['text'] for chunk in chunks_with_metadata]
        logger.info("Generating embeddings for %d chunks...", len(texts))
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info("Embeddings generated.")

        if self.vector_db is None:
            self._initialize_faiss_index(embeddings)
        else:
            self.vector_db.add(embeddings)
        
        self.chunk_metadata.extend(chunks_with_metadata)
        logger.info("Added %d chunks to the index.", len(chunks_with_metadata))

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a similarity search using the given query.

        Args:
            query (str): Natural language query.
            k (int): Number of top relevant chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: List of dictionaries for the top k chunks, each including text, metadata, and distance.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        if self.vector_db is None or self.vector_db.ntotal == 0:
            logger.warning("FAISS index is empty. Unable to perform search.")
            return []

        distances, indices = self.vector_db.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_metadata):
                chunk = self.chunk_metadata[idx].copy()
                chunk['distance'] = distances[0][i]
                results.append(chunk)
        return results

    def save_index(self, index_path: str, metadata_path: str):
        """
        Saves the FAISS index and metadata to disk.

        Args:
            index_path (str): File path to save the FAISS index.
            metadata_path (str): File path to save the metadata.
        """
        if self.vector_db is not None:
            faiss.write_index(self.vector_db, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            logger.info("FAISS index saved to %s and metadata to %s", index_path, metadata_path)
        else:
            logger.error("No FAISS index to save.")

    def load_index(self, index_path: str, metadata_path: str) -> bool:
        """
        Loads the FAISS index and metadata from disk.

        Args:
            index_path (str): File path from which to load the FAISS index.
            metadata_path (str): File path from which to load the metadata.

        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.vector_db = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            logger.info("FAISS index loaded from %s and metadata from %s", index_path, metadata_path)
            return True
        else:
            logger.error("Index file or metadata file not found.")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("=== Initializing VectorIndex (FAISS) ===")
    vector_index = VectorIndex(model_name='all-MiniLM-L6-v2')

    # Create sample chunks relevant to RL, ML, classification/regression, neural networks, and deep learning.
    sample_chunks = [
        {"text": "Reinforcement Learning (RL) involves agents making decisions to maximize cumulative rewards.", "source": "paper_rl.pdf", "page": 1},
        {"text": "Machine Learning (ML) provides approaches for algorithms to learn from data and make predictions.", "source": "paper_ml.pdf", "page": 2},
        {"text": "Classification tasks involve categorizing data into predefined labels.", "source": "paper_classification.pdf", "page": 3},
        {"text": "Regression analysis is used to predict continuous outcomes based on input variables.", "source": "paper_regression.pdf", "page": 4},
        {"text": "Neural Networks, a core of deep learning, consist of layers of interconnected nodes.", "source": "paper_nn.pdf", "page": 5},
        {"text": "Deep Learning techniques have significantly advanced fields such as computer vision and NLP.", "source": "paper_deeplearning.pdf", "page": 6},
    ]

    logger.info("=== Adding sample chunks to the index ===")
    vector_index.add_chunks(sample_chunks)

    # Save the index and metadata to disk.
    index_dir = "data/index"
    os.makedirs(index_dir, exist_ok=True)
    index_file = os.path.join(index_dir, "faiss_index.bin")
    metadata_file = os.path.join(index_dir, "chunk_metadata.pkl")
    vector_index.save_index(index_file, metadata_file)

    # Load the index from disk and perform a search.
    logger.info("=== Loading index from disk ===")
    new_vector_index = VectorIndex(model_name='all-MiniLM-L6-v2')
    if new_vector_index.load_index(index_file, metadata_file):
        query = "How are neural networks used in deep learning?"
        logger.info("=== Searching for query: '%s' ===", query)
        results = new_vector_index.search(query, k=3)
        if results:
            for i, res in enumerate(results):
                logger.info("Result %d: Source: %s, Page: %s, Distance: %.4f", 
                            i + 1, res.get("source", "N/A"), res.get("page", "N/A"), res.get("distance", 0))
                logger.info("Text: %s", res.get("text", "")[:150])
        else:
            logger.info("No results found.")
    else:
        logger.error("Failed to load index from disk.")