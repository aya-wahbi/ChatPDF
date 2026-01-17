from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import os
import pickle # For saving/loading FAISS index and chunk metadata

# Placeholder for vector database imports (FAISS or ChromaDB)
# For FAISS
try:
    import faiss
except ImportError:
    faiss = None
    print("FAISS not installed. Please install with 'pip install faiss-cpu' or 'faiss-gpu'.")

# For ChromaDB
# import chromadb # If you decide to use ChromaDB

class VectorIndex:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_type: str = 'faiss'):
        """
        Initializes the VectorIndex with a SentenceTransformer model and a vector database.

        Args:
            model_name (str): Name of the SentenceTransformer model to use.
            index_type (str): Type of vector database to use ('faiss' or 'chromadb').
        """
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index_type = index_type
        self.vector_db = None
        self.chunk_metadata: List[Dict[str, Any]] = [] # To store original chunks and their metadata

        if self.index_type == 'faiss':
            if faiss is None:
                raise ImportError("FAISS is required for index_type='faiss'. Please install it.")
            # FAISS index will be initialized after first embeddings are added
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.vector_db = None # Will be faiss.IndexFlatL2 or similar
        elif self.index_type == 'chromadb':
            print("ChromaDB integration is not fully implemented in this example. You'll need to add it.")
            # self.vector_db = chromadb.Client() # Example initialization
            # self.collection = self.vector_db.get_or_create_collection("chatpdf_chunks")
        else:
            raise ValueError(f"Unsupported index_type: {index_type}. Choose 'faiss' or 'chromadb'.")
        
        print("Model loaded successfully.")

    def _initialize_faiss_index(self, embeddings: np.ndarray):
        """Initializes the FAISS index with the first set of embeddings."""
        if self.vector_db is None:
            self.vector_db = faiss.IndexFlatL2(self.dimension) # L2 distance for similarity
        self.vector_db.add(embeddings)

    def add_chunks(self, chunks_with_metadata: List[Dict[str, Any]]):
        """
        Adds text chunks and their metadata to the vector index.

        Args:
            chunks_with_metadata (List[Dict[str, Any]]): A list of dictionaries,
                                                        each containing 'text' and other metadata.
        """
        if not chunks_with_metadata:
            return

        texts = [chunk['text'] for chunk in chunks_with_metadata]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        print("Embeddings generated.")

        if self.index_type == 'faiss':
            if self.vector_db is None:
                self._initialize_faiss_index(embeddings)
            else:
                self.vector_db.add(embeddings)
            self.chunk_metadata.extend(chunks_with_metadata)
        elif self.index_type == 'chromadb':
            # Add logic for ChromaDB here
            # self.collection.add(
            #     embeddings=embeddings.tolist(),
            #     documents=texts,
            #     metadatas=[{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks_with_metadata],
            #     ids=[f"chunk_{len(self.chunk_metadata) + i}" for i in range(len(chunks_with_metadata))]
            # )
            # self.chunk_metadata.extend(chunks_with_metadata) # Still good to keep local metadata for easy access
            pass # Placeholder
        print(f"Added {len(chunks_with_metadata)} chunks to the index.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a similarity search for the given query.

        Args:
            query (str): The natural language query.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries for the top k relevant chunks,
                                  including their original text and metadata.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        if self.index_type == 'faiss':
            if self.vector_db is None or self.vector_db.ntotal == 0:
                print("FAISS index is empty. No search can be performed.")
                return []
            
            distances, indices = self.vector_db.search(query_embedding, k)
            
            # Retrieve the actual chunks using the indices
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunk_metadata): # Ensure index is valid
                    chunk = self.chunk_metadata[idx].copy()
                    chunk['distance'] = distances[0][i] # Add distance for debugging/ranking
                    results.append(chunk)
            return results
        elif self.index_type == 'chromadb':
            # Add logic for ChromaDB search here
            # results = self.collection.query(
            #     query_embeddings=query_embedding.tolist(),
            #     n_results=k,
            #     include=['documents', 'metadatas', 'distances']
            # )
            # formatted_results = []
            # for i in range(len(results['documents'][0])):
            #     formatted_results.append({
            #         "text": results['documents'][0][i],
            #         "distance": results['distances'][0][i],
            #         **results['metadatas'][0][i]
            #     })
            # return formatted_results
            pass # Placeholder
        return []

    def save_index(self, index_path: str, metadata_path: str):
        """Saves the FAISS index and chunk metadata to disk."""
        if self.index_type == 'faiss' and self.vector_db is not None:
            faiss.write_index(self.vector_db, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            print(f"FAISS index saved to {index_path} and metadata to {metadata_path}")
        else:
            print("Cannot save index for current configuration or index is not FAISS.")

    def load_index(self, index_path: str, metadata_path: str):
        """Loads the FAISS index and chunk metadata from disk."""
        if self.index_type == 'faiss' and os.path.exists(index_path) and os.path.exists(metadata_path):
            self.vector_db = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            print(f"FAISS index loaded from {index_path} and metadata from {metadata_path}")
            return True
        else:
            print("Could not load index. Files not found or not FAISS type.")
            return False

if __name__ == "__main__":
    # Example Usage
    print("--- Initializing VectorIndex (FAISS) ---")
    vector_index = VectorIndex(index_type='faiss')

    # Create some dummy chunks (in a real scenario, these would come from text_chunking.py)
    sample_chunks = [
        {"text": "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to 'learn' from data.", "source": "paper1.pdf", "page": 1},
        {"text": "Deep learning is a subset of machine learning based on artificial neural networks with representation learning.", "source": "paper1.pdf", "page": 2},
        {"text": "The challenges in medical imaging often involve high dimensionality and noise in data, requiring robust algorithms.", "source": "paper2.pdf", "page": 5},
        {"text": "Computer vision techniques are widely applied in autonomous driving for object detection and scene understanding.", "source": "paper3.pdf", "page": 3},
        {"text": "Natural Language Processing (NLP) focuses on the interaction between computers and human language.", "source": "paper4.pdf", "page": 1},
        {"text": "One common challenge in applying machine learning to medical imaging is the limited availability of labeled datasets.", "source": "paper2.pdf", "page": 6},
        {"text": "Another challenge is the interpretability of deep learning models in critical medical diagnoses.", "source": "paper2.pdf", "page": 7},
    ]

    print("\n--- Adding chunks to index ---")
    vector_index.add_chunks(sample_chunks)

    # Save and load test
    index_dir = "data/index"
    os.makedirs(index_dir, exist_ok=True)
    index_file = os.path.join(index_dir, "faiss_index.bin")
    metadata_file = os.path.join(index_dir, "chunk_metadata.pkl")
    vector_index.save_index(index_file, metadata_file)

    # Create a new index object to test loading
    print("\n--- Loading index from disk ---")
    loaded_vector_index = VectorIndex(index_type='faiss')
    loaded_vector_index.load_index(index_file, metadata_file)

    # Perform a search
    query = "What are the common challenges in applying machine learning to medical imaging?"
    print(f"\n--- Searching for: '{query}' ---")
    results = loaded_vector_index.search(query, k=3)

    if results:
        print("Top 3 relevant chunks:")
        for i, res in enumerate(results):
            print(f"Result {i+1} (Source: {res.get('source', 'N/A')}, Page: {res.get('page', 'N/A')}):")
            print(f"  Text: {res['text'][:150]}...") # Print first 150 chars
            print(f"  Distance: {res.get('distance', 'N/A')}")
            print("-" * 30)
    else:
        print("No results found.")

    query_nlp = "What is NLP?"
    print(f"\n--- Searching for: '{query_nlp}' ---")
    results_nlp = loaded_vector_index.search(query_nlp, k=1)
    if results_nlp:
        print(f"Top 1 relevant chunk for NLP: {results_nlp[0]['text']}")