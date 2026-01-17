from typing import List, Dict, Any
from src.embedding.vector_index import VectorIndex
import os

# Placeholder for LLM integration
# For OpenAI API
# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# For local LLMs via Ollama
# import requests # You'd need to install 'requests'
# def call_ollama(prompt: str, model: str = "llama2"):
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False # For simple, single response
#     }
#     try:
#         response = requests.post(url, json=payload)
#         response.raise_for_status() # Raise an exception for bad status codes
#         return response.json()['response']
#     except requests.exceptions.RequestException as e:
#         print(f"Error calling Ollama: {e}")
#         return "I'm sorry, I couldn't connect to the local LLM."

class ChatPDFAssistant:
    def __init__(self, index_path: str, metadata_path: str, llm_type: str = 'ollama', llm_model_name: str = 'llama2'):
        """
        Initializes the ChatPDFAssistant, loading the vector index and setting up the LLM.

        Args:
            index_path (str): Path to the saved FAISS index.
            metadata_path (str): Path to the saved chunk metadata.
            llm_type (str): Type of LLM to use ('openai' or 'ollama').
            llm_model_name (str): Specific model name for the LLM (e.g., 'gpt-3.5-turbo', 'llama2').
        """
        self.vector_index = VectorIndex(index_type='faiss')
        if not self.vector_index.load_index(index_path, metadata_path):
            raise FileNotFoundError("Could not load vector index. Please ensure ingestion is complete.")
        
        self.llm_type = llm_type
        self.llm_model_name = llm_model_name

        if self.llm_type == 'openai':
            # self.llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print("OpenAI client initialized (requires OPENAI_API_KEY env var).")
        elif self.llm_type == 'ollama':
            print(f"Ollama client initialized. Ensure Ollama server is running and model '{llm_model_name}' is downloaded.")
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Choose 'openai' or 'ollama'.")

    def _generate_answer_with_llm(self, prompt: str) -> str:
        """
        Generates an answer using the configured LLM.
        """
        if self.llm_type == 'openai':
            # response = self.llm_client.chat.completions.create(
            #     model=self.llm_model_name,
            #     messages=[{"role": "user", "content": prompt}]
            # )
            # return response.choices[0].message.content
            return "OpenAI LLM integration placeholder response." # Placeholder
        elif self.llm_type == 'ollama':
            # return call_ollama(prompt, self.llm_model_name)
            return "Ollama LLM integration placeholder response." # Placeholder
        return "LLM integration not implemented."

    def query_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Answers a natural language query by retrieving relevant chunks and using an LLM.

        Args:
            query (str): The user's natural language query.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the generated answer and source citations.
        """
        print(f"Searching for relevant documents for query: '{query}'")
        relevant_chunks = self.vector_index.search(query, k=k)

        if not relevant_chunks:
            return {"answer": "I couldn't find any relevant information in your documents.", "sources": []}

        # Construct the prompt for the LLM
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Prepare sources for citation
        sources = []
        for chunk in relevant_chunks:
            source_info = f"Source: {chunk.get('source', 'Unknown')}"
            if 'page' in chunk:
                source_info += f", Page: {chunk['page']}"
            sources.append(source_info)
        
        # Remove duplicates from sources while preserving order (optional, but good for clean output)
        unique_sources = list(dict.fromkeys(sources))
        sources_str = "\n".join(unique_sources)

        prompt = f"""
        You are an AI research assistant. Use the following pieces of context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a concise answer and cite the source documents at the end.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        print("\n--- Sending prompt to LLM ---")
        answer = self._generate_answer_with_llm(prompt)
        
        return {"answer": answer, "sources": unique_sources}

if __name__ == "__main__":
    # Ensure you have run scripts/ingest_documents.py first to create the index
    index_dir = "data/index"
    index_file = os.path.join(index_dir, "faiss_index.bin")
    metadata_file = os.path.join(index_dir, "chunk_metadata.pkl")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        print("Error: Vector index files not found. Please run 'scripts/ingest_documents.py' first.")
    else:
        print("--- Initializing ChatPDFAssistant ---")
        try:
            # You'll need to set up your LLM (e.g., run Ollama locally or provide OpenAI API key)
            # For testing, you might start with 'ollama' and ensure it's running
            assistant = ChatPDFAssistant(index_file, metadata_file, llm_type='ollama', llm_model_name='llama2')

            # Example queries
            queries = [
                "What are the main challenges in quantum computing?",
                "How does deep learning relate to machine learning?",
                "What applications does quantum computing have?",
                "Tell me about NLP."
            ]

            for q in queries:
                print(f"\n--- User Query: {q} ---")
                response = assistant.query_documents(q, k=3)
                print(f"\nAI Answer: {response['answer']}")
                if response['sources']:
                    print("\nSources:")
                    for source in response['sources']:
                        print(f"- {source}")
                print("\n" + "="*50 + "\n")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your LLM (Ollama or OpenAI) is correctly configured and accessible.")