import os
import logging
import textwrap
import ollama
from typing import List, Dict, Any
from src.embedding.vector_index import VectorIndex

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDFAssistant:
    def __init__(self, index_path: str, metadata_path: str, llm_type: str = 'ollama', llm_model_name: str = 'llama2'):
        """
        Initializes the ChatPDFAssistant, loading the vector index and setting up the LLM.
    
        Args:
            index_path (str): Path to the saved FAISS index.
            metadata_path (str): Path to the saved chunk metadata.
            llm_type (str): Type of LLM to use ('openai' or 'ollama').
            llm_model_name (str): Specific model name for the LLM.
        """
        # Removed index_type because VectorIndex only accepts model_name.
        self.vector_index = VectorIndex()  
        if not self.vector_index.load_index(index_path, metadata_path):
            raise FileNotFoundError("Could not load vector index. Please ensure ingestion is complete.")

        self.llm_type = llm_type
        self.llm_model_name = llm_model_name

        if self.llm_type == 'openai':
            # Uncomment and configure the OpenAI client when needed.
            # self.llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info("OpenAI client initialized (requires OPENAI_API_KEY env var).")
        elif self.llm_type == 'ollama':
            logger.info(f"Ollama client initialized. Ensure Ollama server is running and model '{llm_model_name}' is downloaded.")
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Choose 'openai' or 'ollama'.")

    def _get_llm_response(self, prompt_text: str) -> str:
        """
        Handles the direct interaction with the Ollama server to get a response.
        """
        try:
            response = ollama.chat(
                model=self.llm_model_name,
                messages=[{'role': 'user', 'content': prompt_text}],
                # options={'num_predict': 128},  # Uncomment and adjust if needed.
            )
            return response['message']['content']
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            return f"Ollama API error: {e}"
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return ("Failed to get a proper response from the LLM. "
                    "Please check if Ollama server is running and the model is available.")

    def _generate_answer_with_llm(self, prompt: str) -> str:
        """
        Generates an answer using the configured LLM, dispatching to the correct client.
        """
        if self.llm_type == 'openai':
            # Uncomment and complete the OpenAI API implementation if needed.
            # response = self.llm_client.chat.completions.create(
            #     model=self.llm_model_name,
            #     messages=[{"role": "user", "content": prompt}]
            # )
            # return response.choices[0].message.content
            return "OpenAI LLM integration placeholder response."
        elif self.llm_type == 'ollama':
            return self._get_llm_response(prompt)
        return "LLM integration not implemented."

    def _build_prompt(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates a structured and grounded prompt for the LLM
        using only the retrieved document context.
        """
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])

        prompt = textwrap.dedent(f"""
            You are an AI research assistant.

            Answer the user's question using ONLY the information provided
            in the context below.

            Rules:
            - If the answer is not contained in the context, say:
            "The information is not available in the provided documents."
            - Keep the answer concise and well structured.
            - Use bullet points where appropriate.
            - Do NOT use external knowledge.
            - Do NOT make assumptions.

            Context:
            {context}

            Question:
            {query}

            Answer:
        """)
        return prompt


    def query_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Answers a natural language query by retrieving relevant chunks and using an LLM.
    
        Args:
            query (str): The user's natural language query.
            k (int): The number of top relevant chunks to retrieve.
    
        Returns:
            Dict[str, Any]: A dictionary containing the generated answer and source citations.
        """
        logger.info(f"Searching for relevant documents for query: '{query}'")
        relevant_chunks = self.vector_index.search(query, k=k)
    
        if not relevant_chunks:
            return {"answer": "I couldn't find any relevant information in your documents.", "sources": []}
    
        # Prepare sources with relevance information
        sources_dict = {}

        for chunk in relevant_chunks:
            source = chunk.get("source", "Unknown")
            distance = chunk.get("distance", None)

            # Keep only the most relevant (smallest distance) chunk per source
            if source not in sources_dict or (
                distance is not None and distance < sources_dict[source]["distance"]
            ):
                sources_dict[source] = {
                    "distance": distance
                }

        # Sort sources by relevance (lower distance = more relevant)
        sorted_sources = sorted(
            sources_dict.items(),
            key=lambda x: x[1]["distance"] if x[1]["distance"] is not None else float("inf")
        )

        # Format sources for display
        final_sources = []
        for source, meta in sorted_sources:
            if meta["distance"] is not None:
                final_sources.append(
                    f"{source} (relevance score: {meta['distance']:.4f})"
                )
            else:
                final_sources.append(source)

    
        prompt = self._build_prompt(query, relevant_chunks)
        logger.info("Sending prompt to LLM...")
        answer = self._generate_answer_with_llm(prompt)
    
        return {"answer": answer, "sources": final_sources}


if __name__ == "__main__":
    index_dir = "data/index"
    index_file = os.path.join(index_dir, "faiss_index.bin")
    metadata_file = os.path.join(index_dir, "chunk_metadata.pkl")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        logger.error("Vector index files not found. Please run 'scripts/ingest_documents.py' first.")
    else:
        logger.info("Initializing ChatPDFAssistant")
        try:
            assistant = ChatPDFAssistant(index_file, metadata_file, llm_type='ollama', llm_model_name='llama2')
    
            # Updated queries reflecting topics from your ingested documents
            queries = [
                "What are the fundamental principles of deep learning, and how does it differ from traditional machine learning?",
                "How does backpropagation help in optimizing neural networks?",
                "Can you explain the architecture and advantages of transformers in deep learning?",
                "What are the challenges and potential solutions in implementing deep reinforcement learning algorithms?"
            ]
    
            for q in queries:
                logger.info(f"User Query: {q}")
                response = assistant.query_documents(q, k=3)
                logger.info(f"AI Answer: {response['answer']}")
                if response['sources']:
                    logger.info("Sources:")
                    for source in response['sources']:
                        logger.info(f" - {source}")
                logger.info("=" * 50)
        except FileNotFoundError as e:
            logger.error(e)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.error("Please ensure your LLM (Ollama or OpenAI) is correctly configured and accessible.")