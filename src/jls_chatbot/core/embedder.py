# src/jls_chatbot/core/embedder.py
import os
from typing import List
import numpy as np
from dotenv import load_dotenv

# Using the modern import path for LangChain
from langchain_core.embeddings import Embeddings

load_dotenv()  # Ensures .env is loaded for local development

class GeminiEmbedder(Embeddings):
    """
    LangChain-compatible embedder that uses the google-generativeai library
    for the 'models/embedding-001' model.
    """

    def __init__(self, model_name: str = None, api_key: str = None):
        super().__init__()
        
        # Prioritize the explicitly passed key, then environment, then raise error
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file or environment.")

        # print(self.api_key)
        # exit()
        
        # Allow model name to be passed or use environment default
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
        except ImportError:
            raise ImportError("The 'google-generativeai' package is required. Please install it with 'pip install google-generativeai'")
        except Exception as e:
            raise RuntimeError(f"Failed to configure Google Generative AI: {e}")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Helper function to call the API and handle responses."""
        if not texts:
            return np.array([])
        
        try:
            # The google-generativeai library handles batching implicitly in embed_content
            result = self.client.embed_content(model=self.model_name, content=texts)
            return np.array(result['embedding'])
        except Exception as e:
            raise RuntimeError(f"Google Generative AI embedding failed: {e}") from e

    def _embed_and_normalize(self, texts: List[str]) -> np.ndarray:
        """Embeds and then normalizes the vectors, which is best practice for FAISS."""
        embeddings = self._embed(texts)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # Normalize to unit vectors for accurate cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Add a small epsilon to avoid division by zero
        return embeddings / (norms + 1e-10)

    # --- LangChain Interface Implementation ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """For embedding multiple documents for storage in a vector store."""
        if not texts:
            return []
        arr = self._embed_and_normalize(texts)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        """For embedding a single query to search the vector store."""
        if not text:
            return []
        arr = self._embed_and_normalize([text])
        return arr[0].tolist()

    # --- Convenience method for our ingest pipeline ---
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """A convenience method that returns NumPy arrays directly."""
        return self._embed_and_normalize(texts)

    def get_langchain_embedder(self) -> Embeddings:
        """Returns self to be compatible with LangChain's FAISS loader."""
        return self


def get_preferred_embedder(prefer: str = "gemini", **kwargs) -> GeminiEmbedder:
    """
    Factory function to get a preferred embedder instance.
    Currently only supports 'gemini'.
    """
    if prefer.lower() == "gemini":
        model_name = kwargs.get("model_name")
        return GeminiEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Unknown or unsupported embedder preference: '{prefer}'")