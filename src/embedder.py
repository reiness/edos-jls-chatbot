# src/embedder.py
import os
from typing import List, Iterable, Any
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings

load_dotenv()  # ensure .env is loaded if present

class GeminiEmbedder(Embeddings):
    """
    LangChain-compatible embedder using google-genai (gemini-embedding-001).
    Implements:
      - embed_documents(list[str]) -> list[list[float]]   (LangChain API)
      - embed_query(str) -> list[float]
      - embed_texts(list[str]) -> np.ndarray            (convenience for ingestion)
    """

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment (.env). Please set it and retry.")

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        try:
            from google import genai  # type: ignore
            self.genai = genai
            # Pass api_key explicitly so SDK doesn't rely on other env heuristics
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize google-genai Client: {e}") from e

    def _extract_vectors_from_response(self, resp: Any) -> List[List[float]]:
        """
        Given a response from client.models.embed_content(...),
        return a list of plain python lists (floats) representing embeddings.
        Handles a few possible shapes returned by the SDK.
        """
        # Try several ways to access 'embeddings'
        raw = None
        if hasattr(resp, "embeddings"):
            raw = resp.embeddings
        elif isinstance(resp, dict) and "embeddings" in resp:
            raw = resp["embeddings"]
        else:
            # fallback: maybe the SDK stores the output in resp.output or resp.results
            if hasattr(resp, "results"):
                raw = getattr(resp, "results")
            else:
                raw = None

        if raw is None:
            raise RuntimeError(f"No embeddings found in response object. Response repr: {repr(resp)}")

        vectors = []
        for i, item in enumerate(raw):
            # case 1: already a list / tuple / numpy array
            if isinstance(item, (list, tuple, np.ndarray)):
                vectors.append(list(map(float, item)))
                continue

            # case 2: object with attribute 'values'
            if hasattr(item, "values"):
                vals = getattr(item, "values")
                # sometimes values is an object; try to convert
                if isinstance(vals, (list, tuple, np.ndarray)):
                    vectors.append(list(map(float, vals)))
                    continue
                # if it's another wrapper try to iterate
                try:
                    vectors.append([float(x) for x in vals])
                    continue
                except Exception:
                    pass

            # case 3: object with attribute 'embedding'
            if hasattr(item, "embedding"):
                vals = getattr(item, "embedding")
                if isinstance(vals, (list, tuple, np.ndarray)):
                    vectors.append(list(map(float, vals)))
                    continue
                try:
                    vectors.append([float(x) for x in vals])
                    continue
                except Exception:
                    pass

            # case 4: dict-like
            if isinstance(item, dict):
                if "values" in item:
                    vectors.append([float(x) for x in item["values"]])
                    continue
                if "embedding" in item:
                    vectors.append([float(x) for x in item["embedding"]])
                    continue

            # case 5: try to iterate over item
            try:
                vectors.append([float(x) for x in item])
                continue
            except Exception:
                pass

            # if not parsed, raise with diagnostic
            raise RuntimeError(f"Unable to parse embedding item at index {i}. Type: {type(item)}. Item repr: {repr(item)}")

        return vectors

    def _call_embed_api(self, texts: List[str]):
        """
        Call the google-genai embed_content API with a safe payload.
        Return the raw SDK response.
        """
        # Build safe contents payload. The SDK accepts list of simple strings OR the structured parts.
        # Use structured parts to be explicit:
        contents = [{"parts": [{"text": t}]} for t in texts]
        try:
            resp = self.client.models.embed_content(model=self.model_name, contents=contents)
            return resp
        except Exception as e:
            # Surface the underlying error for debugging
            raise RuntimeError(f"gemini embed_content call failed: {e}") from e

    def _embed_and_normalize(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        resp = self._call_embed_api(texts)
        vectors = self._extract_vectors_from_response(resp)

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # normalize to unit vectors (cosine similarity via inner product)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        arr = arr / norms
        return arr

    # LangChain API
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        arr = self._embed_and_normalize(texts)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        arr = self._embed_and_normalize([text])
        return arr[0].tolist()

    # Convenience for ingest pipeline
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self._embed_and_normalize(texts)
    
    # ADD THIS FUNCTION TO THE END OF src/embedder.py

def get_preferred_embedder(prefer: str = "gemini", **kwargs) -> GeminiEmbedder:
    """
    Factory function to get a preferred embedder instance.
    Currently only supports 'gemini'.
    """
    if prefer.lower() == "gemini":
        # You can pass arguments like model_name through kwargs if needed
        model_name = kwargs.get("model_name")
        return GeminiEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Unknown or unsupported embedder preference: '{prefer}'")
