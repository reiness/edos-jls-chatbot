# src/llm_gemini.py
import os
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from pydantic import BaseModel

class GeminiLLM(LLM, BaseModel):
    """
    LangChain LLM wrapper for google-genai (gemini-2.5-flash).
    Implements _call so LangChain chains can use the client.
    """
    model_name: str = None
    temperature: float = 0.0
    max_output_tokens: int = 512

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.model_name or os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
        try:
            from google import genai
            self.genai = genai
        except Exception as e:
            raise RuntimeError("google-genai SDK not installed; install it to use GeminiLLM") from e
        self.client = self.genai.Client()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Use generate_content with a single prompt string
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            # optional config could be added; keep simple
        )
        # SDK returns an object where textual content is in resp.text (or resp.content); using resp.text
        # The exact property may vary by SDK version; retrieving text robustly:
        out = None
        # Try a few common ways:
        if hasattr(resp, "text"):
            out = resp.text
        else:
            # some SDKs return resp.result or resp.output
            try:
                out = str(resp)
            except Exception:
                out = ""
        return out
