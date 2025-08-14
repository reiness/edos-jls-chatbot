# src/qa_chain.py
import os
import json
import numpy as np
import faiss
from pathlib import Path

INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
METADATA_FILE = Path(INDEX_DIR) / "metadata.jsonl"
INDEX_FILE = Path(INDEX_DIR) / "faiss.index"

def load_index_and_meta():
    index = faiss.read_index(str(INDEX_FILE))
    meta = [json.loads(line) for line in open(METADATA_FILE, 'r', encoding='utf-8')]
    return index, meta

def embed_query_with_provider(query: str, embedder):
    vec = embedder.embed_texts([query])[0]
    return np.array(vec, dtype='float32')

def retrieve(query: str, embedder, top_k=5):
    index, meta = load_index_and_meta()
    qvec = embed_query_with_provider(query, embedder).reshape(1, -1)
    D, I = index.search(qvec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta_item = meta[idx]
        results.append({"score": float(score), "metadata": meta_item})
    return results

def build_context(retrieved):
    parts = []
    for r in retrieved:
        m = r["metadata"]
        parts.append(f"Source: {m.get('source')} (page:{m.get('page')})\n{m.get('text')}\n---")
    return "\n".join(parts)

# Gemini 2.5-Flash generation via google-genai
def answer_with_gemini(question: str, context: str, model=None, temperature=0.0, max_output_tokens=512):
    model = model or os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
    except Exception as e:
        raise RuntimeError("google-genai SDK not installed or GEMINI_API_KEY not set.") from e

    prompt = (
        "You are a strict assistant who MUST answer questions using ONLY the provided SOP context.\n"
        "If the answer is not present in the context, say: 'I don't know, see these sources' and list the most relevant SOP filenames.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely and show sources in [file:page] format:"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        # GPU / options can go to config; keep defaults for now
        # config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens)
    )
    # response.text contains the textual answer (SDK quickstart)
    return response.text
