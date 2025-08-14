# scripts/build_index.py
import sys
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from src.embedder import GeminiEmbedder

CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "./data/chunks"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data/index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_index(force_rebuild: bool = True):
    faiss_path = INDEX_DIR / "faiss_index"
    if faiss_path.exists() and not force_rebuild:
        print("[build_index] index exists and force_rebuild=False, skipping.")
        return

    chunks_file = CHUNKS_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks missing at {chunks_file}. Run ingest first.")

    docs = []
    with open(chunks_file, "r", encoding="utf-8") as fin:
        for line in fin:
            m = json.loads(line)
            meta = {"source": m.get("source"), "page": m.get("page"), "heading": m.get("heading"), "chunk_id": m.get("id")}
            docs.append(Document(page_content=m.get("text",""), metadata=meta))

    # Use GeminiEmbedder (langchain-compatible)
    emb = GeminiEmbedder()
    print("[build_index] creating FAISS index (this may take a while)...")
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(faiss_path))
    print("[build_index] saved index to", faiss_path)

if __name__ == "__main__":
    build_index(force_rebuild=True)
