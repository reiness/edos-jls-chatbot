# src/jls_chatbot/pipeline/build_index.py
import sys
from pathlib import Path
import os
import json

# --- ✨ FINAL FIX: Robust Path and Import Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
# --- END OF FIX ---

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from jls_chatbot.core.embedder import GeminiEmbedder

# --- CONFIGURATION ---
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", PROJECT_ROOT / "data" / "processed" / "chunks"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "data" / "processed" / "index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_index(force_rebuild: bool = True):
    faiss_path = INDEX_DIR / "faiss_index"
    if faiss_path.exists() and not force_rebuild:
        print("[build_index] Index exists and force_rebuild=False, skipping.")
        return

    chunks_file = CHUNKS_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks missing at {chunks_file}. Run ingest.py first.")

    docs = []
    print("[build_index] Preparing documents from enriched chunks...")
    with open(chunks_file, "r", encoding="utf-8") as fin:
        for line in fin:
            chunk = json.loads(line)
            
            title = chunk.get("title", "Unknown Document")
            section = chunk.get("section", "Uncategorized")
            author = chunk.get("author", "Unknown Author")
            date = chunk.get("date", "Unknown Date")
            link = chunk.get("link", "")
            source_filename = chunk.get("source_filename", "unknown.pdf")
            text = chunk.get("text", "")
            
            content_to_embed = (
                f"SOP Title: {title}\n"
                f"Section: {section}\n"
                f"Author: {author}\n"
                f"Date: {date}\n\n"
                f"Content: {text}"
            )
            
            meta = {
                "title": title,
                "section": section,
                "author": author,
                "date": date,
                "link": link,
                "source": source_filename,
                "chunk_id": chunk.get("id"),
                "original_text": text
            }
            
            docs.append(Document(page_content=content_to_embed, metadata=meta))

    embedder = GeminiEmbedder()
    print(f"[build_index] Creating FAISS index with {len(docs)} full-context documents...")
    vs = FAISS.from_documents(docs, embedder.get_langchain_embedder())
    vs.save_local(str(faiss_path))
    print(f"✅ [build_index] Saved FAISS index to {faiss_path}")

if __name__ == "__main__":
    build_index(force_rebuild=True)