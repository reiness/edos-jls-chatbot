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
    print("[build_index] Preparing documents from chunks...")
    with open(chunks_file, "r", encoding="utf-8") as fin:
        for line in fin:
            m = json.loads(line)
            
            # --- CRITICAL MODIFICATION FOR SOP STRUCTURE STARTS HERE ---
            
            # 1. Extract all structured fields from the chunk.
            source = m.get("source", "Unknown Document")
            heading = m.get("heading", "No Section")
            author = m.get("author", "Unknown Author")
            date = m.get("date", "Unknown Date")
            related_sops = m.get("related_sops", []) # Expecting a list
            text = m.get("text", "")
            
            # 2. Create the "perfect" string for the embedding, mirroring the SOP anatomy.
            # This gives the embedding model the full context for maximum precision.
            content_to_embed = (
                f"SOP Title: {source}\n"
                f"Author: {author}\n"
                f"Date: {date}\n"
                f"Section: {heading}\n\n"
                f"Content: {text}"
            )
            
            # 3. Create the enriched metadata for filtering, citation, and linking.
            # We store everything, especially the relational 'related_sops' data.
            meta = {
                "source": source,
                "heading": heading,
                "author": author,
                "date": date,
                "related_sops": related_sops,
                "page": m.get("page"), 
                "chunk_id": m.get("id"),
                "original_text": text # Keep the clean text for display.
            }
            
            # 4. Create the LangChain Document.
            docs.append(Document(page_content=content_to_embed, metadata=meta))
            
            # --- CRITICAL MODIFICATION ENDS HERE ---

    # Use GeminiEmbedder (langchain-compatible)
    emb = GeminiEmbedder()
    print(f"[build_index] Creating FAISS index with {len(docs)} full-context documents (this may take a while)...")
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(faiss_path))
    print("[build_index] Saved knowledge graph index to", faiss_path)

if __name__ == "__main__":
    build_index(force_rebuild=True)