# scripts/ingest.py
import sys
from pathlib import Path
import os
import json
from tqdm import tqdm
import numpy as np
import fitz  # PyMuPDF

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import clean_text, is_probable_heading
from src.embedder import GeminiEmbedder

PDF_FOLDER = os.getenv("PDF_FOLDER", "./data/pdfs")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "./data/chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

def extract_text_from_pdf(path: str):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_num": i+1, "text": text})
    return pages

def chunk_text(text: str, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((start, min(end, L), chunk))
        start = max(end - overlap, end)
        if start == end:
            start += chunk_size
    return chunks

def ingest_all(pdf_folder=PDF_FOLDER):
    # Strict: only GeminiEmbedder
    try:
        provider = GeminiEmbedder()
    except Exception as e:
        print("[ingest][fatal] cannot initialize GeminiEmbedder:", e)
        print("Make sure google-genai is installed and GEMINI credentials are set (GEMINI_API_KEY or Vertex env).")
        raise

    metadata = []
    all_texts = []
    next_id = 0

    pdf_folder = Path(pdf_folder)
    pdf_files = sorted(list(pdf_folder.glob("**/*.pdf")))
    if not pdf_files:
        print(f"[ingest] No PDFs found in {pdf_folder}. Put SOP PDFs there first.")
        return

    for pdf_path in tqdm(pdf_files, desc="PDF files"):
        pages = extract_text_from_pdf(str(pdf_path))
        for p in pages:
            page_text = clean_text(p["text"])
            if not page_text.strip():
                continue

            heading = None
            first_lines = page_text.split("\n")[:5]
            for line in first_lines:
                if is_probable_heading(line):
                    heading = line.strip()
                    break

            for (char_start, char_end, chunk_text_str) in chunk_text(page_text):
                chunk_meta = {
                    "id": next_id,
                    "source": str(pdf_path),
                    "page": p["page_num"],
                    "heading": heading,
                    "char_start": int(char_start),
                    "char_end": int(char_end)
                }
                metadata.append(chunk_meta)
                all_texts.append(chunk_text_str)
                next_id += 1

    if not all_texts:
        print("[ingest] No text chunks produced. Exiting.")
        return

    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "128"))  # smaller by default for reliability
    embeddings_list = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding"):
        batch = all_texts[i:i+batch_size]
        try:
            emb_batch = provider.embed_texts(batch)  # numpy array
        except Exception as e:
            print(f"[ingest][fatal] embedding failed for batch starting at {i}: {e}")
            raise

        if not isinstance(emb_batch, np.ndarray):
            # convert
            emb_batch = np.asarray(emb_batch, dtype=np.float32)
        else:
            emb_batch = emb_batch.astype(np.float32)

        if emb_batch.shape[0] != len(batch):
            print(f"[ingest][warning] embeddings count ({emb_batch.shape[0]}) != batch size ({len(batch)}) at batch starting {i}")

        embeddings_list.append(emb_batch)

    # Concatenate and persist
    embeddings_arr = np.vstack(embeddings_list) if embeddings_list else np.zeros((0,0), dtype=np.float32)

    chunks_out = Path(CHUNKS_DIR) / "chunks.jsonl"
    with open(chunks_out, "w", encoding="utf-8") as f:
        for m, t in zip(metadata, all_texts):
            m2 = m.copy()
            m2["text_preview"] = t[:400]
            m2["text"] = t
            f.write(json.dumps(m2, ensure_ascii=False) + "\n")

    emb_path = Path(CHUNKS_DIR) / "embeddings.npy"
    np.save(emb_path, embeddings_arr)
    print(f"[ingest] Saved {len(metadata)} chunks and embeddings to {CHUNKS_DIR}")

if __name__ == "__main__":
    ingest_all()
