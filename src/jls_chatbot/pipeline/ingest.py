# src/jls_chatbot/pipeline/ingest.py
import sys
from pathlib import Path
import os
import json
import math
import time
from tqdm import tqdm
import numpy as np
import fitz  # PyMuPDF
from dotenv import load_dotenv

# --- Robust Path and Import Setup ---
# This block makes the script runnable by ensuring Python knows where the 'src' directory is.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Load environment variables from .env file at the project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from jls_chatbot.core.utils import clean_text
from jls_chatbot.core.embedder import GeminiEmbedder

# --- CONFIGURATION ---
PDF_FOLDER = Path(os.getenv("PDF_FOLDER", PROJECT_ROOT / "data" / "source_documents"))
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", PROJECT_ROOT / "data" / "processed" / "chunks"))
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def ingest_all():
    """
    Reads PDFs based on .metadata.json, chunks their text, creates embeddings using Gemini,
    and saves the processed data.
    """
    try:
        provider = GeminiEmbedder()
    except Exception as e:
        print(f"[ingest][fatal] cannot initialize GeminiEmbedder: {e}")
        raise

    # --- 1. Load Metadata and Prepare File List ---
    metadata_path = PDF_FOLDER / ".metadata.json"
    if not metadata_path.exists():
        print(f"[ingest][fatal] .metadata.json not found in {PDF_FOLDER}. Run the download script first.")
        return
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        docs_metadata = json.load(f)
        
    metadata_map = {item['local_filename']: item for item in docs_metadata}
    pdf_files_to_process = [PDF_FOLDER / item['local_filename'] for item in docs_metadata]

    # --- 2. Read PDFs and Create Text Chunks ---
    all_texts = []
    metadata = []
    next_id = 0
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    for pdf_path in tqdm(pdf_files_to_process, desc="Ingesting PDFs"):
        if not pdf_path.exists():
            tqdm.write(f"[ingest][warning] Skipping {pdf_path.name} as it was not found.")
            continue
            
        doc_meta = metadata_map.get(pdf_path.name)
        if not doc_meta:
            tqdm.write(f"[ingest][warning] Skipping {pdf_path.name} as it's not in .metadata.json")
            continue

        try:
            with fitz.open(pdf_path) as doc:
                full_text = "\n".join(page.get_text("text") for page in doc)
            cleaned_text = clean_text(full_text)
        except Exception as e:
            tqdm.write(f"[ingest][error] Failed to read or clean {pdf_path.name}: {e}")
            continue

        chunks = text_splitter.split_text(cleaned_text)

        for chunk_text_str in chunks:
            chunk_meta = {
                "id": next_id,
                "title": doc_meta.get("title", "Unknown Title"),
                "section": doc_meta.get("section", "Uncategorized"),
                "author": doc_meta.get("author", "Unknown Author"),
                "date": doc_meta.get("date", "Unknown Date"),
                "link": doc_meta.get("link", ""),
                "source_filename": pdf_path.name,
            }
            metadata.append(chunk_meta)
            all_texts.append(chunk_text_str)
            next_id += 1

    if not all_texts:
        print("[ingest] No text chunks produced. Exiting.")
        return

    # --- 3. Print Debug Summary ---
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "50")) # Uses the safer default of 50
    
    print("\n" + "="*50)
    print("--- PRE-EMBEDDING DEBUG SUMMARY ---")
    print(f"Total documents processed: {len(pdf_files_to_process)}")
    print(f"Total text chunks created: {len(all_texts)}")
    print(f"Embedding batch size: {batch_size}")
    num_api_calls = math.ceil(len(all_texts) / batch_size)
    print(f"Estimated API calls to be made: {num_api_calls}")
    print("--- SAMPLE CHUNK (First chunk to be embedded) ---")
    print(all_texts[0][:500] + "...")
    print("="*50 + "\n")

    # --- 4. Create Embeddings with a Delay ---
    embeddings_list = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding Chunks"):
        batch = all_texts[i:i + batch_size]
        try:
            emb_batch = provider.embed_texts(batch)
            embeddings_list.append(np.asarray(emb_batch, dtype=np.float32))
            # time.sleep(2)  # Cautious 2-second delay between each API call
        except Exception as e:
            print(f"[ingest][fatal] embedding failed for batch starting at {i}: {e}")
            raise

    # --- 5. Save Processed Data ---
    embeddings_arr = np.vstack(embeddings_list)
    emb_path = CHUNKS_DIR / "embeddings.npy"
    np.save(emb_path, embeddings_arr)

    chunks_out = CHUNKS_DIR / "chunks.jsonl"
    with open(chunks_out, "w", encoding="utf-8") as f:
        for m, t in zip(metadata, all_texts):
            m_with_text = m.copy()
            m_with_text["text"] = t
            f.write(json.dumps(m_with_text, ensure_ascii=False) + "\n")

    print(f"✅ [ingest] Saved {len(metadata)} chunks and embeddings to {CHUNKS_DIR}")


if __name__ == "__main__":
    ingest_all()