# src/vectorstore.py
import os
import json
import numpy as np
import faiss
from pathlib import Path

CHUNKS_DIR = os.getenv("CHUNKS_DIR", "./data/chunks")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
os.makedirs(INDEX_DIR, exist_ok=True)

def build_faiss_index(embeddings_path=None, metadata_path=None, index_path=None):
    embeddings_path = embeddings_path or (Path(CHUNKS_DIR) / "embeddings.npy")
    metadata_path = metadata_path or (Path(CHUNKS_DIR) / "chunks.jsonl")
    index_path = index_path or (Path(INDEX_DIR) / "faiss.index")

    embs = np.load(embeddings_path)  # shape (n, d)
    n, d = embs.shape
    # Use inner product on normalized vectors for cosine
    print("Embeddings shape:", embs.shape)

    # Make sure they are float32
    embs = embs.astype('float32')

    # Build index
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index at {index_path}")

    # copy metadata file
    with open(metadata_path, 'r', encoding='utf-8') as fin:
        metadata = [json.loads(line) for line in fin]
    with open(Path(INDEX_DIR) / "metadata.jsonl", "w", encoding="utf-8") as fout:
        for m in metadata:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved metadata.jsonl to {INDEX_DIR}")
