import numpy as np
from src.embedder import get_preferred_embedder
arr = np.load("./data/chunks/embeddings.npy")
print("precomputed shape:", arr.shape)
emb = get_preferred_embedder("gemini")
v = emb.embed_query("dimension-check")
print("query vector len:", len(v))
