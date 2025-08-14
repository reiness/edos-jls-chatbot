# scripts/test_gemini.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedder import GeminiEmbedder

def test():
    ge = GeminiEmbedder()
    arr = ge.embed_texts(["hello world"])
    print("OK. Embedding shape:", getattr(arr, "shape", None))
    print("First 6 values:", arr[0][:6].tolist())

if __name__ == "__main__":
    test()
