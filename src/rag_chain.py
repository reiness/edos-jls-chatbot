# src/rag_chain.py
import os
from pathlib import Path
import json
import numpy as np  # Make sure numpy is imported
from typing import Any

# LangChain v0.1+ and google-genai v0.8.0+ imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough

# --- Environment and Path Setup ---
from dotenv import load_dotenv
load_dotenv()

INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data/index"))
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "./data/chunks"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# --- Prompt Template ---
PROMPT_TEMPLATE = """
You are an assistant that answers user questions using ONLY the provided context.
Each context segment originates from a SOP. After your concise answer, list the sources for the information.

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely and then provide the sources:
"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)


def format_docs(docs: list[Document]) -> str:
    """Helper function to format retrieved documents into a single string for the prompt."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs
    )


def build_or_load_vectorstore(force_rebuild: bool = False) -> FAISS:
    """
    Build or load FAISS index. This version instantiates the query embedder first
    so we don't get UnboundLocalError when loading an existing index.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    faiss_index_path = INDEX_DIR / "faiss_index"

    # --- CREATE query embedder upfront (avoid scoping issues) ---
    embeddings_for_query = None
    # Preferred: your local embedder helper (Gemini)
    try:
        from src.embedder import get_preferred_embedder
        embeddings_for_query = get_preferred_embedder(prefer="gemini")
        print("Using preferred local embedder (gemini) for queries.")
    except Exception as e_pref:
        # Fallback to Google generative embeddings (or another embedder you want)
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings_for_query = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=api_key
            )
            print("Falling back to GoogleGenerativeAIEmbeddings for queries.")
        except Exception as e_fallback:
            raise RuntimeError(
                "Failed to create a query embedder. Tried local get_preferred_embedder and "
                "GoogleGenerativeAIEmbeddings."
            ) from e_fallback

    # If index exists and not forcing rebuild, load it using the embedder instance
    if faiss_index_path.exists() and not force_rebuild:
        print("Loading existing FAISS index from disk.")
        return FAISS.load_local(
            str(faiss_index_path),
            embeddings_for_query,
            allow_dangerous_deserialization=True
        )

    # --- Build from precomputed files ---
    print("Building new FAISS index from pre-computed embeddings.")
    chunks_file = CHUNKS_DIR / "chunks.jsonl"
    embeddings_file = CHUNKS_DIR / "embeddings.npy"
    if not chunks_file.exists() or not embeddings_file.exists():
        raise FileNotFoundError(f"Source files missing. Run ingest.py first. Missing: {chunks_file} or {embeddings_file}")

    # Load texts & metadata
    texts = []
    metadatas = []
    with open(chunks_file, "r", encoding="utf-8") as fin:
        for line in fin:
            m = json.loads(line)
            metadatas.append({"source": m.get("source", "unknown"), "page": m.get("page", 0)})
            texts.append(m.get("text", ""))

    precomputed_vectors = np.load(embeddings_file).astype("float32")
    if len(texts) != precomputed_vectors.shape[0]:
        raise ValueError(f"Mismatch between number of texts ({len(texts)}) and number of embeddings ({precomputed_vectors.shape[0]}).")

    d = precomputed_vectors.shape[1]
    print(f"Precomputed embeddings shape: {precomputed_vectors.shape} (dim = {d})")

    # Optional: sanity-check that the query embedder produces same dim (one small API call)
    try:
        sample_vec = embeddings_for_query.embed_query("sanity-check")
        if len(sample_vec) != d:
            raise ValueError(
                f"Embedding dimension mismatch: precomputed dim={d}, query embedder dim={len(sample_vec)}."
            )
        print(f"Query embedder dimensionality verified: {len(sample_vec)}")
    except Exception as e:
        raise RuntimeError(f"Failed to verify query embedder dimension or mismatch: {e}") from e

    text_embedding_pairs = list(zip(texts, precomputed_vectors))
    vs = FAISS.from_embeddings(text_embedding_pairs, embeddings_for_query, metadatas=metadatas)
    print("Saving new FAISS index to disk.")
    vs.save_local(str(faiss_index_path))
    return vs



def make_qa_chain(k: int = 5, force_rebuild_index: bool = False):
    """Creates a modern RAG chain using LangChain Expression Language (LCEL)."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
    
    vectorstore = build_or_load_vectorstore(force_rebuild=force_rebuild_index)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    return retriever, rag_chain

def answer_query(question: str, k: int = 5, force_rebuild_index: bool = False) -> dict[str, Any]:
    """Runs the QA chain and returns a dictionary with the answer and sources."""
    retriever, qa_chain = make_qa_chain(k=k, force_rebuild_index=force_rebuild_index)
    
    answer_text = qa_chain.invoke(question)
    source_docs = retriever.invoke(question)
    
    sources = []
    for doc in source_docs:
        md = doc.metadata or {}
        sources.append({
            "source": md.get("source", "unknown"),
            "page": md.get("page", None),
            "snippet": doc.page_content[:300].replace("\n", " ")
        })
        
    return {"answer": answer_text, "sources": sources}