# src/jls_chatbot/core/rag_chain.py
import os
from pathlib import Path
from typing import Any

# LangChain v0.1+ imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from dotenv import load_dotenv

# --- Define Project Root and Data Paths Robustly ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "data" / "processed" / "index"))
if not INDEX_DIR.exists():
    raise FileNotFoundError(f"Index directory not found at {INDEX_DIR}. Please run build_index.py first.")

from jls_chatbot.core.embedder import GeminiEmbedder

# Load .env from the project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# --- PERFECTED PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """
You are an expert assistant for our company. Your primary goal is to provide comprehensive and detailed answers to questions based ONLY on the provided context from the company's Standard Operating Procedures (SOPs).

Your instructions are:
1. Synthesize information from all provided context snippets to form a complete and thorough response.
2. If the question asks "how to" do something, provide a clear, step-by-step guide.
3. Answer the user's question fully. Do not leave out details if they are present in the context. Your answer should be detailed, not overly concise.
4. If the context does not contain the answer, you must state that the information is not available in the provided SOPs. Do not use any external knowledge.
5. After providing your detailed answer, you MUST cite the source documents you used. List each source on a new line with its title and a direct link.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
[Your detailed, step-by-step answer based only on the context]

SOURCES:
- [Source Document Title](Link)
- [Source Document Title](Link)
"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved documents into a single string for the prompt, using the clean original text."""
    # ... (This function is correct, no changes needed)
    formatted_sources = []
    for doc in docs:
        meta = doc.metadata
        source_str = (
            f"Source Document: {meta.get('title', 'Unknown Title')}\n"
            f"Section: {meta.get('section', 'Uncategorized')}\n"
            f"Author: {meta.get('author', 'Unknown')}, Date: {meta.get('date', 'Unknown')}\n"
            f"Content Snippet: {meta.get('original_text', '')}"
        )
        formatted_sources.append(source_str)
    return "\n\n---\n\n".join(formatted_sources)


def load_vectorstore() -> FAISS:
    """Loads the pre-built FAISS index from disk."""
    # ... (This function is correct, no changes needed)
    faiss_index_path = INDEX_DIR / "faiss_index"
    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}. Run build_index.py first.")
    embedder = GeminiEmbedder()
    print("Loading existing FAISS index from disk.")
    return FAISS.load_local(
        str(faiss_index_path),
        embedder.get_langchain_embedder(),
        allow_dangerous_deserialization=True
    )

def make_qa_chain(retriever):
    """Creates the full RAG chain using LangChain Expression Language (LCEL)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or Streamlit secrets.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.1)

    rag_chain_from_docs = (
        {
            "context": RunnableLambda(lambda input_dict: format_docs(input_dict["documents"])),
            "question": lambda input_dict: input_dict["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    # --- ✨ FIX: Replaced the final pipe to a dict with RunnablePassthrough.assign ---
    # This correctly constructs a runnable chain instead of a plain dictionary.
    setup_and_retrieval = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    )
    
    rag_chain_with_source = setup_and_retrieval.assign(
        answer=rag_chain_from_docs,
    )
    
    return rag_chain_with_source

def answer_query(question: str, search_type: str = "similarity", search_kwargs: dict = {"k": 5}) -> dict[str, Any]:
    """Runs the QA chain and returns a dictionary with the answer and sources."""
    # ... (This function is correct, no changes needed)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    qa_chain = make_qa_chain(retriever)
    
    result = qa_chain.invoke(question)
    
    sources = []
    for doc in result.get("documents", []):
        md = doc.metadata or {}
        sources.append({
            "title": md.get("title", "Unknown Title"),
            "section": md.get("section", "Uncategorized"),
            "link": md.get("link", "#"),
            "snippet": md.get("original_text", "")[:350].replace("\n", " ") + "..."
        })
        
    return {"answer": result.get("answer"), "sources": sources}