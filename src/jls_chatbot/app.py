# src/jls_chatbot/app.py
import sys
import os
import logging
from pathlib import Path
import time
from dotenv import load_dotenv

# --- ✨ FINAL FIX: Robust Path and Import Setup ---
# This block makes the app runnable by ensuring Python knows where the 'src' directory is.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
# --- END OF FIX ---

# This now loads the .env file from the project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

import streamlit as st
# This import will now work correctly
from jls_chatbot.core.rag_chain import answer_query

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("jls_chatbot_app")
logger.info("Starting Streamlit SOP RAG Chatbot")

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="SOP RAG Chatbot")
st.title("JLS Company SOP Chatbot 🤖")

# --- Sidebar ---
with st.sidebar:
    st.header("Options")
    
    st.markdown("""
    **Relevance Threshold** determines how similar a document must be to your question to be included. 
    A higher value means stricter matching.
    """)
    relevance_threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.75, 0.05)

    st.markdown("---")
    st.info("To update the knowledge base, run the data pipeline scripts locally, then push the updated `data` folder to GitHub.")

    if st.button("Clear chat history"):
        st.session_state.history = []
        st.success("Chat history cleared.")
        st.rerun()

# --- Session state initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Input area ---
if query := st.chat_input("Ask a question about our SOPs..."):
    with st.spinner("Searching for relevant SOPs and generating an answer..."):
        try:
            logger.info("Answering query: %s", query)
            search_kwargs = {'score_threshold': relevance_threshold}
            res = answer_query(query, search_type="similarity_score_threshold", search_kwargs=search_kwargs)
            
            st.session_state.history.append({
                "query": query,
                "answer": res.get("answer", "No answer found."),
                "sources": res.get("sources", []),
                "ts": time.time(),
            })
            st.rerun()
        except Exception as e:
            logger.exception("Error during answer_query")
            st.error(f"An error occurred: {e}")


# --- Conversation display ---
for turn in st.session_state.history:
    with st.chat_message(name="user", avatar="👤"):
        st.write(turn["query"])
        
    with st.chat_message(name="assistant", avatar="🤖"):
        st.markdown(turn["answer"])
        
        sources = turn.get("sources", [])
        if sources:
            st.markdown("**Sources Found:**")
            for i, s in enumerate(sources):
                title = s.get("title", "Unknown Title")
                link = s.get("link", "#")
                section = s.get("section", "N/A")
                snippet = s.get("snippet", "...")

                with st.expander(f"**{i+1}. {title}** (Section: *{section}*)"):
                    st.markdown(f"**Source Link:** [{title}]({link})")
                    st.markdown(f"**Snippet:**\n>{snippet}")

if not st.session_state.history:
    st.info("Ask a question to get started with the JLS SOP Assistant!")