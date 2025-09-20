# src/jls_chatbot/app.py
import sys
import os
import logging
from pathlib import Path
import time
import json
import collections
from dotenv import load_dotenv
import streamlit as st

# --- Robust Path and Import Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from jls_chatbot.core.rag_chain import answer_query

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("jls_chatbot_app")

# --- Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    if st.session_state.get("password_correct", False):
        return True

    correct_password = st.secrets.get("PASSWORD")
    if not correct_password:
        st.error("Password not configured. Please contact an administrator.")
        return False

    with st.form("password_form"):
        st.title("JLS Company SOP Chatbot 🤖")
        st.markdown("---")
        password = st.text_input("Please enter the password to continue", type="password")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if password == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("The password you entered is incorrect.")
    return False

# --- Page Rendering Functions ---

def render_chatbot_page():
    """Renders the main chatbot interface."""
    st.title("JLS Company SOP Chatbot 🤖")

    with st.sidebar:
        st.header("Chatbot Options")
        st.markdown("""
        **Relevance Threshold** determines how similar a document must be to your question to be included. 
        A higher value means stricter matching.
        """)
        relevance_threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.35, 0.05)
        st.markdown("---")
        if st.button("Clear chat history"):
            st.session_state.history = []
            st.success("Chat history cleared.")
            st.rerun()

    if "history" not in st.session_state:
        st.session_state.history = []

    if query := st.chat_input("Ask a question about our SOPs..."):
        with st.spinner("Searching for relevant SOPs and generating an answer..."):
            try:
                search_kwargs = {'score_threshold': relevance_threshold}
                res = answer_query(query, search_type="similarity_score_threshold", search_kwargs=search_kwargs)
                st.session_state.history.append({
                    "query": query, "answer": res.get("answer", "No answer found."),
                    "sources": res.get("sources", []), "ts": time.time(),
                })
                st.rerun()
            except Exception as e:
                logger.exception("Error during answer_query")
                st.error(f"An error occurred: {e}")

    # --- Conversation display ---
    # ... (rest of the chatbot display logic is the same)
    for turn in st.session_state.history:
        with st.chat_message(name="user", avatar="👤"):
            st.write(turn["query"])
        with st.chat_message(name="assistant", avatar="🤖"):
            st.markdown(turn["answer"])
            sources = turn.get("sources", [])
            if sources:
                st.markdown("**Sources Found:**")
                for i, s in enumerate(sources):
                    with st.expander(f"**{i+1}. {s.get('title', 'Unknown Title')}** (Section: *{s.get('section', 'N/A')}*)"):
                        st.markdown(f"**Source Link:** [{s.get('title', 'Unknown Title')}]({s.get('link', '#')})")
                        st.markdown(f"**Snippet:**\n>{s.get('snippet', '...')}")

    if not st.session_state.history:
        st.info("Ask a question to get started!")


def render_intro_page():
    # ... (function content is the same)
    st.header("Introduction 👋")
    st.markdown("""
    Hey there! My name is **Edo Afrianto**, an intern from Indonesia, and I'm the one who built this chatbot. I really love doing stuff like this—messing with AI, building tools, and trying to make our work life a little bit easier.

    This whole project is running on Streamlit's free community cloud, so I'm paying exactly **$0 in development costs**, except for my electricity bill, haha.

    I know this bot isn't perfect yet, but the goal is to make it a genuinely helpful assistant for everyone, especially when you're curious about a process or forget a specific detail in an SOP.

    To make it better, I'd really appreciate your feedback! If you have a second, please fill out this Google Form. Your help will be a huge part of improving this tool for everyone. Thank you!
    
    *https://forms.gle/DTULrdagAYo2ehhi7*
    """)

def render_updates_page():
    # ... (function content is the same)
    st.header("Possible Future Updates 🚀")
    st.markdown("""
    Right now, the bot is still in a **beta version** because it doesn't cover all of our SOPs just yet. Most of the documents in its knowledge base are the ones that new interns typically read during onboarding. But, in the future, I'll be working on:

    #### 1. Cover More SOPs
    The top priority is to ingest SOPs from all other departments to make the chatbot a comprehensive resource for the entire company.

    #### 2. Cover Flashcards
    This is a bit tricky and will take some time. The flashcard documents use a very different format than the SOPs, so they'll require a custom data processing approach.

    #### 3. Make it a Chrome Extension
    This is the long-term dream! It would involve thinking about a proper backend and how to deploy it with minimal cost. I don't want to take the easy way out and just rent a server, because the operational cost would bleed us dry as the company grows. The goal is to find a smart, scalable, and cost-effective solution.
    """)

def render_knowledge_base_page():
    # ... (function content is the same)
    st.header("Current Knowledge Base 📚")
    metadata_path = PROJECT_ROOT / "data" / "source_documents" / ".metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            docs_metadata = json.load(f)
        sections = collections.defaultdict(list)
        for doc in docs_metadata:
            sections[doc.get("section", "Uncategorized")].append(doc)
        st.write(f"The chatbot currently has knowledge of **{len(docs_metadata)}** documents across **{len(sections)}** sections.")
        st.markdown("---")
        for section_name, docs in sorted(sections.items()):
            with st.expander(f"**{section_name}** ({len(docs)} documents)"):
                for doc in sorted(docs, key=lambda x: x['title']):
                    st.markdown(f"- [{doc.get('title')}]({doc.get('link')})")
    else:
        st.error("The .metadata.json file was not found. Please run the data pipeline first.")


# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="SOP RAG Chatbot")

if check_password():
    
    # --- ✨ NEW SIDEBAR NAVIGATION ---
    # Initialize session state for the page if it doesn't exist
    if "page" not in st.session_state:
        st.session_state.page = "Chatbot"

    # Get the current page from session state
    page = st.session_state.page
    
    st.sidebar.title("Navigation")

    # Use button type to highlight the active page
    if st.sidebar.button("Chatbot", use_container_width=True, type="primary" if page == "Chatbot" else "secondary"):
        st.session_state.page = "Chatbot"
        st.rerun()

    if st.sidebar.button("Introduction", use_container_width=True, type="primary" if page == "Introduction" else "secondary"):
        st.session_state.page = "Introduction"
        st.rerun()
        
    if st.sidebar.button("Future Updates", use_container_width=True, type="primary" if page == "Future Updates" else "secondary"):
        st.session_state.page = "Future Updates"
        st.rerun()

    if st.sidebar.button("Knowledge Base", use_container_width=True, type="primary" if page == "Knowledge Base" else "secondary"):
        st.session_state.page = "Knowledge Base"
        st.rerun()

    st.sidebar.markdown("---")

    # --- Page Content ---
    if st.session_state.page == "Chatbot":
        render_chatbot_page()
    elif st.session_state.page == "Introduction":
        render_intro_page()
    elif st.session_state.page == "Future Updates":
        render_updates_page()
    elif st.session_state.page == "Knowledge Base":
        render_knowledge_base_page()