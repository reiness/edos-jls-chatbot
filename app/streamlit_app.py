import sys
import os
import logging
from pathlib import Path
import hashlib
import time
import base64
import io
from dotenv import load_dotenv
import re 
import shutil

# --- Ensure project root is on sys.path so `src` imports succeed ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env early so POPPLER_PATH and others are available
load_dotenv()

import streamlit as st
from src.rag_chain import answer_query, build_or_load_vectorstore

# --- Logging to console + file for debugging ---
LOG_FILE = ROOT / "streamlit_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("sop_rag_chatbot")
logger.info("Starting Streamlit SOP RAG Chatbot")

# Optional PDF->image fallback
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
    logger.info("pdf2image and Pillow available")
except Exception as e:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image or Pillow not available: %s", e)

# App config
st.set_page_config(layout="wide", page_title="SOP RAG Chatbot (LangChain + Gemini)")
st.title("JLS SOP Chatbot")

# Debug: environment snapshot
POPPLER_PATH = os.getenv("POPPLER_PATH")
logger.debug("POPPLER_PATH=%s", POPPLER_PATH)
logger.debug("INDEX_DIR=%s", os.getenv("INDEX_DIR", "./data/index"))
logger.debug("PDF_FOLDER=%s", os.getenv("PDF_FOLDER", "./data/pdfs"))

# --- Sidebar ---
with st.sidebar:
    st.header("Options")
    top_k = st.slider("Number of sources to look", 1, 10, 5)

    if st.button("Rebuild index (force)"):
        with st.spinner("Rebuilding vectorstore (this can take a while)..."):
            try:
                logger.info("User requested index rebuild (force)")
                build_or_load_vectorstore(force_rebuild=True)
                st.success("Index rebuilt successfully.")
            except Exception:
                logger.exception("Index rebuild failed")
                st.error("Failed to rebuild index. See logs for details.")

    st.markdown("---")
    st.write("Index directory:", os.getenv("INDEX_DIR", "./data/index"))
    st.markdown("---")
    if st.button("Clear chat history"):
        st.session_state.history = []
        # st.session_state.view_pdf_key = None # Also clear the view key
        st.success("Chat history cleared.")
        st.rerun()

    st.markdown("---")
    st.write("pdf2image installed:", PDF2IMAGE_AVAILABLE)
    st.write("POPPLER_PATH:", POPPLER_PATH)
    st.markdown("---")

# --- Session state initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

# PDF view feature is hidden, so this state is no longer needed
# if "view_pdf_key" not in st.session_state:
#     st.session_state.view_pdf_key = None

# --- Helper: robust preview renderer with logging (currently not used) ---
def render_pdf_preview(src_path: str, turn_id: str, i: int, safe_hash: str):
    """
    Renders a PDF preview by serving it from a static directory for reliability,
    with fallbacks for image conversion and direct download.
    """
    logger.debug("render_pdf_preview called for: %s", src_path)
    
    # 1. --- Validate the source file ---
    if not src_path or not os.path.exists(src_path):
        logger.warning("File not found on disk: %s", src_path)
        st.error("PDF source file not found.")
        return

    file_name = os.path.basename(src_path)
    
    # 2. --- Serve the file from a static directory (Primary Method) ---
    try:
        # Define the static directory relative to the running script
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)

        # Sanitize the filename to be URL-friendly
        safe_file_name = re.sub(r'[^\w\.\-]', '_', file_name)
        unique_file_name = f"{safe_hash}_{safe_file_name}"
        
        static_file_path = static_dir / unique_file_name

        # Copy the file only if it's not already there to save I/O
        if not static_file_path.exists():
            shutil.copy(src_path, static_file_path)
            logger.info("Copied '%s' to static dir as '%s'", src_path, unique_file_name)
        
        # Generate the URL that Streamlit provides for the static file
        file_url = f"/static/{unique_file_name}"

        # Display the preview in an iframe
        st.write(f"Previewing: {file_name}")
        iframe = f'<iframe src="{file_url}" width="100%" height="800" style="border:none;" title="PDF Preview"></iframe>'
        st.components.v1.html(iframe, height=800, scrolling=True)
        logger.info("Successfully rendered iframe with static URL for: %s", file_name)
        return # Success, so we exit the function

    except Exception as e:
        logger.exception("Primary iframe method failed for %s. Moving to fallbacks.", src_path)
        st.warning("Browser preview failed. Trying image fallback...")

    # 3. --- Fallback to Image Rendering (pdf2image) ---
    try:
        file_bytes = Path(src_path).read_bytes()
        
        if PDF2IMAGE_AVAILABLE:
            logger.debug("Attempting pdf2image conversion for %s", file_name)
            poppler_path = POPPLER_PATH or None
            images = convert_from_bytes(
                file_bytes, first_page=1, last_page=1, dpi=150, poppler_path=poppler_path
            )
            if images:
                img = images[0]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.image(buf, caption=f"First page of {file_name}", use_column_width=True)
                logger.info("Successfully rendered first page using pdf2image for %s", file_name)
        else:
            logger.warning("pdf2image not available. Skipping image fallback.")
            st.info("PDF preview is not available in this browser. Please download the file.")

    except Exception as e:
        logger.exception("pdf2image fallback failed for %s", src_path)
        st.error("Could not render the PDF as an image.")
    
    # 4. --- Final Fallback: Direct Download ---
    st.write("You can download the file to view it locally.")
    try:
        file_bytes = Path(src_path).read_bytes()
        st.download_button(
            label=f"Download {file_name}",
            data=file_bytes,
            file_name=file_name,
            key=f"dl_final_fallback_{turn_id}_{i}_{safe_hash}",
        )
    except Exception as e:
        logger.exception("Failed to create final download button for %s", src_path)
        st.error(f"Could not prepare {file_name} for download.")


# --- New: snippet cleaner ---
def complete_sentence(snippet: str) -> str:
    """Ensure snippet ends with a sentence for display."""
    if not snippet:
        return snippet
    snippet = snippet.strip()
    if snippet.endswith(('.', '!', '?', '."', '!"', '?"')):
        return snippet
    return snippet + "..."

# --- Input area ---
query = st.text_input("Ask a question about SOPs", "")
ask_pressed = st.button("Ask")

if ask_pressed and query.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            logger.info("Answering query: %s", query)
            res = answer_query(query, k=top_k)
        except Exception:
            logger.exception("Error during answer_query")
            st.error("Error during query. See logs for details.")
            res = {"answer": "", "sources": []}

    answer = res.get("answer", "")
    sources = res.get("sources", [])

    # When a new question is asked, hide any open PDF.
    # st.session_state.view_pdf_key = None

    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "sources": sources,
        "ts": time.time(),
    })
    # No rerun here, we want the new answer to show up naturally.


# --- Conversation display ---
history = st.session_state.history
if not history:
    st.info("No conversation yet. Ask a question to get started.")
else:
    # Iterate backwards to show the most recent conversation first
    for turn_idx in range(len(history) - 1, -1, -1):
        turn = history[turn_idx]
        q_text = turn.get("query", "")
        a_text = turn.get("answer", "")
        sources = turn.get("sources", []) or []
        ts = turn.get("ts", 0)

        turn_id = hashlib.sha1(f"{q_text}_{ts}".encode()).hexdigest()[:10]

        with st.container():
            st.markdown(f"**Q:** {q_text}")
            st.markdown(f"**A:** {a_text}")

            if sources:
                st.markdown("**Sources:**")
                for i, s in enumerate(sources):
                    src_path = s.get("source")
                    page = s.get("page")
                    snippet = complete_sentence(s.get("snippet") or "")

                    st.markdown(f"{i+1}. `{src_path}` — page: {page}")
                    st.write(snippet)

                    abs_path = os.path.abspath(src_path) if src_path else ""
                    safe_hash = hashlib.sha1(abs_path.encode()).hexdigest()[:12]
                    
                    # --- PDF VIEW FEATURE HIDDEN ---
                    # All logic for the "View PDF" button and rendering is removed.
                    # Only the download button remains.
                    
                    if src_path and os.path.exists(src_path):
                        try:
                            with open(src_path, "rb") as f:
                                file_bytes = f.read()
                            file_name = os.path.basename(src_path)
                            st.download_button(
                                label=f"Download {file_name}",
                                data=file_bytes,
                                file_name=file_name,
                                key=f"download_{turn_id}_{i}_{safe_hash}",
                            )
                        except Exception:
                            logger.exception("Unable to prepare download for %s", src_path)
                            st.error(f"Unable to prepare download for {src_path}")
                    else:
                        st.write("PDF not available.")

        st.markdown("---")


st.caption("Tip: If you switch embedding models or re-run ingestion, press 'Rebuild index (force)' in the sidebar.")

