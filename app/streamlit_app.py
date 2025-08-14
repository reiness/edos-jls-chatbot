import sys
import os
import logging
from pathlib import Path
import hashlib
import time
import base64
import io
from dotenv import load_dotenv

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
st.title("SOP RAG Chatbot — Enhanced")

# Debug: environment snapshot
POPPLER_PATH = os.getenv("POPPLER_PATH")
logger.debug("POPPLER_PATH=%s", POPPLER_PATH)
logger.debug("INDEX_DIR=%s", os.getenv("INDEX_DIR", "./data/index"))
logger.debug("PDF_FOLDER=%s", os.getenv("PDF_FOLDER", "./data/pdfs"))

# --- Sidebar ---
with st.sidebar:
    st.header("Options")
    top_k = st.slider("Top K", 1, 10, 5)

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
        st.success("Chat history cleared.")

    st.markdown("---")
    st.write("pdf2image installed:", PDF2IMAGE_AVAILABLE)
    st.write("POPPLER_PATH:", POPPLER_PATH)
    st.markdown("---")

# --- Session state initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

if "view_pdf_key" not in st.session_state:
    st.session_state.view_pdf_key = None

# --- Helper: robust preview renderer with logging ---
def render_pdf_preview(src_path: str, turn_id: str, i: int, safe_hash: str):
    """Render a PDF preview with detailed logging. Writes logs to console + file."""
    logger.debug("render_pdf_preview called: %s", src_path)
    if not src_path or not os.path.exists(src_path):
        logger.warning("render_pdf_preview: file not found: %s", src_path)
        st.error("PDF not found on disk.")
        return

    try:
        file_bytes = Path(src_path).read_bytes()
        logger.info("Read %d bytes from %s", len(file_bytes), src_path)

        # Quick magic check for PDF header
        header = file_bytes[:8]
        try:
            header_text = header.decode(errors="replace")
        except Exception:
            header_text = str(header)
        logger.debug("File header (first 8 bytes): %s", header_text)
        if not header.startswith(b"%PDF"):
            logger.warning("File does not start with %%PDF signature (may not be a PDF)")
            st.warning("File does not appear to be a valid PDF. You can download to inspect it.")

        file_size_kb = len(file_bytes) / 1024
        file_size_mb = file_size_kb / 1024
        file_name = os.path.basename(src_path)

        st.write(f"Previewing: {file_name} — {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")
        logger.debug("Previewing file %s size: %.2f MB", file_name, file_size_mb)

        MAX_EMBED_MB = 10.0
        if file_size_mb > MAX_EMBED_MB:
            logger.info("File too large to embed (%s MB). Offering download.", file_size_mb)
            st.warning(f"File is large ({file_size_mb:.1f} MB). Embedding in browser may fail. Please download or open locally.")
            st.download_button(
                label=f"Download {file_name}",
                data=file_bytes,
                file_name=file_name,
                key=f"dl_fallback_{turn_id}_{i}_{safe_hash}",
            )
            return

        # Build base64 data URI
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        iframe = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800" style="border:none;"></iframe>'

        embed_failed = False
        try:
            logger.debug("Attempting to render iframe embed for %s", file_name)
            st.components.v1.html(iframe, height=800, scrolling=True)
            logger.info("Iframe embed succeeded for %s", file_name)
        except Exception as e_iframe:
            embed_failed = True
            logger.exception("Iframe embed failed for %s", file_name)
            # Fallback to embed tag
            try:
                embed = f'<embed src="data:application/pdf;base64,{b64}" width="100%" height="800" type="application/pdf">'
                st.components.v1.html(embed, height=800)
                logger.info("Embed tag succeeded for %s", file_name)
                embed_failed = False
            except Exception as e_embed:
                logger.exception("Embed tag also failed for %s", file_name)
                st.write("Preview failed in browser. You can download the file below.")
                st.write("Preview exceptions:")
                st.write(str(e_iframe))
                st.write(str(e_embed))

        # If embedding failed, show Open in new tab anchor (some browsers handle it better)
        if embed_failed:
            try:
                open_in_tab_html = f'<a href="data:application/pdf;base64,{b64}" target="_blank" rel="noopener">Open PDF in new tab</a>'
                st.components.v1.html(open_in_tab_html, height=40)
                logger.debug("Rendered Open-in-new-tab anchor for %s", file_name)
            except Exception:
                logger.exception("Failed to render Open-in-new-tab anchor")
                st.markdown(f'[Open PDF in new tab](data:application/pdf;base64,{b64})')

        # If embedding failed and pdf2image is available, try rasterizing first page
        if embed_failed and PDF2IMAGE_AVAILABLE:
            try:
                poppler_path = POPPLER_PATH or None
                logger.debug("Attempting pdf2image conversion. poppler_path=%s", poppler_path)
                images = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=150, poppler_path=poppler_path)
                img = images[0]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                st.image(buf, use_column_width=True)
                st.write("(Rendered first page as an image using pdf2image)")
                logger.info("pdf2image rendered first page for %s", file_name)
            except Exception:
                logger.exception("pdf2image conversion failed for %s", file_name)
                st.write("pdf2image conversion failed. See logs for details.")
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_bytes,
                    file_name=file_name,
                    key=f"dl_fallback_{turn_id}_{i}_{safe_hash}",
                )
        elif embed_failed and not PDF2IMAGE_AVAILABLE:
            logger.warning("Embedding failed and pdf2image not installed")
            st.write("Embedding failed and pdf2image is not installed. To enable image fallback install pdf2image and poppler.")
            st.download_button(
                label=f"Download {file_name}",
                data=file_bytes,
                file_name=file_name,
                key=f"dl_fallback_{turn_id}_{i}_{safe_hash}",
            )

    except Exception:
        logger.exception("Unexpected error while rendering preview for %s", src_path)
        st.error("Failed to render PDF preview. See logs for details.")
        try:
            st.download_button(
                label=f"Download {os.path.basename(src_path)}",
                data=Path(src_path).read_bytes() if os.path.exists(src_path) else b"",
                file_name=os.path.basename(src_path),
                key=f"dl_error_{turn_id}_{i}_{safe_hash}",
            )
        except Exception:
            logger.exception("Failed to provide download fallback")


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

    # Attach timestamp for stable keys and ordering
    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "sources": sources,
        "ts": time.time(),
    })

# --- Conversation display (most recent first) ---
history = st.session_state.history
if not history:
    st.info("No conversation yet. Ask a question to get started.")
else:
    for turn_idx in range(len(history) - 1, max(-1, len(history) - 11), -1):
        turn = history[turn_idx]
        q_text = turn.get("query", "")
        a_text = turn.get("answer", "")
        sources = turn.get("sources", []) or []
        ts = turn.get("ts", 0)

        turn_id = hashlib.sha1(f"{q_text}_{ts}".encode()).hexdigest()[:10]

        st.markdown(f"**Q:** {q_text}")
        st.markdown(f"**A:** {a_text}")

        if sources:
            st.markdown("**Sources:**")
            for i, s in enumerate(sources):
                src_path = s.get("source")
                page = s.get("page")
                snippet = s.get("snippet") or ""

                st.markdown(f"{i+1}. `{src_path}` — page: {page}")
                st.write(snippet)

                col_view, col_download = st.columns([1, 4])

                abs_path = os.path.abspath(src_path) if src_path else ""
                safe_hash = hashlib.sha1(abs_path.encode()).hexdigest()[:12]
                view_key = f"view_{turn_id}_{i}_{safe_hash}"
                download_key = f"download_{turn_id}_{i}_{safe_hash}"

                with col_view:
                    if src_path and os.path.exists(src_path):
                        if st.button(f"View PDF {i+1}", key=view_key):
                            st.session_state.view_pdf_key = view_key
                            logger.debug("View button pressed: %s (turn %s index %d)", src_path, turn_id, i)
                    else:
                        st.write("PDF not found")

                with col_download:
                    if src_path and os.path.exists(src_path):
                        try:
                            file_bytes = Path(src_path).read_bytes()
                            file_name = os.path.basename(src_path)
                            st.write(f"{file_name} — {len(file_bytes)/1024:.1f} KB")
                            st.download_button(
                                label=f"Download {file_name}",
                                data=file_bytes,
                                file_name=file_name,
                                key=download_key,
                            )
                        except Exception:
                            logger.exception("Unable to prepare download for %s", src_path)
                            st.error(f"Unable to prepare download for {src_path}")
                    else:
                        st.write("PDF not available for download.")

                # Render preview if requested
                if st.session_state.view_pdf_key == view_key:
                    render_pdf_preview(src_path, turn_id, i, safe_hash)

        st.markdown("---")

# --- Footer / small help ---
st.caption("Tip: If you switch embedding models or re-run ingestion, press 'Rebuild index (force)' in the sidebar.")
