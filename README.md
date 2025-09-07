# SOP RAG Chatbot (LangChain + Gemini + Streamlit)

A Retrieval-Augmented Generation (RAG) chatbot designed for querying thousands of company SOPs quickly and accurately.  
Built with **LangChain**, **Google Gemini API**, **FAISS vector search**, and **Streamlit** for an interactive web UI.

---

## Features

- **RAG-based QA** using company SOP PDFs as a knowledge base.
- **FAISS Vectorstore** for fast similarity search.
- **Gemini API** for both embedding and answer generation.
- **Interactive Streamlit UI** with:
  - Adjustable `Top K` retrieval slider.
  - PDF preview (inline, new tab, or image fallback via `pdf2image`).
  - Downloadable source PDFs.
  - Chat history display and reset.
  - Force index rebuild button.
- **Detailed logging** to help debug embedding, retrieval, and PDF viewing issues.

---

## 1. Prerequisites

- **Python 3.10+**
- **pip** and **virtualenv**
- **Google Gemini API Key** (or Vertex AI setup)
- **Poppler** (required for PDF-to-image preview fallback)
- (Windows) **Visual C++ Redistributable** (for Poppler to work)

---

## 2. Install & Setup

```bash
# Clone the repository
git clone https://github.com/your-org/sop-rag-chatbot.git
cd sop-rag-chatbot

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## 3. Configure Environment Variables
Create a .env file in the project root:

```bash
# --- API Keys ---
GEMINI_API_KEY="your_gemini_api_key"

# --- Models ---
EMBEDDING_MODEL=gemini-embedding-001
GENERATION_MODEL=gemini-2.5-flash

# --- Local directories ---
PDF_FOLDER=./data/pdfs
CHUNKS_DIR=./data/chunks
INDEX_DIR=./data/index
PORT=8501

# --- PDF Preview ---
POPPLER_PATH=C:\\poppler-24.08.0\\Library\\bin
```

Notes:
* POPPLER_PATH must point to the bin folder of Poppler installation.

* On macOS/Linux, you may omit POPPLER_PATH if Poppler is in your system PATH.

## 4. Add SOP PDFs
Put all your SOP PDFs inside the folder specified in .env:

```bash
data/
  pdfs/
    SOP1.pdf
    SOP2.pdf
    ...
```

## 5. Build the Vectorstore
Before first use, you must create the FAISS index:

```bash
.venv\Scripts\activate
python -m scripts.ingest 
python -m scripts.build_index 
```

This will:
1. Read all PDFs from PDF_FOLDER.

2. Extract text, chunk it, and embed it using EMBEDDING_MODEL.

3. Save FAISS index to INDEX_DIR.

## 6. Run the App
```bash
.venv\Scripts\activate
streamlit run app/streamlit_app.py
```
Then open http://localhost:8501 in your browser.


## 7. Using the Chatbot
1. Enter a question about any SOP.

2. Adjust Top K to control how many relevant chunks are retrieved.

3. Click Ask.

4. View sources with View PDF (inline or fallback).

5. Download PDFs with Download.

6. Use Rebuild index (force) in the sidebar if you update or add SOP PDFs.


## 8. Logging
All logs are written to both console and:

```bash
streamlit_debug.log
```
Logs include:
* FAISS search details
* PDF reading and size info
* PDF preview errors and fallbacks
* pdf2image conversion status

