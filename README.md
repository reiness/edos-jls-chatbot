
# SOP RAG Chatbot (LangChain + Gemini + Streamlit)

A production-ready, Retrieval-Augmented Generation (RAG) chatbot designed to accurately answer questions from a large knowledge base of company Standard Operating Procedures (SOPs).

This project is built with a modern, modular Python structure and features an end-to-end data pipeline that automatically downloads, processes, and indexes documents from a Google Sheet. The interactive front-end is built with **Streamlit**.

-----

## ✨ Key Features

  - **Automated Data Pipeline:** Downloads documents directly from a Google Sheet, processes them, and builds the knowledge base with a series of simple commands.
  - **Metadata-Driven Ingestion:** Uses a `.metadata.json` file to enrich the knowledge base with crucial context like document titles, authors, and sections.
  - **Advanced Semantic Chunking:** Employs `RecursiveCharacterTextSplitter` to create meaningful, coherent text chunks that respect paragraph and sentence boundaries.
  - **Context-Enriched Embeddings:** Injects metadata (title, section, etc.) into the text before embedding, leading to highly accurate document retrieval.
  - **Smart Retrieval:** Uses a **Relevance Score Threshold** instead of a fixed number of sources, ensuring that only the most relevant information is used to generate answers.
  - **Clean Streamlit UI:** A user-friendly interface for asking questions, adjusting retrieval settings, and exploring rich, clickable source citations.

-----

## 📂 Project Structure

The project is organized into a standard Python package structure for clarity and scalability.

```
JLS-Chatbot/
│
├── .streamlit/
│   └── secrets.toml         # For deployment secrets
│
├── data/
│   ├── source_documents/    # Raw downloaded PDFs and their metadata
│   └── processed/           # The output of your data pipeline
│
├── src/
│   └── jls_chatbot/         # The main Python package
│       ├── pipeline/        # Data pipeline scripts
│       ├── core/            # Core components (RAG chain, embedder)
│       └── app.py           # The Streamlit app
│
├── .env                     # Local secrets and config
├── .gitignore
├── README.md
└── requirements.txt
```

-----

## 🛠️ Step 1: Installation & Setup

#### Prerequisites

  - Python 3.10+
  - Git
  - A Google Account with access to the target Google Sheet.

#### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/jls-chatbot.git
cd jls-chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

#### Configuration

Create a `.env` file in the project root by copying `.env.example`. This file is for **local development only**.

```dotenv
# .env

# --- API Keys ---
# Used by all scripts that call the Gemini API
GEMINI_API_KEY="your_gemini_api_key_here"

# --- Data Directories (defaults are recommended) ---
# Your downloader script will create and place files here
PDF_FOLDER=./data/source_documents
# Your pipeline will create and place processed files here
CHUNKS_DIR=./data/processed/chunks
INDEX_DIR=./data/processed/index
```

-----

## ⚙️ Step 2: The Data Pipeline Workflow

This three-step process converts your documents from a Google Sheet into a searchable knowledge base. Run these commands from the **project's root directory**.

#### 2.1: Download Documents

This script connects to Google Sheets, downloads all linked Google Docs as PDFs, and creates the crucial `.metadata.json` file.

**First-Time Setup:** Place your `credentials.json` file (from Google Cloud) in the project's root directory before the first run. The script will guide you through a one-time browser authentication.

```bash
python -m src.jls_chatbot.pipeline.download
```

#### 2.2: Ingest & Embed Documents

This script reads `.metadata.json`, processes each PDF, chunks the text, and calls the Gemini API to create embeddings for each chunk.

```bash
python -m src.jls_chatbot.pipeline.ingest
```

#### 2.3: Build the Search Index

This final step takes the chunks and embeddings and builds the fast, searchable FAISS vector index.

```bash
python -m src.jls_chatbot.pipeline.build_index
```

Your knowledge base is now ready\! You only need to re-run this pipeline when your SOPs change.

-----

## 🚀 Step 3: Running the Chatbot

Launch the Streamlit web application with the following command from the project's root directory:

```bash
streamlit run src/jls_chatbot/app.py
```

Navigate to the local URL provided in your terminal to start asking questions\!

-----

## ☁️ Step 4: Deployment

You can deploy this app for free on Streamlit Community Cloud.

1.  **Create a `secrets.toml` file:** In a new `.streamlit` folder, create a `secrets.toml` file containing your `GEMINI_API_KEY`. This file should be listed in your `.gitignore`.
2.  **Push to GitHub:** Push your entire project—including the populated `data/processed` folder—to a public GitHub repository.
3.  **Deploy:** Go to [share.streamlit.io](https://share.streamlit.io/), link your GitHub account, and deploy the repository. Paste the contents of your `secrets.toml` file into the app's secrets manager in the Streamlit dashboard.