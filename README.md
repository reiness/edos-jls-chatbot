
# JLS SOP Chatbot (v1.0.0-beta)

A production-ready, Retrieval-Augmented Generation (RAG) chatbot designed to accurately answer questions from a large knowledge base of company Standard Operating Procedures (SOPs).

This project features an end-to-end data pipeline that processes documents from Google Docs and a clean, interactive user interface built with Streamlit.

## Table of Contents

  - JLS SOP Chatbot: User Guide
      - How to Use the Chatbot
  - JLS SOP Chatbot: Developer & Admin Guide
      - Key Features
      - Project Structure
      - Step 1: Installation & Setup
      - Step 2: The Data Pipeline Workflow
      - Step 3: Running the Chatbot Locally
      - Step 4: Deployment

-----


# JLS SOP Chatbot: User Guide

This guide is for anyone who wants to ask questions and get answers from the chatbot.


### How to Use the Chatbot

Using the chatbot is simple and secure.

1.  **Access the Chatbot:** When you first visit the app, you will see a login screen. Enter the password provided to you by the administrator and click "Submit" to access the main chat interface.

2.  **Ask a Question:** Once logged in, type any question you have about our company's SOPs into the chat box at the bottom of the screen and press Enter.

3.  **Explore the Sources:** After the chatbot provides an answer, it will list the source documents it used. You can click on the expandable sections to see a relevant snippet from the document and a direct link to the full Google Doc for more details.

4.  **Adjust the Relevance Threshold (Optional):** In the sidebar, you can adjust the "Relevance Threshold."
    * A **higher value** makes the search stricter, which is good for very specific questions.
    * A **lower value** makes the search broader, which can be helpful for more general questions that might require information from multiple sources.

-----

# JLS SOP Chatbot: Developer & Admin Guide

This guide is for engineers or administrators who need to set up, maintain, or deploy the chatbot application to a dedicated server.

## ✨ Key Features

  - **Automated Data Pipeline:** Downloads documents directly from a Google Sheet and builds the knowledge base with a series of simple commands.
  - **Metadata-Driven Ingestion:** Uses a `.metadata.json` file to enrich the knowledge base with crucial context like document titles, authors, and sections.
  - **Advanced Semantic Chunking:** Employs `RecursiveCharacterTextSplitter` to create meaningful, coherent text chunks.
  - **Smart Retrieval:** Uses a Relevance Score Threshold to ensure only the most relevant information is used to generate answers.

## 📂 Project Structure

The project is organized into a standard Python package structure for clarity and scalability.

```
JLS-Chatbot/
│
├── .streamlit/
│   └── secrets.toml         # Only for streamlit deployment secrets
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

#### Installation & Configuration

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

# 4. Configure Environment
# Create a .env file in the project root (you can copy .env.example).
# This is for local development ONLY and should be in your .gitignore.
```

Your `.env` file should contain your `GEMINI_API_KEY` and can be used to override default data paths if needed.

-----

## ⚙️ Step 2: The Data Pipeline Workflow

This three-step process converts your documents into a searchable knowledge base. Run these commands from the **project's root directory**.

#### 2.1: Download Documents

This script connects to Google Sheets and downloads all linked SOPs as PDFs.

**First-Time Setup:** Place your `credentials.json` file (from Google Cloud) in the project's root directory before the first run. The script will guide you through a one-time browser authentication.

```bash
python src/jls_chatbot/pipeline/download.py
```

#### 2.2: Ingest & Embed Documents

This script processes the downloaded PDFs, chunks them, and creates embeddings using the Gemini API.

```bash
python src/jls_chatbot/pipeline/ingest.py
```

#### 2.3: Build the Search Index

This final step creates the fast, searchable FAISS vector index.

```bash
python src/jls_chatbot/pipeline/build_index.py
```

Your knowledge base is now ready\! Re-run this pipeline whenever the source documents are updated.

-----

## 🚀 Step 3: Running the Chatbot Locally

Launch the Streamlit web application with the following command:

```bash
streamlit run src/jls_chatbot/app.py
```

-----

## ☁️ Step 4: Deployment

You can deploy this app for free on Streamlit Community Cloud.

1.  **Prepare Secrets:** Create a `.streamlit/secrets.toml` file locally and add your `GEMINI_API_KEY` to it. **Add this file to `.gitignore`.**
2.  **Push to GitHub:** Push your entire project—including the populated `data/processed` folder to a public GitHub repository.
3.  **Deploy:** Go to [share.streamlit.io](https://share.streamlit.io/), link your GitHub account, and deploy the repository. In the app's settings, copy and paste the contents of your local `secrets.toml` file into the Secrets manager.