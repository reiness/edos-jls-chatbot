Of course. This is the final and most important piece of documentation. A great `README` makes a project usable for everyone.

Here is the fully updated `README.md`, incorporating all the features and fixes we've implemented, including the new security measures, and structured for both regular users and developers.

-----

# JLS SOP Chatbot (v1.0.0-beta)

A production-ready, Retrieval-Augmented Generation (RAG) chatbot designed to accurately answer questions from a large knowledge base of company Standard Operating Procedures (SOPs).

This project features an end-to-end data pipeline that processes documents from Google Docs and a secure, interactive user interface built with Streamlit.

## Table of Contents

- [JLS SOP Chatbot (v1.0.0-beta)](#jls-sop-chatbot-v100-beta)
  - [Table of Contents](#table-of-contents)
- [JLS SOP Chatbot: User Guide](#jls-sop-chatbot-user-guide)
  - [How to Use the Chatbot](#how-to-use-the-chatbot)
- [JLS SOP Chatbot: Developer \& Admin Guide](#jls-sop-chatbot-developer--admin-guide)
  - [✨ Key Features](#-key-features)
  - [📂 Project Structure](#-project-structure)
  - [🛠️ Step 1: Installation \& Setup](#️-step-1-installation--setup)
      - [Prerequisites](#prerequisites)
      - [Installation \& Configuration](#installation--configuration)
  - [⚙️ Step 2: The Data Pipeline Workflow](#️-step-2-the-data-pipeline-workflow)
      - [2.1: Download Documents](#21-download-documents)
      - [2.2: Ingest \& Embed Documents](#22-ingest--embed-documents)
      - [2.3: Build the Search Index \& Manifest](#23-build-the-search-index--manifest)
  - [🚀 Step 3: Running the Chatbot Locally](#-step-3-running-the-chatbot-locally)
  - [☁️ Step 4: Deployment](#️-step-4-deployment)
  - [🚨 Troubleshooting](#-troubleshooting)
-----

# JLS SOP Chatbot: User Guide

This guide is for internal team members who want to ask questions and get answers from the chatbot.

## How to Use the Chatbot

Using the chatbot is simple and secure.

1.  **Access the Chatbot:** When you first visit the app's URL, you will see a login screen. Enter the password provided to you by the administrator and click "Submit" to access the main chat interface.

2.  **Navigate:** Use the sidebar to switch between the main **Chatbot**, the **Introduction** page, or to browse the **Knowledge Base**.

3.  **Ask a Question:** In the Chatbot section, type any question you have about our company's SOPs into the chat box at the bottom of the screen and press Enter.

4.  **Explore the Sources:** After the chatbot provides an answer, it will list the source documents it used. You can click on the expandable sections to see a relevant snippet and a direct link to the full Google Doc for more details.

5.  **Adjust the Relevance Threshold (Optional):** In the chatbot's sidebar, you can adjust the "Relevance Threshold."

      * A **higher value** makes the search stricter, which is good for very specific questions.
      * A **lower value** makes the search broader, which can be helpful for more general questions.

-----

# JLS SOP Chatbot: Developer & Admin Guide

This guide is for engineers or administrators who need to set up the project, maintain the knowledge base, or deploy the application.

## ✨ Key Features

  - **Automated Data Pipeline:** Downloads, processes, and indexes documents from a Google Sheet with a series of simple commands.
  - **Secure by Design:** Features a password-protected UI and encrypts the knowledge base manifest to keep the list of internal documents private in the public repository.
  - **Advanced Semantic Chunking:** Employs `RecursiveCharacterTextSplitter` to create meaningful text chunks for high-quality retrieval.
  - **Context-Enriched Embeddings:** Injects metadata (title, section, etc.) into the text before embedding, leading to highly accurate search results.
  - **Smart Retrieval:** Uses a Relevance Score Threshold instead of a fixed number of sources, ensuring only the most relevant information is used to generate answers.

## 📂 Project Structure

The project is organized into a standard Python package structure for clarity and scalability.

```
JLS-Chatbot/
│
├── .streamlit/
│   └── secrets.toml         # For deployment secrets
│
├── data/
│   ├── source_documents/    # Raw downloaded PDFs and .metadata.json (local only)
│   └── processed/           # The output of the data pipeline (safe to deploy)
│       ├── chunks/
│       │   └── chunks.jsonl
│       ├── index/
│       │   └── faiss_index/
│       └── knowledge_base_manifest.enc  # Encrypted manifest file
│
├── src/
│   └── jls_chatbot/         # The main Python package
│       ├── pipeline/
│       ├── core/
│       └── app.py
│
├── .env                     # Local secrets and config (DO NOT COMMIT)
├── .gitignore
├── README.md
└── requirements.txt
```

-----

## 🛠️ Step 1: Installation & Setup

#### Prerequisites

  - Python 3.10+
  - Git
  - A Google Cloud Project with the **Vertex AI API** enabled.

#### Installation & Configuration

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-org/jls-chatbot.git
    cd jls-chatbot
    ```

2.  **Create and Activate a Virtual Environment**

    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    # source .venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Google Credentials**

      * For the download script to work, you need `credentials.json` from Google Cloud. Place this file in the **root directory** of the project. It should be listed in your `.gitignore`.

5.  **Configure Environment (`.env` file)**

      * Create a `.env` file in the project root. This file is for **local development only**.
      * **Generate an Encryption Key:** Run this command once and copy the output.
        ```bash
        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
        ```
      * **Populate your `.env` file:**
        ```dotenv
        # .env

        # --- REQUIRED SECRETS ---
        GEMINI_API_KEY="your-google-gemini-api-key"
        MANIFEST_KEY="your-generated-encryption-key-from-above"

        # --- PIPELINE SETTINGS (Optional) ---
        # Adjust batch size for the embedding process if you hit rate limits.
        EMBED_BATCH_SIZE=20

        # --- DATA PATHS (Defaults are recommended) ---
        PDF_FOLDER=./data/source_documents
        CHUNKS_DIR=./data/processed/chunks
        INDEX_DIR=./data/processed/index
        ```

-----

## ⚙️ Step 2: The Data Pipeline Workflow

This three-step process converts your source documents into a secure, searchable knowledge base. Run these commands from the **project's root directory**.

#### 2.1: Download Documents

Connects to Google Sheets, downloads all linked Google Docs as PDFs, and creates the raw `.metadata.json` file.

```bash
python src/jls_chatbot/pipeline/download.py
```

#### 2.2: Ingest & Embed Documents

Processes the PDFs, chunks them, and creates embeddings using the Gemini API. This is the most time-consuming step.

```bash
python src/jls_chatbot/pipeline/ingest.py
```

#### 2.3: Build the Search Index & Manifest

Creates the final FAISS vector index and the secure, encrypted `knowledge_base_manifest.enc` file.

```bash
python src/jls_chatbot/pipeline/build_index.py
```

Re-run this entire pipeline whenever the source SOPs are updated.

-----

## 🚀 Step 3: Running the Chatbot Locally

Launch the Streamlit web application with the following command:

```bash
streamlit run src/jls_chatbot/app.py
```

-----

## ☁️ Step 4: Deployment

Deploy the app for free on Streamlit Community Cloud.

1.  **Prepare Secrets:** Create a `.streamlit/secrets.toml` file locally. Add your **`GEMINI_API_KEY`**, the app **`PASSWORD`**, and your **`MANIFEST_KEY`** to it. Add this file to `.gitignore`.
2.  **Push to GitHub:** Push your entire project—**including the populated `data/processed` folder** to a public GitHub repository. Do **not** push `data/source_documents`.
3.  **Deploy:** Go to [share.streamlit.io](https://share.streamlit.io/), link your GitHub account, and deploy the repository. In the app's settings, copy and paste the contents of your local `secrets.toml` file into the Secrets manager.

-----

## 🚨 Troubleshooting

  - **`ModuleNotFoundError: No module named 'jls_chatbot'`**: This means you are not running the command from the project's root directory. Make sure your terminal is in the main `JLS-Chatbot/` folder.
  - **`429 Quota Exceeded (limit: 0)` Error**: The **Vertex AI API** is not enabled in your Google Cloud Project. Please follow the steps in the Google Cloud Console to enable it.
  - **`429 Quota Exceeded` (Normal Rate Limit)**: The ingestion script is running too fast. Decrease the `EMBED_BATCH_SIZE` in your `.env` file (e.g., from 20 to 10) and try again.