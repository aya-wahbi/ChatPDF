# ChatPDF Assistant

This project implements a local ChatPDF assistant that allows you to query your documents using a local Large Language Model (LLM) powered by Ollama. It ingests PDF documents, creates vector embeddings, stores them, and then uses these embeddings to answer questions based on the content of your PDFs.

## Features

*   **Local LLM Integration:** Uses Ollama for entirely local question answering, ensuring data privacy.
*   **Document Ingestion:** Processes PDF files from a specified directory.
*   **Vector Database:** Stores document embeddings for efficient semantic search.
*   **Contextual Answering:** Retrieves relevant document chunks to provide informed answers.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.8+:**
    *   Download from [python.org](https://www.python.org/downloads/)

2.  **Ollama:**
    *   Download and install Ollama from [ollama.ai](https://ollama.ai/).
    *   After installation, you'll need to pull a model. We recommend `llama2` for general use:
        ```bash
        ollama pull llama2
        ```
    

## Project Setup

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

If you haven't already, clone this repository to your local machine:

```bash
git clone https://github.com/aya-wahbi/ChatPDF.git
cd ChatPDF
```

### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
    
# Create the virtual environment
```bash
python -m venv venv

```
# Activate the virtual environment (Windows)

```bash
.\venv\Scripts\activate
```


# Activate the virtual environment (macOS/Linux)
# source venv/bin/activate

You should see (venv) appear at the beginning of your command prompt, indicating the virtual environment is active.

### 3. Install Python Dependencies
With your virtual environment active, install all required Python packages:


```bash
pip install -r requirements.txt
```

### 4. Place Your Documents
Place the PDF documents you want to query into the data/raw directory within this project.


#### How to Run the Project
The project typically involves two main steps: ingesting your documents and then querying them.

1. Ingest Documents
First, you need to process your PDF documents and create their embeddings. This will build or update your vector index.


```bash
python -m scripts.ingest_documents
```


2. Ask Questions
Once the documents are ingested, you can run the answer_generation script to ask questions.

```bash
python -m src.query.answer_generation
```

This script should then prompt you to enter a query or run with a predefined query.

### Project Structure
ChatPDF-GenerativeAI/
├── README.md                # Project overview, goals, and guide
├── requirements.txt         # All dependencies required for development
├── setup.py                 # (Optional) For packaging and installation purposes
├── .gitignore               # Files and directories to ignore in Git
├── docs/                    # Documentation resources
│   ├── project_overview.md  # Detailed overview of the project
│   ├── design_document.md   # Architecture and design decisions
│   └── usage_guide.md       # How to use the tool once built
├── data/                    # Data storage folder
│   ├── raw/                 # Original documents (PDFs, DOCX, etc.)
│   └── processed/           # Output from the ingestion/preprocessing phase
├── notebooks/               # Jupyter notebooks for exploration and prototyping
│   └── exploration.ipynb    # Experiment with models, sample extraction, etc.
├── src/                     # Source code for the project
│   ├── __init__.py          
│   ├── ingestion/           # Code for document loading and text extraction
│   │   ├── __init__.py
│   │   ├── pdf_extraction.py    # Example: functions to extract text from PDFs
│   │   └── document_loader.py   # Handle multiple file formats
│   ├── preprocessing/       # Code for text cleaning and chunking
│   │   ├── __init__.py
│   │   └── text_chunking.py     # Chunk text into manageable segments
│   ├── embedding/           # Code for vector embeddings and database indexing
│   │   ├── __init__.py
│   │   └── vector_index.py      # Create and query vector indices (using FAISS, ChromaDB, etc.)
│   ├── query/               # Query handling and LLM prompt generation
│   │   ├── __init__.py
│   │   └── answer_generation.py # Generate answers from embeddings and query context
│   ├── gui/                 # Graphical User Interface code (e.g., PyQt5 or PySide6)
│   │   ├── __init__.py
│   │   └── chatpdf_ui.py        # Main window and UI event handling
│   └── utils/               # Utility functions and helper modules
│       ├── __init__.py
│       └── helpers.py           # Reusable helper functions across modules
├── tests/                   # Unit and integration tests for each module
│   ├── ingestion_tests.py
│   ├── preprocessing_tests.py
│   ├── embedding_tests.py
│   ├── query_tests.py
│   └── gui_tests.py             # Optionally test basic UI functionality (if applicable)
└── scripts/                 # Standalone scripts for running processes
    ├── run_app.py           # Main script to launch the UI/application
    └── ingest_documents.py  # Script to ingest and process documents from the data folder
─────────────────────────────
Folder Explanations
─────────────────────────────

• README.md:
   – Provides an overview of the project, its purpose, and how to get started.
   – Can include instructions for installation, configuration, and running the application.

• requirements.txt & setup.py:
   – List all the Python packages (e.g., unstructured, sentence-transformers, faiss-cpu, PyQt5) that your project depends on.
   – Using setup.py enables you to build the project as a package if needed.

• docs/:
   – Contains all project documentation, like design decisions, usage guides, and any planning documents.
   – This is particularly helpful for internal team communication and onboarding new collaborators.

• data/:
   – raw/ holds the original research papers and documents as downloaded or collected.
   – processed/ is where you put text outputs after running ingestion and preprocessing scripts.

• notebooks/:
   – Use this space for exploratory analysis or model prototyping.
   – Jupyter notebooks can help test ideas and visualize results before integrating code into the main pipeline.

• src/:
   – The central source code directory. The subfolders break the project into logical components:
   – ingestion: Extracting text from various file formats
   – preprocessing: Chunking and cleaning extracted text
   – embedding: Converting text chunks to vector embeddings and indexing
   – query: Handling user query processing and answer generation
   – gui: Developing the desktop application interface
   – utils: Shared helper functions and utilities for code reuse

• tests/:
   – Stores unit tests and integration tests to validate each component of your system.
   – Writing tests early on ensures that refactoring or adding new features won’t break existing functionality.

• scripts/:
   – Contains utility scripts for common tasks such as launching the application or performing batch ingestion.
   – This helps to separate one-off scripts from the structured application code.