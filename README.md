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
This step has to be done only when new data/PDFs are added, in case you want to run the app on the already indgested data you can skip this step.
First, you need to process your PDF documents and create their embeddings. This will build or update your vector index.


```bash
python -m scripts.ingest_documents
```


2. Ask Questions
Once the documents are ingested, you can run the answer_generation script to ask questions.

```bash
python -m src.query.answer_generation
```
This script should then prompt you to enter a query or run with a predefined query

3. Run the UI script (run the app).
```bash
python -m src.scripts.run_app
```
