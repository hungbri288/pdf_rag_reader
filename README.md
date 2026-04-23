# PDF RAG Reader

A lightweight retrieval-augmented generation (RAG) pipeline for PDF content. This repository extracts text from PDFs, builds a vector database with Chroma, and answers natural language questions using the relevant document chunks as context.

## Contents

- `get_embedding_function.py` — returns an embeddings model using Ollama's `nomic-embed-text`.
- `populate_database.py` — loads PDF files from the `data/` directory, chunks the text, and saves the vectors into a Chroma database.
- `query_data.py` — queries the Chroma database and uses a language model to answer prompts with relevant source context.
- `test_rag.py` — runs tests to validate the RAG workflow and answer quality.

## Requirements

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Setup

1. Place your PDF files in the `data/` directory.
2. Ensure Ollama and any required model endpoints are available for embeddings and model inference.
3. Run the database population step before querying.

## Usage

### Build the vector database

```bash
python populate_database.py
```

Optional reset mode:

```bash
python populate_database.py --reset
```

This will:
- scan the `data/` folder for PDF files
- extract and split the text into 800-token chunks with 80-token overlap
- store vector embeddings in the local Chroma database
- skip documents that are already indexed unless `--reset` is used

### Query the database

```bash
python query_data.py "your question here"
```

This script:
- searches Chroma for the 5 most relevant chunks
- passes them as context to the language model
- returns the generated answer and source information

## Testing

Run the RAG test suite with:

```bash
pytest -s
```

## Notes

- If you use Ollama locally, make sure it is running and accessible.
- Adjust your PDF dataset in `data/` and rerun `populate_database.py` when new files are added.
- The vector database is stored in `chroma/chroma.sqlite3`.

## Directory Structure

```
get_embedding_function.py
populate_database.py
query_data.py
README.md
requirements.txt
test_rag.py
chroma/
  chroma.sqlite3
  <vector data folder>/
data/
  <your PDF files>
```
