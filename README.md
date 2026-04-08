get_embedding_function.py
Returns an embeddings model using Ollama's "nomic-embed-text" for converting text to vectors

populate_database.py: `python populate_database.py` to run
Loads all PDF from the \data folder, splits them into 800-token chunks with 80-token overlap, and store them in a Chroma vector database. Skip docs already in the database. Supports --reset flag to clear & rebuild

query_data.py: `python query_data.py "prompt"` to run
Takes a question, searches the Chroma database for 5 most relevant chunks, feeds them as context to the Mistral LLM & return the LLM's answer + sources

test_rag.py: `pytest -s` to run
test RAG on questions. Use Mistral as evaluator to check if system's answers contain expected responses.
