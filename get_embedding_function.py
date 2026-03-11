from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    #embeddings = BedrockEmbeddings(
    #    region_name="us-east-1",
    #    model_id="amazon.titan-embed-text-v1"
    #)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
