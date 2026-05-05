# Imports:
from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    return OllamaEmbeddings(
        model = 'mxbai-embed-large',
    )

# nomic-embed-text > Light Model
# mxbai-embed-large > Performance Model