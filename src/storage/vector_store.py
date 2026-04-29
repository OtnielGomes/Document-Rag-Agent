# Imports:
from __future__ import annotations
from functools import lru_cache
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Imports SRC:
from src.ingestion.embeddings import get_embeddings
from src.config import CHROMA_DIR, TOP_K_RESULTS

# Directories
PERSIST_DIRECTORY = str(CHROMA_DIR)
COLLECTION_NAME = 'document_rag_collection'
DEFAULT_FETCH_K = 12
DEFAULT_LAMBDA_MULT = 0.4

# Vector Store
@lru_cache(maxsize = 1)
def get_vector_store() -> Chroma:
    embeddings = get_embeddings()

    return Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory = PERSIST_DIRECTORY,
    )

# Documents Chunks
def build_documents_from_chunks(
    chunks
) -> List[Document]:
    
    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content = chunk.text,
                metadata = {
                    'chunk_id': chunk.chunk_id,
                    'source': chunk.source,
                    'page_number': chunk.page_number,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'section': getattr(chunk, 'section', None)

                },
            )
        )
    
    return documents

# Index Chunks 
def index_chunks(
    chunks
) -> int:
    
    vector_store = get_vector_store()
    documents = build_documents_from_chunks(chunks)

    if not documents:
        return 0
    
    ids = [doc.metadata['chunk_id'] for doc in documents]
    vector_store.add_documents(documents = documents, ids = ids)
    return len(documents)

# Retriever
def get_retriever(
  source: str | None = None,
  section: str | None = None,
  k: int = TOP_K_RESULTS,  
  fetch_k: int = DEFAULT_FETCH_K,
  lambda_mult: float = DEFAULT_LAMBDA_MULT,  
):

    vector_store = get_vector_store()

    fetch_k = max(fetch_k, k)

    search_kwargs = {
        'k': k,
        'fetch_k': fetch_k,
        'lambda_mult': lambda_mult,
    }

    filter_dict = {}
    if source is not None:
        filter_dict['source'] = source

    if section is not None:
        filter_dict['section'] = section

    if filter_dict:
        search_kwargs['filter'] = filter_dict

    return vector_store.as_retriever(
        search_type = 'mmr',
        search_kwargs = search_kwargs,
    )

# Search Similar Chunks
def search_similar_chunks(
    query: str,
    source: str,
    k: int = TOP_K_RESULTS,
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_LAMBDA_MULT,
    section: str | None = None
):
    retriever = get_retriever(
        source = source,
        section = section,
        k = k,
        fetch_k = fetch_k,
        lambda_mult = lambda_mult,
    )

    return retriever.invoke(query)
