# Imports:
import time

# Import SRC:
from src.graph.state import GraphState
from src.storage.vector_store import search_similar_chunks
from src.llm.qa_chain import generate_answer
from src.config import TOP_K_RESULTS

# Retrieved docs
def retrieve_documents(
    state: GraphState
) -> dict:
    
    start = time.time()

    question = state['question']
    source = state['source']

    results = search_similar_chunks(
        query = question,
        source = source,
        k = TOP_K_RESULTS,
    )

    elapsed = time.time() - start

    return {
        'retrieved_docs': results,
        'retrieval_latency': round(elapsed, 3)
    }

# Generate Response
def generate_response(
    state: GraphState
) -> dict:

    start = time.time()

    question = state['question']
    retrieved_docs = state.get('retrieved_docs', [])

    if retrieved_docs:
        answer = generate_answer(question, retrieved_docs)
    else:
        answer = "I couldn't find relevant context in the indexed document."

    elapsed = time.time() - start

    return {
        'answer': answer,
        'generation_latency': round(elapsed, 3),
    }
