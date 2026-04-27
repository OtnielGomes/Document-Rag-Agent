# Imports: 
from typing import Any, List, TypedDict

# GraphState
class GraphState(
    TypedDict,
    total = False
):
    question: str
    source: str
    retrieved_docs: List[Any]
    answer: str
    retrieval_latency: float
    generation_latency: float
    total_latency: float
