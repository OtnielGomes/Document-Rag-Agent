# Imports 
from typing import TypedDict, List, Optional

class GraphState(
    TypedDict,
    total = False
):
    question: str
    document_text: str
    chunks: List[str]
    retrieved_chunks: List[str]
    answer: str
    sources: List[str]
    latency_ms: float
    error: Optional[str]
