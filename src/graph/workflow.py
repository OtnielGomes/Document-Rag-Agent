# Imports:
from langgraph.graph import StateGraph, START, END

# Imports SRC:
from src.graph.state import GraphState
from src.graph.nodes import retrieve_documents, generate_response

# Work Flow
def build_rag_workflow():
    builder = StateGraph(GraphState)

    builder.add_node('retrieve_documents', retrieve_documents)
    builder.add_node('generate_response', generate_response)

    builder.add_edge(START, 'retrieve_documents')
    builder.add_edge('retrieve_documents', 'generate_response')
    builder.add_edge('generate_response', END)

    return builder.compile()


