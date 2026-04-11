#document-rag-agent

RAG agent for analyzing PDF documents with Streamlit, LangGraph, and vector search.

##Objective
To allow the user to upload a PDF, ask questions about the content, and receive answers with the context retrieved from the document.

##MVP
- Loading PDF
- Text Extraction
- Chunks
- Embeds
- Vector Search
- Question and Answer with Source
- Simple Workflow with LangGraph
- Basic Latency Metrics

## Stack
- Python
- Streamlit
- LangGraph
- Data Blocks
- Vector Search / Vector Layer
- LLM Provider

## Structure
```bash
agent-rag-document/
├──app.py
├── requirements.txt
├──app.yaml
├── README.md
├── .env.example
└── src/
└── graph/
└── state.py
```

## Roadmap
1. Create interface of 1. Upload

2. Create PDF parser
3. Segmentation implemented
4. Generate embeds
5. Create retriever
6. To respond with LLM
7. Orchestrate with LangGraph
8. Measure latency
