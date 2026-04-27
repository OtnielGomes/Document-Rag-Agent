# Imports:
import streamlit as st
import time

# Imports SRC:
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_text
from src.storage.vector_store import index_chunks
from src.graph.workflow import build_rag_workflow
from src.observability.metrics import build_run_log, save_run_log
from src.config import ensure_directories, CHUNK_SIZE, CHUNK_OVERLAP

# Set config directories
ensure_directories()

# Set Page, Title and Caption
st.set_page_config(page_title = 'Document RAG Assistant', layout = 'wide')
st.title('Document RAG Assistant')
st.caption('Upload PDF + questions and answers with RAG workflow')

# Check Inputs
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'indexed_file_name' not in st.session_state:
    st.session_state.indexed_file_name = None

if 'parsed_doc' not in st.session_state:
    st.session_state.parsed_doc = None

if 'chunks' not in st.session_state:
    st.session_state.chunks = []

if 'indexed_count' not in st.session_state:
    st.session_state.indexed_count = 0

# Uploaded Files
uploaded_file = st.file_uploader('Upload a PDF', type = ['pdf'])

if uploaded_file is not None:
    if st.session_state.indexed_file_name != uploaded_file.name:
        parsed_doc = parse_pdf(uploaded_file, source_name = uploaded_file.name)
        chunks = chunk_text(
            parsed_doc.pages,
            source_name = uploaded_file.name,
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
        )
        indexed_count = index_chunks(chunks)

        st.session_state.parsed_doc = parsed_doc
        st.session_state.chunks = chunks
        st.session_state.indexed_count = indexed_count
        st.session_state.indexed_file_name = uploaded_file.name

    parsed_doc = st.session_state.parsed_doc
    chunks = st.session_state.chunks
    indexed_count = st.session_state.indexed_count

    st.success(f'PDF uploaded successfully. Pages read: {parsed_doc.total_pages}')
    st.write(f'Total chunks generated: {len(chunks)}')
    st.write(f'Total chunks indexed in Chroma: {indexed_count}')

    st.subheader('Preview of the extracted text')
    st.text_area(
        'Extracted content',
        parsed_doc.full_text[:3000] if parsed_doc.full_text else 'No text found.',
        height = 300,
    )

    if chunks:
        st.subheader('Preview of the first chunk')
        st.text_area(
            'First chunk',
            chunks[0].text,
            height = 200,
        )

# Question
question = st.text_input('Type your question about the document.')

if st.button('Send a question'):
    start = time.time()

    if not uploaded_file:
        st.warning('Send a PDF before asking a question.')

    elif not question.strip():
        st.warning('Type a question.')

    else:
        try:
            workflow = build_rag_workflow()

            result = workflow.invoke(
                {
                    'question': question,
                    'source': uploaded_file.name,
                }
            )

            answer = result['answer']
            retrieved_docs = result.get('retrieved_docs', [])
            retrieval_latency = result.get('retrieval_latency', 0)
            generation_latency = result.get('generation_latency', 0)

            elapsed = round(retrieval_latency + generation_latency, 3)

            st.write(f'Retrieved docs: {len(retrieved_docs)}')

            if retrieved_docs:
                st.subheader('Retrieved context')

                for i, doc in enumerate(retrieved_docs, start = 1):
                    st.markdown(
                        f"**Result {i}** | source: {doc.metadata.get('source')} | "
                        f"page: {doc.metadata.get('page_number')}"
                    )
                    st.write(doc.page_content[:500])
            
            st.session_state.messages.append(
                {
                    'question': question,
                    'answer': answer,
                    'latency': elapsed,
                }
            )

            log_data = build_run_log(
                question = question,
                source = uploaded_file.name,
                answer = answer,
                retrieved_docs_count = len(retrieved_docs),
                retrieval_latency = retrieval_latency,
                generation_latency = generation_latency,
                total_latency = elapsed,
            )

            save_run_log(log_data)

        except Exception as e:
            st.error(f'Error while processing the question: {e}')


# Output
st.markdown("""
<style>
.answer-box {
    background-color: #f5f7fb;
    border: 1px solid #d9e2f1;
    border-radius: 12px;
    padding: 16px 18px;
    margin-top: 8px;
    margin-bottom: 8px;
}
.answer-label {
    font-weight: 600;
    margin-bottom: 8px;
    color: #1f2937;
}
</style>
""", unsafe_allow_html = True)


for item in reversed(st.session_state.messages):
    st.markdown(f"**Question:** {item['question']}")

    st.markdown('<div class="answer-box">', unsafe_allow_html = True)
    st.markdown('<div class="answer-label">Answer:</div>', unsafe_allow_html = True)
    st.markdown(item["answer"])
    st.markdown('</div>', unsafe_allow_html = True)

    st.caption(f"Latency: {item['latency']}s")
    st.divider()