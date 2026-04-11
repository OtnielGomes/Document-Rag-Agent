# Import 
import streamlit as st
import time
# SRC/ Functions Utils:
import sys
sys.path.append('../src')
from src.ingestion.pdf_parser import parse_pdf

st.set_page_config(page_title = 'Document RAG Assistant', layout = 'wide')

st.title('Document RAG Assistant')
st.caption('Upload PDF + questions and answers with RAG workflow')

if 'messages' not in st.session_state:
  st.session_state.messages = []

uploaded_file = st.file_uploader('Upload a PDF', type = ['pdf'])

if uploaded_file is not None:
  parsed_doc = parse_pdf(uploaded_file, source_name = uploaded_file.name)

st.sucess(f'PDF uploaded successfully. Pages read: {parsed_doc.total_pages}')
st.subheader('Preview of the extracted text')
st.text_area(
  'Extracted content',
  parsed_doc.full_text[:3000] if parsed_doc.full_text else 'No text found.',
  height = 300,
)

question = st.text_input('Type your question about the document.')

if st.button('Send a question'):
    start = time.time()
    
    if not uploaded_file:
      st.warning('Send a PDF before asking a question.')
    
    elif not question.strip():
      st.warning('Type a question.')

    else:
      fake_answer = 'Example response. In the next step we will connect parser, retrieval, and LLM.'
      elapsed = time.time() - start

      st.session_state.messages.append(
          {
            'question': question,
            'answer': fake_answer,
            'latency': round(elapsed, 3)
          }
      )

for item in reversed(st.session_state.messages):
  st.markdown(f"**Question:** {item['question']}")  
  st.markdown(f"**answer:** {item['answer']}") 
  st.caption(f"Latency: {item['latency']}s")
  st.divider()
