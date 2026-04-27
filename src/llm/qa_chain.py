# Imports
import os 
from ollama import Client
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Local directories
USE_OLLAMA_CLOUD = os.getenv('USE_OLLAMA_CLOUD', 'false').lower() == 'true'
LOCAL_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')
CLOUD_MODEL = os.getenv('OLLAMA_CLOUD_MODEL', 'gpt-oss:20b')

# Get LLM
def get_llm():
    return ChatOllama(
        model = LOCAL_MODEL,
        temperature = 0,
    )

# Get Cloud Client
def get_cloud_client():
    api_key = os.getenv('OLLAMA_API_KEY')

    if not api_key:
        raise ValueError('OLLAMA_API_KEY not found in environment variables.')

    return Client(
        host = 'https://ollama.com',
        headers = {'Authorization': f'Bearer {api_key}'},
    )

# Build Messages
def build_messages(
    question: str,
    context: str
):

    system_prompt = """
    You are a document analysis assistant.

    Answer using only the retrieved context.

    Rules:
    - Use only the provided context.
    - Do not use prior knowledge.
    - Do not speculate or invent facts.
    - You may combine explicit facts from multiple context chunks.
    - You may perform simple deterministic calculations directly derived from the context.
    - If the answer is not stated explicitly and cannot be directly derived from the context, reply exactly:
    "This is not stated in the document."

    Completeness rules:
    - If the question asks for a list (for example: courses, certifications, skills, experiences, tools), include all relevant items found in the context.
    - Do not provide only examples when the context contains a longer list.
    - If the retrieved context appears partial, answer with what is present and say the list may be incomplete.

    Style rules:
    - Be concise and factual.
    - Use short bullet points when helpful.
    - If the answer is derived by calculation, state that briefly.
    """.strip()

    user_prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """.strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]    

# Build Lang Chain Prompt
def build_langchain_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", """You are a document analysis assistant.

            Answer using only the retrieved context.

            Rules:
            - Use only the provided context.
            - Do not use prior knowledge.
            - Do not speculate or invent facts.
            - You may combine explicit facts from multiple context chunks.
            - You may perform simple deterministic calculations directly derived from the context.
            - If the answer is not stated explicitly and cannot be directly derived from the context, reply exactly:
            "This is not stated in the document."

            Completeness rules:
            - If the question asks for a list (for example: courses, certifications, skills, experiences, tools), include all relevant items found in the context.
            - Do not provide only examples when the context contains a longer list.
            - If the retrieved context appears partial, answer with what is present and say the list may be incomplete.

            Style rules:
            - Be concise and factual.
            - Use short bullet points when helpful.
            - If the answer is derived by calculation, state that briefly.
            """),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

# Generate Answer
def generate_answer(
    question: str,
    retrieved_docs
):
    context = '\n\n'.join(doc.page_content for doc in retrieved_docs)

    # Ollama Cloud
    if USE_OLLAMA_CLOUD:
        client = get_cloud_client()
        response = client.chat(
            model = CLOUD_MODEL,
            messages = build_messages(question, context),
        )
        return response['message']['content']

    
    llm = get_llm()
    prompt = build_langchain_prompt()
    chain = prompt | llm
    response = chain.invoke(
        {
            'context': context,
            'question': question,
        }
    )
    return response.content
