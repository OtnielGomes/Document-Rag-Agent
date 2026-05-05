# Imports:
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chunk Class
@dataclass
class Chunk:
    chunk_id: str
    source: str
    page_number: int
    extraction_method: str
    start_char: int
    end_char: int
    text: str


# Chunks
def chunk_text(
    pages,
    source_name: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 240,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = []

    for page in pages:
        page_text = page.text.strip()

        if not page_text:
            continue

        docs = splitter.create_documents(
            texts = [page_text],
            metadatas = [{
                'source': source_name,
                'page_number': page.page_number,
                'extraction_method': page.extraction_method,
            }],
        )

        for chunk_index, doc in enumerate(docs, start=1):
            start_char = doc.metadata.get('start_index', 0)
            end_char = start_char + len(doc.page_content)

            chunks.append(
                Chunk(
                    chunk_id = f"{source_name}-p{page.page_number}-c{chunk_index}",
                    source=source_name,
                    page_number = page.page_number,
                    extraction_method = page.extraction_method,
                    start_char = start_char,
                    end_char = end_char,
                    text = doc.page_content,
                )
            )

    return chunks