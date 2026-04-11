# Imports
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO, List

from pypdf import PdfReader

@dataclass
class PageDocument:
    page_number: int
    text: str
    source: str

@dataclass
class ParsedDocument:
    source: str
    full_text: str
    pages: List[PageDocument]
    total_pages: int

def _normalize_text(text: str) -> str:
    if not text:
        return ''
    text = text.replace("\x00", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return '\n'.join(lines).strip()

def parse_pdf(
    file_obj: BinaryIO,
    source_name: str = 'uploaded.pdf'
) -> ParsedDocument:
    raw_bytes = file_obj.read()
    pdf_stream = BytesIO(raw_bytes)

    reader = PdfReader(pdf_stream)
    pages: List[PageDocument] = []

    for idx, page in enumerate(reader.pages, start = 1):
        extracted = page.extract_text()  or ''
        cleaned = _normalize_text(extracted)

        pages.append(
            PageDocument(
                page_number = idx,
                text = cleaned,
                source = source_name,
            )
        )
    full_text = '\n\n'.join(page.text for page in pages if page.text).strip()

    return ParsedDocument(
        source = source_name,
        full_text = full_text,
        pages = pages,
        total_pages = len(reader.pages)
    )
