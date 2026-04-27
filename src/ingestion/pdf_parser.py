# Imports:
from dataclasses import dataclass
import re
import unicodedata
import pymupdf

# Data Classe
@dataclass
class ParsedDocument:
    source_name: str
    full_text: str
    pages: list[str]
    total_pages: int

# List Of 
PRIVATE_USE_AND_REPLACEMENT_PATTERN = r"[\uE000-\uF8FF\uFFF0-\uFFFF ]"
ZERO_WIDTH_PATTERN = r"[\u200B-\u200D\uFEFF]"
ASCII_CONTROL_PATTERN = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
EXCESS_SYMBOLS_PATTERN = r"[■□▪◆◦●◼◻]"
BAD_LINE_CHARS_PATTERN = r"[^\w\sÀ-ÿ.,;:!?()/%+\-–—&·]"
COMMON_LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

EXPLICIT_BAD_TOKENS = [
    "",
    "",
    "",
    "Ð",
    "",
    "¢",
    "®",
    "",
]

# Normalize Unicode
def normalize_unicode_artifacts(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)

    for bad, good in COMMON_LIGATURES.items():
        text = text.replace(bad, good)
    
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("•", "- ")
    text = text.replace("·", " · ")

    text = re.sub(PRIVATE_USE_AND_REPLACEMENT_PATTERN, " ", text)
    text = re.sub(ZERO_WIDTH_PATTERN, "", text)
    text = re.sub(ASCII_CONTROL_PATTERN, " ", text)
    text = re.sub(EXCESS_SYMBOLS_PATTERN, " ", text)

    return text

# Remove Explicit Bad Tokens
def remove_explicit_bad_tokens(
    text: str
) -> str:

    for token in EXPLICIT_BAD_TOKENS:
        text = text.replace(token, " ")
    
    return text

# Normalize White Space
def normalize_whitespace(
    text: str
) -> str:
    
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def fix_spaced_letters_in_line(
    line: str
) -> str:
    
    if not line:
        return ""
    
    if re.fullmatch(r"(?:[A-Za-zÀ-ÿ0-9&/-]\s+){4,}[A-Za-zÀ-ÿ0-9&/-]", line):
        return re.sub(r"\s+", "", line)
    
    if re.search(r"(?:\b[A-Za-zÀ-ÿ]\b\s+){4,}\b[A-Za-zÀ-ÿ]\b", line):
        return re.sub(r"(?<=\b[A-Za-zÀ-ÿ])\s+(?=[A-Za-zÀ-ÿ]\b)", "", line)
    
    return line



# Probably Garbage
def is_probably_garbage(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False
    
    if len(stripped) <= 3:
        return False
    
    alnum_count = sum(ch.isalnum() for ch in stripped)
    total_count = len(stripped)

    bad_chars = len(re.findall(BAD_LINE_CHARS_PATTERN, stripped))
    alnum_ratio = alnum_count / max(total_count, 1)
    bad_ratio = bad_chars / max(total_count, 1)

    if alnum_ratio < 0.35 and bad_ratio > 0.25:
        return True
    
    return False

# Drop Noise
def drop_noise_lines(lines: list[str]) -> list[str]:
    
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        if len(stripped) <= 4 and sum(ch.isalnum() for ch in stripped) == 0:
            continue

        weird_ratio = sum(
            not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()/%+-&·")
            for ch in stripped
        ) / max(len(stripped), 1)

        if weird_ratio > 0.35:
            continue

        cleaned.append(stripped)
    
    return cleaned

# Clean Text
def clean_text(
    text: str
) -> str:
    
    text = normalize_unicode_artifacts(text)
    text = remove_explicit_bad_tokens(text)

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        
        line = line.strip()
        
        if not line:
            cleaned_lines.append("")
            continue

        line = fix_spaced_letters_in_line(line)
        line = re.sub(r"[ ]{2,}", " ", line)

        if is_probably_garbage(line):
            continue

        cleaned_lines.append(line)
    
    cleaned_lines = drop_noise_lines(cleaned_lines)

    text = "\n".join(cleaned_lines)
    text = normalize_whitespace(text)
    return text

# Extract Page Text Native
def extract_page_text_native(
    page
) -> str:
    
    text = page.get_text("text", sort = True)
    return clean_text(text)

# Extract Page Text Blocks
def extract_page_text_blocks(
    page
) -> str:
    
    blocks = page.get_text("blocks", sort = True)
    ordered_blocks = []

    for block in blocks:
        x0, y0, x1, y1, block_text, block_no, block_type = block

        if block_type != 0:
            continue

        block_text = clean_text(block_text)

        if block_text:
            ordered_blocks.append((y0, x0, block_text))
    
    ordered_blocks.sort(key = lambda item: (item[0], item[1]))
    page_text = "\n\n".join(block_text for _, _, block_text in ordered_blocks)

    return clean_text(page_text)

# Extract Page Text ORC
def extract_page_text_ocr(
    page
) -> str:
    textpage = page.get_textpage_ocr(dpi = 300, full = True)
    text = page.get_text("text", textpage = textpage, sort = True)
    return clean_text(text)

# Extract Page Text
def extract_page_text(
    page
) -> str:
    
    text = extract_page_text_native(page)

    if len(text.strip()) >= 40:
        return text
    
    text = extract_page_text_blocks(page)

    if len(text.strip()) >= 40:
        return text
    
    try:
        text = extract_page_text_ocr(page)
        return text
    
    except Exception:
        return text

def parse_pdf(
    uploaded_file, 
    source_name: str
) -> ParsedDocument:

    pdf_bytes = uploaded_file.getvalue()
    doc = pymupdf.open(stream = pdf_bytes, filetype = 'pdf')

    pages = []

    try:
        for page in doc:
            page_text = extract_page_text(page)
            pages.append(page_text)
    finally:
        doc.close()

    full_text = '\n\n'.join(page for page in pages if page.strip())
    full_text = clean_text(full_text)

    return ParsedDocument(
        source_name = source_name,
        full_text = full_text,
        pages = pages,
        total_pages = len(pages),
    )
