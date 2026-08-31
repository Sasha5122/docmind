"""Cut a parsed document into chunks of ~300-800 tokens that remember their page.

Why chunk at all: one embedding = one "meaning". A whole PDF is too much
meaning for one vector; one sentence has too little context. Chunks of a few
hundred tokens with a small overlap are the standard compromise.

Algorithm (paragraph-aware, then overlap):
  1. For each page, split the text into paragraphs; skip pages with no text.
  2. Walk the paragraphs in order, packing them into the current chunk until the
     next one would push it over `max_tokens`; then emit and start a new chunk.
  3. A paragraph longer than the limit is split by words into smaller pieces.
  4. Each new chunk starts with the last ~`overlap_tokens` tokens of the previous
     one, so a sentence cut at a boundary is whole in at least one chunk.
  5. `page` = page of the chunk's first ORIGINAL paragraph (not the overlap).
     Chunks may run across a page boundary; they cite the page they start on.
"""

import re
from dataclasses import dataclass

import tiktoken

from docmind.ingest.parser import ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int  # 0-based position within the document
    page: int  # 1-based page where the chunk starts
    text: str
    token_count: int  # must equal count_tokens(text)


def count_tokens(text: str) -> int:
    """Number of tokens (word-pieces) in `text`. ~0.75 words per token in English."""
    return len(_ENCODING.encode(text))


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; drop empty paragraphs; keep inner line breaks."""
    return [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]


def chunk_document(
    doc: ParsedDocument, max_tokens: int = 500, overlap_tokens: int = 50
) -> list[TextChunk]:
    """Return the document's chunks in reading order.

    Guarantees the tests check:
      - every chunk has 0 < token_count <= max_tokens and token_count == count_tokens(text)
      - chunk_index runs 0, 1, 2 ... with no gaps
      - page numbers are valid and never decrease
      - every source paragraph appears in at least one chunk
      - with overlap_tokens > 0, consecutive chunks share text; with 0 they do not
      - a document with no text returns []
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than max_tokens")

    # Pieces are (page, text) pairs; each piece already fits into one chunk.
    piece_limit = max_tokens - overlap_tokens if overlap_tokens else max_tokens
    pieces = [
        (page.number, piece)
        for page in doc.pages
        for paragraph in split_paragraphs(page.text)
        for piece in _split_long_paragraph(paragraph, piece_limit)
    ]

    chunks: list[TextChunk] = []
    current: list[str] = []  # texts in the chunk being built (may start with an overlap tail)
    has_content = False  # True once `current` holds at least one original piece
    start_page = 0

    def emit() -> None:
        nonlocal current, has_content
        text = "\n\n".join(current)
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                page=start_page,
                text=text,
                token_count=count_tokens(text),
            )
        )
        tail = _tail(text, overlap_tokens)
        current = [tail] if tail else []
        has_content = False

    for page, piece in pieces:
        candidate = "\n\n".join([*current, piece])
        if count_tokens(candidate) > max_tokens:
            if has_content:
                emit()
                candidate = "\n\n".join([*current, piece])
            if count_tokens(candidate) > max_tokens:
                # Only the overlap tail is in the way; drop it rather than exceed the limit.
                current = []
        if not has_content:
            start_page = page
            has_content = True
        current.append(piece)

    if has_content:
        emit()
    return chunks


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Return [paragraph] if it fits, else word-packed pieces that each fit."""
    if count_tokens(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    words: list[str] = []
    for word in paragraph.split():
        if words and count_tokens(" ".join([*words, word])) > limit:
            pieces.append(" ".join(words))
            words = []
        words.append(word)
    if words:
        pieces.append(" ".join(words))
    return pieces


def _tail(text: str, overlap_tokens: int) -> str:
    """The last ~overlap_tokens tokens of `text`, cut on word boundaries."""
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        tail.insert(0, word)
        if count_tokens(" ".join(tail)) >= overlap_tokens:
            break
    return " ".join(tail)
