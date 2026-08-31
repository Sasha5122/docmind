"""Cut a parsed document into chunks of ~300-800 tokens that remember their page.

>>> HAND-WRITTEN BY THE USER (learning mode). Claude provides the contract,
>>> helpers and tests; the body of `chunk_document` is yours. <<<

Why chunk at all: one embedding = one "meaning". A whole PDF is too much
meaning for one vector; one sentence has too little context. Chunks of a few
hundred tokens with a small overlap are the standard compromise.

Suggested algorithm (paragraph-aware, then overlap):
  1. For each page, split the text into paragraphs (see `split_paragraphs`).
     Skip pages with no text.
  2. Walk the paragraphs in order and keep a "current chunk" (list of paragraphs).
     If adding the next paragraph would push the chunk over `max_tokens`,
     emit the current chunk first.
  3. A single paragraph longer than `max_tokens` must itself be split
     (e.g. by sentences or words) so no chunk ever exceeds the limit.
  4. Overlap: when you start a new chunk, seed it with the last
     ~`overlap_tokens` tokens of text from the previous chunk (take the last
     few words until `count_tokens` says you have enough). Overlap 0 = none.
  5. `page` of a chunk = the page where its FIRST original paragraph came from.
     `chunk_index` counts 0, 1, 2 ... across the whole document.
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
    raise NotImplementedError("your turn: see the module docstring for the plan")
