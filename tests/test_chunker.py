"""Tests for the hand-written chunker. Run: uv run pytest tests/test_chunker.py -v"""

import pytest

from docmind.ingest.chunker import TextChunk, chunk_document, count_tokens, split_paragraphs
from docmind.ingest.parser import Page, ParsedDocument

SENTENCE = "Der Versicherer erbringt die Leistungen gemäss den nachstehenden Bestimmungen. "


def paragraph(n_sentences: int, tag: str) -> str:
    return f"[{tag}] " + SENTENCE * n_sentences


def make_doc(pages: list[str]) -> ParsedDocument:
    return ParsedDocument(
        filename="test.pdf",
        sha256="0" * 64,
        lang="de",
        pages=[Page(number=i + 1, text=text) for i, text in enumerate(pages)],
    )


@pytest.fixture
def doc() -> ParsedDocument:
    """3 pages, 12 short paragraphs (~30 tokens each), page 2 is empty."""
    page1 = "\n\n".join(paragraph(2, f"p1-{i}") for i in range(6))
    page3 = "\n\n".join(paragraph(2, f"p3-{i}") for i in range(6))
    return make_doc([page1, "", page3])


def test_helpers() -> None:
    assert count_tokens("") == 0
    assert 5 <= count_tokens(SENTENCE) <= 25
    assert split_paragraphs("a\n\nb\n \n\nc\n") == ["a", "b", "c"]


def test_chunks_respect_max_tokens_and_report_true_counts(doc: ParsedDocument) -> None:
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=0)
    assert len(chunks) >= 3
    for c in chunks:
        assert isinstance(c, TextChunk)
        assert 0 < c.token_count <= 100
        assert c.token_count == count_tokens(c.text)


def test_chunk_index_is_sequential(doc: ParsedDocument) -> None:
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_pages_are_valid_and_never_go_backwards(doc: ParsedDocument) -> None:
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=0)
    pages = [c.page for c in chunks]
    assert set(pages) <= {1, 3}  # page 2 is empty, so no chunk may claim it
    assert pages == sorted(pages)
    assert chunks[0].page == 1 and chunks[-1].page == 3


def test_every_paragraph_is_kept(doc: ParsedDocument) -> None:
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=0)
    all_text = "\n".join(c.text for c in chunks)
    for page in doc.pages:
        for para in split_paragraphs(page.text):
            assert para in all_text


def test_overlap_repeats_end_of_previous_chunk(doc: ParsedDocument) -> None:
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        tail = " ".join(prev.text.split()[-3:])
        assert tail in nxt.text, (
            f"chunk {nxt.chunk_index} does not start with the end of {prev.chunk_index}"
        )


def test_no_overlap_means_no_repetition(doc: ParsedDocument) -> None:
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=0)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.text.split()[-1:] != nxt.text.split()[:1] or "[" in nxt.text[:5]


def test_oversized_paragraph_is_split() -> None:
    huge = paragraph(60, "big")  # ~900 tokens in one paragraph
    chunks = chunk_document(make_doc([huge]), max_tokens=200, overlap_tokens=0)
    assert len(chunks) >= 4
    assert all(c.token_count <= 200 for c in chunks)
    assert all(c.page == 1 for c in chunks)


def test_empty_document_returns_nothing() -> None:
    assert chunk_document(make_doc([])) == []
    assert chunk_document(make_doc(["", "   \n\n  "])) == []


def test_is_deterministic(doc: ParsedDocument) -> None:
    assert chunk_document(doc) == chunk_document(doc)
