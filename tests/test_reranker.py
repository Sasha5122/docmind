"""Reranker contract; the real cross-encoder only runs with DOCMIND_RUN_SLOW=1."""

import os

import pytest

from docmind.retrieval.reranker import (
    CrossEncoderReranker,
    FakeReranker,
    NoopReranker,
    get_reranker,
)
from docmind.retrieval.search import RetrievedChunk


def chunk(chunk_id: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, 1, "doc.pdf", None, 1, "de", text, 0.0)


CANDIDATES = [
    chunk(1, "Der Zug nach Bern faehrt um neun Uhr ab."),
    chunk(2, "Die Versicherung deckt Schaeden durch Feuer und Hagel."),
    chunk(3, "Versicherung: Schaeden durch Feuer sind ausgeschlossen, Hagel gedeckt."),
]


def test_fake_reranker_orders_by_overlap_and_truncates() -> None:
    out = FakeReranker().rerank("Deckt die Versicherung Schaeden durch Feuer?", CANDIDATES, 2)
    assert [c.chunk_id for c in out] == [2, 3]
    assert out[0].score >= out[1].score > 0


def test_noop_reranker_keeps_order() -> None:
    out = NoopReranker().rerank("anything", CANDIDATES, 2)
    assert [c.chunk_id for c in out] == [1, 2]


def test_get_reranker_switch() -> None:
    assert isinstance(get_reranker(enabled=False), NoopReranker)
    real = get_reranker(enabled=True)
    assert isinstance(real, CrossEncoderReranker) and real._model is None  # lazy


def test_empty_candidates() -> None:
    assert CrossEncoderReranker().rerank("q", [], 5) == []


@pytest.mark.skipif(os.environ.get("DOCMIND_RUN_SLOW") != "1", reason="loads ~2 GB model")
def test_real_reranker_prefers_the_answering_chunk() -> None:
    out = CrossEncoderReranker().rerank(
        "Quels dommages l'assurance couvre-t-elle ?", CANDIDATES, 3
    )  # French question, German chunks
    assert out[0].chunk_id in {2, 3}
    assert out[-1].chunk_id == 1
