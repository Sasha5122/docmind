"""Second-stage ranking: read each (question, chunk) pair together and score the match.

The first stage (hybrid retrieval) is fast but shallow: it compares the question to each
chunk through a vector or a bag of words. A cross-encoder reranker feeds the question AND
the chunk text into one model pass, so it can see whether the chunk actually answers the
question. It is slow per pair, which is why it only runs on the top ~20 candidates.

`BAAI/bge-reranker-v2-m3` is multilingual, like bge-m3, so a French question can be
scored against a German chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from docmind.retrieval.search import RetrievedChunk

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class Reranker(Protocol):
    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the `top_k` best chunks, best first, with `score` = reranker score."""
        ...


class CrossEncoderReranker:
    """bge-reranker-v2-m3 via sentence-transformers' CrossEncoder (lazy-loaded, ~2.2 GB)."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(question, chunk.text) for chunk in chunks]
        scores = self._load().predict(pairs, batch_size=self.batch_size)
        return _top(chunks, [float(s) for s in scores], top_k)


class NoopReranker:
    """Keeps the incoming order — the 'reranker off' arm of the M3 experiment."""

    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(chunks[:top_k])


class FakeReranker:
    """For tests: scores a chunk by how many question words it contains."""

    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        words = {w.lower() for w in question.split()}
        scores = [
            float(sum(1 for w in chunk.text.lower().split() if w in words)) for chunk in chunks
        ]
        return _top(chunks, scores, top_k)


def _top(
    chunks: Sequence[RetrievedChunk], scores: Sequence[float], top_k: int
) -> list[RetrievedChunk]:
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: (-p[1], p[0].chunk_id))
    return [replace(chunk, score=score) for chunk, score in ranked[:top_k]]


def get_reranker(enabled: bool = True, model_name: str = DEFAULT_RERANKER) -> Reranker:
    return CrossEncoderReranker(model_name) if enabled else NoopReranker()
