"""Hybrid retrieval: merge "search by meaning" and "search by words" into one ranking.

Why both: vectors find paraphrases and cross-language matches but blur exact tokens;
full-text search nails exact tokens ("Art. 12", "FINMA-RS 2023/1", a product name) but
knows nothing about meaning. Regulatory questions need both.

How the merge works — Reciprocal Rank Fusion (RRF):
  Each retriever returns a ranked list. A chunk's fused score is the sum, over the lists
  it appears in, of 1 / (K + rank), rank starting at 1. K (default 60) softens the gap
  between rank 1 and rank 2 so one retriever cannot dominate. Scores of the two systems
  are never compared directly — only their ranks are — which is why RRF needs no tuning
  of incompatible score scales (cosine similarity vs ts_rank_cd).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from sqlalchemy.orm import Session

from docmind.ingest.embedder import Embedder
from docmind.retrieval.search import RetrievedChunk, keyword_search, vector_search

RRF_K = 60


@dataclass(frozen=True)
class HybridResult:
    chunks: list[RetrievedChunk]  # fused ranking, best first; `score` is the RRF score
    vector_ids: list[int]  # what each side contributed, for eval and debugging
    keyword_ids: list[int]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Merge several best-first lists into one; each item's score = sum of 1/(k + rank)."""
    fused: dict[int, float] = {}
    first_seen: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [replace(first_seen[chunk_id], score=score) for chunk_id, score in ordered]


def hybrid_search(
    session: Session,
    question: str,
    embedder: Embedder,
    k: int = 10,
    candidates: int = 20,
    lang: str | None = None,
    document_ids: Sequence[int] | None = None,
    mode: str = "hybrid",
) -> HybridResult:
    """Top-k chunks for `question`.

    `mode` = "hybrid" (default), "vector" or "keyword" — the switch used by the M3
    experiment "hybrid vs vector-only". `candidates` is how many each side fetches before
    fusion; more candidates = better recall, slightly slower.
    """
    if mode not in {"hybrid", "vector", "keyword"}:
        raise ValueError(f"unknown retrieval mode {mode!r}")

    vector_hits: list[RetrievedChunk] = []
    keyword_hits: list[RetrievedChunk] = []
    if mode in {"hybrid", "vector"}:
        query_vector = embedder.embed([question])[0]
        vector_hits = vector_search(session, query_vector, candidates, lang, document_ids)
    if mode in {"hybrid", "keyword"}:
        keyword_hits = keyword_search(session, question, candidates, lang, document_ids)

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
    return HybridResult(
        chunks=fused[:k],
        vector_ids=[c.chunk_id for c in vector_hits],
        keyword_ids=[c.chunk_id for c in keyword_hits],
    )
