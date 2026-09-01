"""The two basic retrievers: search by meaning (vectors) and search by words (full-text).

Both return the same `RetrievedChunk` shape, ranked best-first, so the hybrid layer
can merge them without caring which one produced a hit.

- `vector_search`  : pgvector cosine distance between the query vector and `chunks.embedding`.
                     Works across languages because bge-m3 maps DE/FR/EN/IT into one space.
- `keyword_search` : Postgres full-text search on the generated `chunks.tsv` column, terms
                     OR-ed, ranked with `ts_rank_cd`. Exact terms (article numbers, product
                     names, "Art. 12") are where this side wins over vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Select, Text, func, select
from sqlalchemy.orm import Session

from docmind.ingest.parser import detect_language
from docmind.models import Chunk, Document

# ISO code -> Postgres text-search configuration (stemmer + stop words)
TS_CONFIGS = {"de": "german", "fr": "french", "it": "italian", "en": "english"}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    title: str | None
    page: int
    lang: str | None
    text: str
    score: float  # higher is better; scale differs per retriever

    @property
    def citation(self) -> str:
        return f"[{self.filename}, p. {self.page}]"


def _base_query(lang: str | None, document_ids: Sequence[int] | None) -> Select:
    query = select(Chunk, Document.filename, Document.title).join(Chunk.document)
    if lang:
        query = query.where(Chunk.lang == lang)
    if document_ids:
        query = query.where(Chunk.document_id.in_(list(document_ids)))
    return query


def _rows_to_chunks(rows, scores: list[float]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            title=title,
            page=chunk.page,
            lang=chunk.lang,
            text=chunk.text,
            score=score,
        )
        for (chunk, filename, title), score in zip(rows, scores, strict=True)
    ]


def vector_search(
    session: Session,
    query_vector: Sequence[float],
    k: int = 20,
    lang: str | None = None,
    document_ids: Sequence[int] | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks by cosine similarity (score = 1 - cosine distance, so 1.0 = identical)."""
    distance = Chunk.embedding.cosine_distance(list(query_vector)).label("distance")
    query = (
        _base_query(lang, document_ids)
        .add_columns(distance)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )
    rows = session.execute(query).all()
    return _rows_to_chunks(
        [(r.Chunk, r.filename, r.title) for r in rows], [1.0 - float(r.distance) for r in rows]
    )


def ts_config_for(query: str, lang: str | None = None) -> str:
    """Pick the stemmer for the query: explicit lang, else detected, else 'simple'."""
    code = lang or detect_language(query)
    return TS_CONFIGS.get(code or "", "simple")


def _or_tsquery(config: str, query: str):
    """Stemmed query terms joined with OR.

    `plainto_tsquery` removes stop words and stems, but joins terms with AND, so a natural
    question ("Wie lang ist die Kuendigungsfrist?") only matches chunks containing EVERY
    word. Swapping `&` for `|` keeps stemming and lets `ts_rank_cd` reward chunks that
    match more of the terms — a poor man's BM25.
    """
    anded = func.plainto_tsquery(config, query)  # e.g. 'lang' & 'kundigungsfrist'
    # Parse the rewritten string with the 'simple' config: the terms are already stemmed,
    # and running the German/French stemmer twice would mangle them.
    return func.to_tsquery("simple", func.replace(func.cast(anded, Text), "&", "|"))


def keyword_search(
    session: Session,
    query: str,
    k: int = 20,
    lang: str | None = None,
    document_ids: Sequence[int] | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks by full-text rank. `lang` filters chunks AND selects the query stemmer."""
    if not query.strip():
        return []
    config = ts_config_for(query, lang)
    tsquery = _or_tsquery(config, query)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery).label("rank")
    stmt = (
        _base_query(lang, document_ids)
        .add_columns(rank)
        .where(Chunk.tsv.op("@@")(tsquery))
        .order_by(rank.desc(), Chunk.id)
        .limit(k)
    )
    rows = session.execute(stmt).all()
    return _rows_to_chunks(
        [(r.Chunk, r.filename, r.title) for r in rows], [float(r.rank) for r in rows]
    )
