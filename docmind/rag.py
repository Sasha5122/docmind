"""The question-answering pipeline, end to end:

    question -> hybrid retrieval (top `candidates`) -> reranker (top `k`)
             -> prompt with numbered sources -> LLM -> answer + citations + timings

Everything injectable (embedder, reranker, llm) so tests run with fakes and the eval
harness can swap one piece at a time.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session

from docmind.ingest.embedder import Embedder
from docmind.ingest.parser import detect_language
from docmind.llm.base import LLM
from docmind.llm.prompt import Citation, build_prompt, extract_citations
from docmind.retrieval.hybrid import hybrid_search
from docmind.retrieval.reranker import Reranker
from docmind.retrieval.search import RetrievedChunk


@dataclass(frozen=True)
class Timings:
    retrieval_s: float
    rerank_s: float
    llm_s: float
    total_s: float


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    lang: str
    citations: list[Citation]
    contexts: list[RetrievedChunk]  # what the model saw, in prompt order ([1], [2], ...)
    retrieved_ids: list[int]  # every candidate id before reranking (for recall@k)
    model: str
    backend: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timings: Timings
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RagConfig:
    k: int = 5  # chunks handed to the LLM
    candidates: int = 20  # chunks fetched per retriever before fusion/reranking
    retrieval_mode: str = "hybrid"  # hybrid | vector | keyword
    max_tokens: int = 800


def answer_question(
    session: Session,
    question: str,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLM,
    config: RagConfig | None = None,
    lang: str | None = None,
    document_ids: list[int] | None = None,
) -> Answer:
    config = config or RagConfig()
    started = time.perf_counter()
    answer_lang = lang or detect_language(question) or "en"

    t0 = time.perf_counter()
    retrieved = hybrid_search(
        session,
        question,
        embedder,
        k=config.candidates,
        candidates=config.candidates,
        document_ids=document_ids,
        mode=config.retrieval_mode,
    )
    retrieval_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    contexts = reranker.rerank(question, retrieved.chunks, config.k)
    rerank_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if contexts:
        system, user = build_prompt(question, contexts, answer_lang)
        response = llm.complete(system, user, max_tokens=config.max_tokens)
        text = response.text.strip()
    else:
        response = None
        text = _no_context_message(answer_lang)
    llm_s = time.perf_counter() - t0

    return Answer(
        question=question,
        answer=text,
        lang=answer_lang,
        citations=extract_citations(text, contexts),
        contexts=contexts,
        retrieved_ids=[c.chunk_id for c in retrieved.chunks],
        model=response.model if response else "",
        backend=llm.name,
        input_tokens=response.input_tokens if response else 0,
        output_tokens=response.output_tokens if response else 0,
        cost_usd=response.cost_usd if response else 0.0,
        timings=Timings(retrieval_s, rerank_s, llm_s, time.perf_counter() - started),
        meta={
            "retrieval_mode": config.retrieval_mode,
            "vector_ids": retrieved.vector_ids,
            "keyword_ids": retrieved.keyword_ids,
        },
    )


def _no_context_message(lang: str) -> str:
    return {
        "de": "Dazu finde ich in den Dokumenten keine Angaben.",
        "fr": "Je ne trouve aucune information à ce sujet dans les documents.",
        "it": "Non trovo informazioni al riguardo nei documenti.",
    }.get(lang, "I could not find anything about this in the documents.")
