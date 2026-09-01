"""The ingestion pipeline: PDF -> redacted pages -> chunks -> vectors -> database rows.

Two layers so the interesting part is testable without a database:
- `prepare_document(parsed)`      pure transformation (redact + chunk), no I/O
- `ingest_file(path, session, embedder)`  parse, prepare, embed, store; skips files whose
  sha256 is already in `documents` so re-running the CLI is safe and cheap.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from docmind.ingest.chunker import TextChunk, chunk_document
from docmind.ingest.embedder import Embedder
from docmind.ingest.parser import Page, ParsedDocument, parse_pdf
from docmind.models import Chunk, Document
from docmind.pii.redactor import redact

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDocument:
    parsed: ParsedDocument  # pages already redacted
    chunks: list[TextChunk]
    pii_counts: dict[str, int] = field(default_factory=dict)  # e.g. {"PERSON": 3, "IBAN": 1}

    @property
    def pii_total(self) -> int:
        return sum(self.pii_counts.values())


@dataclass(frozen=True)
class IngestReport:
    filename: str
    status: str  # "ingested" | "skipped" | "failed"
    chunks: int = 0
    pii_total: int = 0
    seconds: float = 0.0
    error: str | None = None


def prepare_document(
    parsed: ParsedDocument, max_tokens: int = 500, overlap_tokens: int = 50
) -> PreparedDocument:
    """Redact every page, then chunk. PII never reaches the chunker or the database."""
    pages: list[Page] = []
    counts: dict[str, int] = {}
    for page in parsed.pages:
        result = redact(page.text, parsed.lang)
        pages.append(Page(number=page.number, text=result.text))
        for label, n in result.counts.items():
            counts[label] = counts.get(label, 0) + n
    redacted = ParsedDocument(
        filename=parsed.filename,
        sha256=parsed.sha256,
        lang=parsed.lang,
        pages=pages,
        title=parsed.title,
    )
    chunks = chunk_document(redacted, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    return PreparedDocument(parsed=redacted, chunks=chunks, pii_counts=counts)


def ingest_file(
    path: Path,
    session: Session,
    embedder: Embedder,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> IngestReport:
    """Ingest one PDF. Never raises: problems come back as a `failed` report."""
    started = time.perf_counter()
    try:
        parsed = parse_pdf(path)
        if session.scalar(select(Document.id).where(Document.sha256 == parsed.sha256)):
            log.info("skip %s (already ingested)", path.name)
            return IngestReport(path.name, "skipped", seconds=time.perf_counter() - started)

        prepared = prepare_document(parsed, max_tokens, overlap_tokens)
        vectors = embedder.embed([c.text for c in prepared.chunks])
        document = Document(
            filename=parsed.filename,
            sha256=parsed.sha256,
            title=parsed.title,
            lang=parsed.lang,
            page_count=parsed.page_count,
        )
        document.chunks = [
            Chunk(
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                lang=parsed.lang,
                text=chunk.text,
                token_count=chunk.token_count,
                embedding=vector,
            )
            for chunk, vector in zip(prepared.chunks, vectors, strict=True)
        ]
        session.add(document)
        session.commit()
        seconds = time.perf_counter() - started
        log.info(
            "ingested %s: %d pages, %d chunks, %d PII spans, %.1fs",
            path.name,
            parsed.page_count,
            len(prepared.chunks),
            prepared.pii_total,
            seconds,
        )
        return IngestReport(
            path.name, "ingested", len(prepared.chunks), prepared.pii_total, seconds
        )
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
        session.rollback()
        log.exception("failed %s", path.name)
        return IngestReport(
            path.name, "failed", seconds=time.perf_counter() - started, error=str(exc)
        )


def find_pdfs(target: Path) -> list[Path]:
    """A single PDF, or every *.pdf under a directory (sorted, recursive)."""
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*.pdf") if p.is_file())


def ingest_path(
    target: Path,
    session: Session,
    embedder: Embedder,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[IngestReport]:
    return [
        ingest_file(pdf, session, embedder, max_tokens, overlap_tokens) for pdf in find_pdfs(target)
    ]
