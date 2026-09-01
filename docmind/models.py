"""Table definitions: one row per source document, one row per text chunk.

`Chunk.embedding` is a pgvector column of 1024 floats — the output size of the
multilingual embedding model `BAAI/bge-m3` chosen in CLAUDE.md.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docmind.db import Base

EMBEDDING_DIM = 1024

# Full-text "bag of stemmed words", built by Postgres with the stemmer for the chunk's language.
# Keep in sync with migrations/versions/7c2e1a9b4d10_*.py.
TSV_SQL = (
    "to_tsvector(CASE lang WHEN 'de' THEN 'german'::regconfig "
    "WHEN 'fr' THEN 'french'::regconfig WHEN 'it' THEN 'italian'::regconfig "
    "ELSE 'english'::regconfig END, text)"
)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Hash of the file bytes: lets ingest skip a PDF it has already seen.
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    lang: Mapped[str | None] = mapped_column(String(2))  # ISO 639-1: de, fr, en, it
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # 0, 1, 2 ... within a doc
    page: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based page the chunk starts on
    lang: Mapped[str | None] = mapped_column(String(2))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    tsv = mapped_column(TSVECTOR, Computed(TSV_SQL, persisted=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class AuditLog(Base):
    """One row per question asked through the API — the compliance trail.

    Stores what was asked, what was retrieved and cited, which model answered, and the
    cost/latency. The answer text is stored too, so a reviewer can reproduce a complaint.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[str | None] = mapped_column(String(2))
    answer: Mapped[str | None] = mapped_column(Text)
    backend: Mapped[str | None] = mapped_column(String(20))
    model: Mapped[str | None] = mapped_column(String(100))
    retrieval_mode: Mapped[str | None] = mapped_column(String(10))
    retrieved_chunk_ids: Mapped[list[int] | None] = mapped_column(JSONB)
    cited_chunk_ids: Mapped[list[int] | None] = mapped_column(JSONB)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(12, 8))
    latency_s: Mapped[float | None] = mapped_column(Float)
    retrieval_s: Mapped[float | None] = mapped_column(Float)
    rerank_s: Mapped[float | None] = mapped_column(Float)
    llm_s: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(10), nullable=False)  # ok | error
    error: Mapped[str | None] = mapped_column(Text)
