"""Sanity checks on the table definitions (no database needed)."""

from docmind.db import Base
from docmind.models import EMBEDDING_DIM, Chunk, Document


def test_both_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {"documents", "chunks", "audit_log"}


def test_chunk_has_citation_fields_and_embedding() -> None:
    cols = Chunk.__table__.columns
    for name in ("document_id", "page", "chunk_index", "text", "embedding"):
        assert name in cols
    assert cols["embedding"].type.dim == EMBEDDING_DIM


def test_deleting_a_document_cascades_to_chunks() -> None:
    fk = next(iter(Chunk.__table__.columns["document_id"].foreign_keys))
    assert fk.ondelete == "CASCADE"
    assert Document.chunks.property.cascade.delete_orphan
