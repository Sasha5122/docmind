"""add full-text column and search indexes to chunks

Revision ID: 7c2e1a9b4d10
Revises: 31ba3f51a986
Create Date: 2026-09-01

- `tsv`: a stored, generated tsvector (Postgres' bag of stemmed words) built with the
  stemmer matching the chunk's language. Postgres keeps it up to date itself.
- GIN index on `tsv`  -> fast keyword (BM25-style) search
- HNSW index on `embedding` with cosine ops -> fast approximate nearest-neighbour search
"""

from collections.abc import Sequence

from alembic import op

revision: str = "7c2e1a9b4d10"
down_revision: str | Sequence[str] | None = "31ba3f51a986"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same expression lives in docmind/models.py (Chunk.tsv) so the ORM knows the column.
TSV_EXPRESSION = """
to_tsvector(
    CASE lang
        WHEN 'de' THEN 'german'::regconfig
        WHEN 'fr' THEN 'french'::regconfig
        WHEN 'it' THEN 'italian'::regconfig
        ELSE 'english'::regconfig
    END,
    text
)
"""


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS ({TSV_EXPRESSION}) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN tsv")
