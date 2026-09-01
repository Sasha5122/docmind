"""add audit_log table

Revision ID: b91f3c7d2e55
Revises: 7c2e1a9b4d10
Create Date: 2026-09-01

One row per /ask call: who asked what, which chunks were retrieved and cited, which model
answered, how many tokens, what it cost and how long it took. This is the compliance trail.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b91f3c7d2e55"
down_revision: str | Sequence[str] | None = "7c2e1a9b4d10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("lang", sa.String(length=2), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("backend", sa.String(length=20), nullable=True),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("retrieval_mode", sa.String(length=10), nullable=True),
        sa.Column("retrieved_chunk_ids", postgresql.JSONB(), nullable=True),
        sa.Column("cited_chunk_ids", postgresql.JSONB(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(12, 8), nullable=True),
        sa.Column("latency_s", sa.Float(), nullable=True),
        sa.Column("retrieval_s", sa.Float(), nullable=True),
        sa.Column("rerank_s", sa.Float(), nullable=True),
        sa.Column("llm_s", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=10), nullable=False),  # ok | error
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])
    op.create_index("ix_audit_log_username", "audit_log", ["username"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_username", table_name="audit_log")
    op.drop_index("ix_audit_log_created_at", table_name="audit_log")
    op.drop_table("audit_log")
