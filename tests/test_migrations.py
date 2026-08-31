"""Runs the real Alembic migrations against the Docker Postgres.

Skipped automatically when the database is not reachable (e.g. Docker off),
so the fast unit tests still run anywhere.
"""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from docmind.db import get_engine


@pytest.fixture(scope="module")
def migrated_db():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database not reachable: {exc}")
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    yield engine


def test_migration_creates_both_tables(migrated_db) -> None:
    tables = set(inspect(migrated_db).get_table_names())
    assert {"documents", "chunks"} <= tables


def test_vector_extension_is_enabled(migrated_db) -> None:
    with migrated_db.connect() as conn:
        version = conn.execute(
            text("select extversion from pg_extension where extname = 'vector'")
        ).scalar()
    assert version is not None


def test_downgrade_then_upgrade_round_trips(migrated_db) -> None:
    cfg = Config("alembic.ini")
    command.downgrade(cfg, "base")
    assert "chunks" not in inspect(migrated_db).get_table_names()
    command.upgrade(cfg, "head")
    assert "chunks" in inspect(migrated_db).get_table_names()
