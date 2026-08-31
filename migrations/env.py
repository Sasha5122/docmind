"""Alembic entry point: tells Alembic which database to talk to and which tables exist.

- The URL comes from `docmind.config` (so `.env` is the single source of truth).
- `target_metadata` is our `Base.metadata`; importing `docmind.models` registers
  the tables on it so `alembic revision --autogenerate` can diff code vs database.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

import docmind.models  # noqa: F401  (registers tables on Base.metadata)
from docmind.config import get_settings
from docmind.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade head --sql`)."""
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations."""
    engine = create_engine(get_settings().database_url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
