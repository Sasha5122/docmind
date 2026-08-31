"""Database connection setup.

One engine (= the pool of connections to Postgres) for the whole process, built
from the URL in `docmind.config`. Every table class inherits from `Base` so
SQLAlchemy and Alembic can see them all in one place (`Base.metadata`).
"""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from docmind.config import get_settings


class Base(DeclarativeBase):
    """Parent of every table class in `docmind.models`."""


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine (created once, then reused)."""
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def get_session() -> Session:
    """Open a new unit of work. Use as `with get_session() as session:`."""
    return sessionmaker(bind=get_engine())()
