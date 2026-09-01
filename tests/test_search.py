"""Vector and keyword search against the Docker database (skipped when it is down).

Uses FakeEmbedder: its vectors carry no meaning, but a chunk's own vector must come back
first, which is enough to prove the SQL, the index and the ordering are right.
"""

from pathlib import Path

import pytest
from sqlalchemy import delete, text

from docmind.db import get_engine, get_session
from docmind.ingest.embedder import FakeEmbedder
from docmind.ingest.pipeline import ingest_file
from docmind.models import Document
from docmind.retrieval.search import keyword_search, ts_config_for, vector_search
from tests.test_parser import make_pdf

GERMAN_PAGE = (
    "Artikel 12 Elementarschaeden. Versichert sind Schaeden durch Hochwasser, "
    "Ueberschwemmung, Sturm, Hagel, Lawinen und Schneedruck. "
) * 12
ENGLISH_PAGE = (
    "Article 7 Claims handling. The insurer settles a claim within thirty days after "
    "receiving all documents required to assess the loss. "
) * 12


@pytest.fixture(scope="module")
def corpus():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            has_tsv = conn.execute(
                text(
                    "select 1 from information_schema.columns "
                    "where table_name='chunks' and column_name='tsv'"
                )
            ).scalar()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database not reachable: {exc}")
    if not has_tsv:
        pytest.skip("run `uv run alembic upgrade head` first")

    tmp = Path(__file__).parent / "_tmp_search"
    tmp.mkdir(exist_ok=True)
    embedder = FakeEmbedder()
    with get_session() as session:
        session.execute(delete(Document).where(Document.filename.like("test-search-%")))
        session.commit()
        de = ingest_file(make_pdf(tmp / "test-search-de.pdf", [GERMAN_PAGE]), session, embedder)
        en = ingest_file(make_pdf(tmp / "test-search-en.pdf", [ENGLISH_PAGE]), session, embedder)
        assert de.status == en.status == "ingested"
        yield session, embedder
        session.execute(delete(Document).where(Document.filename.like("test-search-%")))
        session.commit()
    for f in tmp.glob("*.pdf"):
        f.unlink()
    tmp.rmdir()


def test_vector_search_returns_own_chunk_first(corpus) -> None:
    session, embedder = corpus
    hits = vector_search(session, embedder.embed(["anything"])[0], k=5)
    assert hits  # something comes back even for a meaningless query
    own = keyword_search(session, "Lawinen", k=1, lang="de")[0]
    [best, *_] = vector_search(session, embedder.embed([own.text])[0], k=3)
    assert best.chunk_id == own.chunk_id
    assert best.score == pytest.approx(1.0, abs=1e-4)


def test_vector_search_lang_filter(corpus) -> None:
    session, embedder = corpus
    hits = vector_search(session, embedder.embed(["x"])[0], k=50, lang="en")
    assert hits and all(h.lang == "en" for h in hits)


def test_keyword_search_finds_exact_term_and_stems(corpus) -> None:
    session, _ = corpus
    hits = keyword_search(session, "Hochwasser Hagel", k=5, lang="de")
    assert hits and hits[0].filename == "test-search-de.pdf"
    assert hits[0].citation == "[test-search-de.pdf, p. 1]"
    # English stemmer: "claims" matches "claim"
    hits = keyword_search(session, "settling claims", k=5, lang="en")
    assert hits and hits[0].filename == "test-search-en.pdf"


def test_keyword_search_no_match_and_empty_query(corpus) -> None:
    session, _ = corpus
    assert keyword_search(session, "xyzzyplugh", k=5) == []
    assert keyword_search(session, "   ", k=5) == []


def test_ts_config_detection() -> None:
    assert ts_config_for("Welche Schaeden sind versichert?", "de") == "german"
    assert ts_config_for("The insurer settles the claim within thirty days") == "english"
    assert ts_config_for("???") == "simple"
