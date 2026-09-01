"""End-to-end with fakes for embedder/reranker/LLM; needs the Docker database."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text

from docmind.api.app import build_app
from docmind.db import get_engine, get_session
from docmind.ingest.embedder import FakeEmbedder
from docmind.ingest.pipeline import ingest_file
from docmind.llm.backends import FakeLLM
from docmind.models import AuditLog, Document
from docmind.rag import RagConfig, answer_question
from docmind.retrieval.reranker import FakeReranker
from tests.test_parser import make_pdf

PAGE = (
    "Artikel 9 Selbstbehalt. Der Selbstbehalt betraegt 200 Franken pro Schadenfall. "
    "Bei Glasbruch entfaellt der Selbstbehalt. "
) * 10


@pytest.fixture(scope="module")
def corpus():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            ok = conn.execute(text("select to_regclass('chunks')")).scalar()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database not reachable: {exc}")
    if not ok:
        pytest.skip("run alembic upgrade head")
    tmp = Path(__file__).parent / "_tmp_rag"
    tmp.mkdir(exist_ok=True)
    with get_session() as session:
        session.execute(delete(Document).where(Document.filename.like("test-rag-%")))
        session.commit()
        report = ingest_file(make_pdf(tmp / "test-rag.pdf", [PAGE]), session, FakeEmbedder())
        assert report.status == "ingested"
        doc_id = session.scalar(select(Document.id).where(Document.filename == "test-rag.pdf"))
        # Restrict every query to the test document so the real corpus cannot interfere.
        yield session, [doc_id]
        session.execute(delete(Document).where(Document.filename.like("test-rag-%")))
        session.commit()
    for f in tmp.glob("*.pdf"):
        f.unlink()
    tmp.rmdir()


def test_answer_question_returns_citations_and_timings(corpus) -> None:
    session, doc_ids = corpus
    llm = FakeLLM("Der Selbstbehalt betraegt 200 Franken [1].")
    result = answer_question(
        session,
        "Wie hoch ist der Selbstbehalt?",
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=llm,
        config=RagConfig(k=3),
        document_ids=doc_ids,
    )
    assert result.lang == "de"
    assert result.answer.startswith("Der Selbstbehalt")
    assert result.citations and result.citations[0].filename == "test-rag.pdf"
    assert result.citations[0].page == 1
    assert result.contexts and result.retrieved_ids
    assert result.timings.total_s >= 0
    system, user = llm.calls[0]
    assert "Answer in German" in system and "Selbstbehalt" in user


def test_answer_question_without_matches_does_not_call_llm(corpus) -> None:
    session, _ = corpus
    llm = FakeLLM()
    result = answer_question(
        session,
        "zqxjv wpltk",  # keyword side finds nothing; restrict vector side to no documents
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=llm,
        document_ids=[-1],
    )
    assert llm.calls == []
    assert result.citations == [] and "could not find" in result.answer


def test_api_ask_health_metrics(corpus) -> None:
    _, doc_ids = corpus
    app = build_app(embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM("Ja [1]."))
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["database"] == "ok"

        r = client.post(
            "/ask",
            json={"question": "Entfaellt der Selbstbehalt bei Glasbruch?", "document_ids": doc_ids},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"] == "Ja [1]."
        assert body["citations"][0]["filename"] == "test-rag.pdf"
        assert body["backend"] == "fake" and body["latency_s"] >= 0

        assert client.post("/ask", json={"question": "x"}).status_code == 422
        assert client.post("/ask", json={"question": "abc", "lang": "xx"}).status_code == 422

        m = client.get("/metrics").json()
        assert m["requests_total"] == 1 and m["latency_p50_s"] is not None

        # every answered question leaves an audit row
        session, _ = corpus
        row = session.get(AuditLog, body["audit_id"])
        assert row is not None and row.username == "anonymous" and row.status == "ok"
        assert row.cited_chunk_ids == [body["citations"][0]["chunk_id"]]
        session.delete(row)
        session.commit()


def test_api_basic_auth(corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    from docmind.config import get_settings

    monkeypatch.setenv("BASIC_AUTH_USER", "alice")
    monkeypatch.setenv("BASIC_AUTH_PASSWORD", "s3cret")
    get_settings.cache_clear()
    try:
        app = build_app(embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM("Ja [1]."))
        with TestClient(app) as client:
            assert (
                client.post("/ask", json={"question": "Wie hoch ist der Selbstbehalt?"}).status_code
                == 401
            )
            r = client.post(
                "/ask",
                json={"question": "Wie hoch ist der Selbstbehalt?"},
                auth=("alice", "s3cret"),
            )
            assert r.status_code == 200
            session, _ = corpus
            row = session.get(AuditLog, r.json()["audit_id"])
            assert row.username == "alice"
            session.delete(row)
            session.commit()
    finally:
        get_settings.cache_clear()
