"""The eval runner end to end with fakes (needs the Docker database)."""

import json
from pathlib import Path

import pytest
from sqlalchemy import delete, select, text

from docmind.db import get_engine, get_session
from docmind.eval.golden import GoldenItem, SourceRef, load_golden
from docmind.eval.runner import markdown_summary, run_eval, summarise, write_report
from docmind.ingest.embedder import FakeEmbedder
from docmind.ingest.pipeline import ingest_file
from docmind.llm.backends import FakeLLM
from docmind.models import Document
from docmind.rag import RagConfig
from docmind.retrieval.reranker import FakeReranker
from tests.test_parser import make_pdf

PAGE = "Artikel 3 Kuendigungsfrist. Die Kuendigungsfrist betraegt drei Monate. " * 12


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
    tmp = Path(__file__).parent / "_tmp_eval"
    tmp.mkdir(exist_ok=True)
    with get_session() as session:
        session.execute(delete(Document).where(Document.filename.like("test-eval-%")))
        session.commit()
        ingest_file(make_pdf(tmp / "test-eval.pdf", [PAGE]), session, FakeEmbedder())
        doc_id = session.scalar(select(Document.id).where(Document.filename == "test-eval.pdf"))
        yield session, [doc_id]
        session.execute(delete(Document).where(Document.filename.like("test-eval-%")))
        session.commit()
    for f in tmp.glob("*.pdf"):
        f.unlink()
    tmp.rmdir()


def test_run_eval_scores_and_writes_report(corpus, tmp_path: Path) -> None:
    session, doc_ids = corpus
    items = [
        GoldenItem(
            "q1", "de", "Wie lang ist die Kuendigungsfrist?", "Drei Monate.",
            (SourceRef("test-eval.pdf", 1),),
        ),
        GoldenItem("q2", "de", "Wie hoch ist die Praemie fuer Haustiere?", "", (), "unanswerable"),
    ]  # fmt: skip
    llm = FakeLLM("Die Kuendigungsfrist betraegt drei Monate [1].")
    judge = FakeLLM(
        '{"statements": [{"text": "drei Monate", "supported": true}], "score": 1.0, "reason": "ok"}'
    )
    report = run_eval(
        session,
        items,
        FakeEmbedder(),
        FakeReranker(),
        llm,
        RagConfig(k=3),
        judge,
        label="unit",
        document_ids=doc_ids,  # keep the real corpus out of a unit test
    )
    q1, q2 = report.results
    assert q1.recall_at_5 == 1.0 and q1.mrr > 0
    assert q1.citation_precision == 1.0 and q1.faithfulness == 1.0 and q1.correctness == 1.0
    # q2 is unanswerable but the fake LLM answers anyway -> counted as wrong
    assert q2.recall_at_5 is None and q2.correctness == 0.0 and not q2.abstained
    assert report.summary["recall_at_5"] == 1.0 and report.summary["errors"] == 0
    assert report.by_lang["de"]["questions"] == 2

    json_path, md_path = write_report(report, tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["label"] == "unit" and len(data["results"]) == 2
    md = md_path.read_text(encoding="utf-8")
    assert "recall@5 | 100 %" in md and md == markdown_summary(report)


def test_summarise_empty() -> None:
    s = summarise([])
    assert s["questions"] == 0 and s["recall_at_5"] is None and s["latency_p95_s"] is None


def test_default_golden_file_loads_if_present() -> None:
    path = Path("data/eval/golden.jsonl")
    if not path.exists():
        pytest.skip("golden set not written yet")
    items = load_golden(path)
    assert len(items) >= 60
    assert {i.lang for i in items} >= {"de", "fr", "en"}
