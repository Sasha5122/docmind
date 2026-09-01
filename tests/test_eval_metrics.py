"""Metrics are pure functions; the judge is exercised with a FakeLLM."""

import json
from pathlib import Path

import pytest

from docmind.eval.golden import GoldenItem, SourceRef, load_golden
from docmind.eval.metrics import (
    citation_coverage,
    citation_precision,
    judge_correctness,
    judge_faithfulness,
    mean,
    mrr,
    recall_at_k,
)
from docmind.llm.backends import FakeLLM
from docmind.llm.prompt import Citation
from docmind.retrieval.search import RetrievedChunk


def hit(i: int, filename: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(i, 1, filename, None, page, "de", "t", 0.0)


EXPECTED = [SourceRef("a.pdf", 10)]


def test_recall_at_k_and_page_tolerance() -> None:
    retrieved = [hit(1, "b.pdf", 10), hit(2, "a.pdf", 11), hit(3, "a.pdf", 10)]
    assert recall_at_k(retrieved, EXPECTED, k=1) == 0.0
    assert recall_at_k(retrieved, EXPECTED, k=2) == 1.0  # page 11 is within ±1 of 10
    assert recall_at_k([hit(1, "a.pdf", 13)], EXPECTED, k=5) == 0.0
    assert recall_at_k(retrieved, [], k=5) == 0.0


def test_mrr() -> None:
    assert mrr([hit(1, "b.pdf", 1), hit(2, "a.pdf", 10)], EXPECTED) == 0.5
    assert mrr([hit(1, "b.pdf", 1)], EXPECTED) == 0.0


def test_citation_precision_and_coverage() -> None:
    cites = [Citation(1, 1, "a.pdf", 10, "t"), Citation(2, 2, "b.pdf", 3, "t")]
    assert citation_precision(cites, EXPECTED) == 0.5
    assert citation_precision([], EXPECTED) is None
    text = "Der Selbstbehalt betraegt 200 Franken [1]. Das ist alles hier."
    assert citation_coverage(text) == 0.5
    assert citation_coverage("Ja.") == 1.0


def test_judges_parse_json_even_with_prose() -> None:
    faithful = FakeLLM(
        'Sure! {"statements": [{"text": "a", "supported": true}, '
        '{"text": "b", "supported": false}]}'
    )
    assert judge_faithfulness(faithful, "a. b.", ["ctx"]).score == 0.5
    correct = FakeLLM('```json\n{"score": 1.0, "reason": "same facts"}\n```')
    assert judge_correctness(correct, "q", "a", "ref").score == 1.0
    with pytest.raises(ValueError):
        judge_correctness(FakeLLM("no json here"), "q", "a", "ref")


def test_mean_ignores_none() -> None:
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def test_load_golden_validates(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    rows = [
        {
            "id": "q1",
            "lang": "de",
            "question": "x?",
            "reference_answer": "y",
            "sources": [{"file": "a.pdf", "page": 3}],
        },
        {"id": "q2", "lang": "en", "question": "z?", "category": "unanswerable"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items = load_golden(path)
    assert isinstance(items[0], GoldenItem) and items[0].sources == (SourceRef("a.pdf", 3),)
    assert not items[1].answerable
    assert len(load_golden(path, limit=1)) == 1

    rows.append(
        {"id": "q1", "lang": "de", "question": "dup?", "sources": [{"file": "a", "page": 1}]}
    )
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)
