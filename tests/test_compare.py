"""The experiments comparison table."""

import json
from pathlib import Path

from docmind.eval.compare import main, table


def report(label: str, recall: float, created: str) -> dict:
    return {
        "label": label,
        "n": 20,
        "created_at": created,
        "config": {"llm_backend": "ollama", "llm_model": "qwen2.5:7b"},
        "summary": {"recall_at_5": recall, "latency_p50_s": 1.5, "cost_usd_per_1000": 0.0},
    }


def test_table_has_one_row_per_report() -> None:
    md = table(
        [
            report("baseline", 0.8, "2026-09-01T10:00:00"),
            report("vector", 0.7, "2026-09-01T11:00:00"),
        ]
    )
    assert md.count("\n") == 4  # header, separator, two rows
    assert "| baseline | 20 | ollama:qwen2.5:7b | 80 %" in md
    assert "| vector | 20 | ollama:qwen2.5:7b | 70 %" in md


def test_main_sorts_by_time_and_writes(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(report("later", 0.5, "2026-09-02T00:00:00")), encoding="utf-8")
    b.write_text(json.dumps(report("earlier", 0.6, "2026-09-01T00:00:00")), encoding="utf-8")
    out = tmp_path / "exp.md"
    assert main([str(a), str(b), "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[2].startswith("| earlier") and lines[3].startswith("| later")
