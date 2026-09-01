"""Regression gate arithmetic and the no-op tracer."""

import json
from pathlib import Path

from docmind.config import Settings
from docmind.eval.gate import compare, load_summary, main
from docmind.observability import Tracer, build_tracer


def test_compare_flags_only_real_drops() -> None:
    base = {
        "faithfulness": 0.90,
        "correctness": 0.80,
        "recall_at_5": 0.85,
        "citation_precision": 0.7,
    }
    ok, lines = compare(base, {**base, "faithfulness": 0.86}, max_drop_points=5)
    assert ok  # -4 points is within tolerance
    ok, lines = compare(base, {**base, "faithfulness": 0.84}, max_drop_points=5)
    assert not ok and any("FAIL" in line for line in lines)
    ok, _ = compare(base, {"faithfulness": None, "correctness": 0.9}, max_drop_points=5)
    assert ok  # missing metrics are skipped, improvements pass


def test_main_reads_full_report_or_bare_summary(tmp_path: Path) -> None:
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    base.write_text(json.dumps({"summary": {"recall_at_5": 0.8}}), encoding="utf-8")
    cand.write_text(json.dumps({"recall_at_5": 0.5}), encoding="utf-8")
    assert load_summary(base) == {"recall_at_5": 0.8}
    assert main([str(base), str(cand)]) == 1
    assert main([str(base), str(cand), "--max-drop", "40"]) == 0


def test_tracer_is_noop_without_keys() -> None:
    tracer = build_tracer(Settings(_env_file=None, langfuse_public_key=""))
    assert isinstance(tracer, Tracer) and not tracer.enabled
    tracer.flush()  # must not raise
