"""Regression gate for CI: compare a new eval summary against the committed baseline.

    python -m docmind.eval.gate reports/baseline.json reports/latest.json --max-drop 5

Exit code 1 (and a readable table) if any gated metric dropped by more than `max_drop`
percentage points. Metrics that are missing on either side (e.g. no judge in CI) are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATED = ("faithfulness", "correctness", "recall_at_5", "citation_precision")


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", data)  # accept a full report or a bare summary


def compare(baseline: dict, candidate: dict, max_drop_points: float) -> tuple[bool, list[str]]:
    ok = True
    lines = [f"{'metric':20} {'baseline':>9} {'new':>9} {'delta':>8}  verdict"]
    for metric in GATED:
        b, c = baseline.get(metric), candidate.get(metric)
        if b is None or c is None:
            lines.append(f"{metric:20} {'-':>9} {'-':>9} {'-':>8}  skipped")
            continue
        delta = (c - b) * 100
        failed = delta < -max_drop_points
        ok &= not failed
        lines.append(
            f"{metric:20} {b * 100:8.1f}% {c * 100:8.1f}% {delta:+7.1f}p  "
            f"{'FAIL' if failed else 'ok'}"
        )
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docmind.eval.gate")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-drop", type=float, default=5.0, help="percentage points")
    args = parser.parse_args(argv)
    ok, lines = compare(load_summary(args.baseline), load_summary(args.candidate), args.max_drop)
    print("\n".join(lines))
    print(
        "\nGATE PASSED"
        if ok
        else f"\nGATE FAILED: a metric dropped by more than {args.max_drop} points"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
