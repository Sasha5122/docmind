"""Turn several eval reports into one comparison table (for the README experiments section).

python -m docmind.eval.compare reports/eval_*_baseline.json reports/eval_*_vector.json ...
python -m docmind.eval.compare reports/eval_*.json --out reports/experiments.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COLUMNS = [
    ("recall_at_5", "recall@5", "pct"),
    ("recall_at_10", "recall@10", "pct"),
    ("mrr", "MRR", "num"),
    ("context_hit", "ctx hit@k", "pct"),
    ("citation_precision", "cit. precision", "pct"),
    ("faithfulness", "faithfulness", "pct"),
    ("correctness", "correctness", "pct"),
    ("abstention_rate", "abstain", "pct"),
    ("latency_p50_s", "p50 s", "num"),
    ("latency_p95_s", "p95 s", "num"),
    ("cost_usd_per_1000", "$/1000 q", "usd"),
]


def _fmt(value, kind: str) -> str:
    if value is None:
        return "–"
    if kind == "pct":
        return f"{100 * value:.0f} %"
    if kind == "usd":
        return f"{value:.2f}"
    return f"{value:.2f}"


def table(reports: list[dict]) -> str:
    head = "| run | n | LLM | " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    sep = "|---|---|---|" + "---|" * len(COLUMNS)
    rows = [head, sep]
    for r in reports:
        s, c = r["summary"], r["config"]
        cells = [_fmt(s.get(key), kind) for key, _, kind in COLUMNS]
        rows.append(
            f"| {r['label']} | {r['n']} | {c.get('llm_backend')}:{c.get('llm_model')} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docmind.eval.compare")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
    loaded = [r for r in loaded if "summary" in r]
    loaded.sort(key=lambda r: r["created_at"])
    md = table(loaded)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"written {args.out}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
