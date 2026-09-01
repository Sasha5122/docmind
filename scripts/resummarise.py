"""Recompute a report's summary after marking wall-clock outliers (the laptop fell asleep
mid-question, so one question "took" 84 minutes).

    uv run python scripts/resummarise.py reports/eval_<stamp>_baseline.json --max-latency 600

Questions above the cap keep their scores (they were answered correctly) but are excluded
from the latency statistics; their ids are written into config.latency_outliers_excluded so
the exclusion is visible in every table built from the file. Rewrites the .json, the .md
next to it and reports/latest.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docmind.eval.metrics import is_abstention  # noqa: E402
from docmind.eval.runner import (  # noqa: E402
    EvalReport,
    QuestionResult,
    _grouped,
    markdown_summary,
    summarise,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--max-latency", type=float, default=600.0, help="seconds")
    parser.add_argument(
        "--rescore-abstention",
        action="store_true",
        help="re-detect 'not in the documents' answers with the current metrics.is_abstention "
        "(the stored answers are re-read; no model is called)",
    )
    args = parser.parse_args(argv)

    data = json.loads(args.report.read_text(encoding="utf-8"))
    names = {f.name for f in fields(QuestionResult)}
    results = [
        QuestionResult(**{k: v for k, v in r.items() if k in names}) for r in data["results"]
    ]
    outliers = [r.id for r in results if r.latency_s > args.max_latency]

    if args.rescore_abstention:
        for r in results:
            if r.error is not None:
                continue
            now = is_abstention(r.answer)
            if now == r.abstained:
                continue
            print(f"  {r.id} ({r.category}): abstained {r.abstained} -> {now}: {r.answer[:90]!r}")
            r.abstained = now
            answerable = r.category != "unanswerable"
            if not answerable:
                r.correctness = 1.0 if now else 0.0
                r.faithfulness = 1.0 if now else None
            elif now:
                # Reference says the answer exists: abstaining is wrong but invents nothing.
                r.correctness, r.faithfulness, r.judge_detail = 0.0, 1.0, {}
            r.citation_coverage = 1.0 if now else r.citation_coverage

    # Latency stats without the outliers; every other metric over all questions.
    kept = [r for r in results if r.id not in outliers]
    summary = summarise(results)
    for key, value in summarise(kept).items():
        if key.startswith(("latency_", "retrieval_", "rerank_", "llm_")):
            summary[key] = value
    config = {**data["config"], "latency_outliers_excluded": outliers,
              "latency_outlier_cap_s": args.max_latency}  # fmt: skip
    report = EvalReport(
        created_at=data["created_at"],
        label=data["label"],
        config=config,
        n=data["n"],
        summary=summary,
        by_lang=_grouped(results, "lang"),
        by_category=_grouped(results, "category"),
        results=results,
    )
    args.report.write_text(report.to_json(), encoding="utf-8")
    args.report.with_suffix(".md").write_text(markdown_summary(report), encoding="utf-8")
    latest = {"created_at": report.created_at, "label": report.label, "n": report.n,
              "config": report.config, "summary": report.summary}  # fmt: skip
    (args.report.parent / "latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{args.report.name}: excluded {outliers} from latency stats; "
          f"p50={summary['latency_p50_s']} p95={summary['latency_p95_s']} "
          f"mean={summary['latency_mean_s']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
