"""Write the numbers from reports/ into README.md between the marker comments.

    uv run python scripts/fill_readme.py reports/eval_<stamp>_baseline.json [more reports...]

The first report is the baseline (RESULTS + BACKENDS sections); all reports go into the
EXPERIMENTS table. Re-runnable: it replaces whatever sits between the markers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docmind.eval.compare import table  # noqa: E402

README = Path(__file__).resolve().parent.parent / "README.md"


def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f} %"


def num(v, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def results_section(r: dict) -> str:
    s, c = r["summary"], r["config"]
    lines = [
        f"Baseline run `{r['label']}` — {r['n']} questions, LLM `{c['llm_backend']}:{c['llm_model']}`, "
        f"retrieval `{c['retrieval_mode']}` (k={c['k']}, {c['candidates']} candidates), "
        f"reranker `{c['reranker']}`, judge `{c['judge']}`, {r['created_at'][:10]}, "
        f"machine `{c.get('machine', '?')}`.",
        "",
        "| metric | all | " + " | ".join(r["by_lang"].keys()) + " |",
        "|---|---|" + "---|" * len(r["by_lang"]),
    ]
    for key, label in [
        ("recall_at_5", "recall@5"),
        ("context_hit", "context hit@k (after reranking)"),
        ("citation_precision", "citation precision"),
        ("faithfulness", "faithfulness (judge)"),
        ("correctness", "answer correctness (judge)"),
    ]:
        cells = [pct(g.get(key)) for g in r["by_lang"].values()]
        lines.append(f"| {label} | {pct(s.get(key))} | " + " | ".join(cells) + " |")
    lines += [
        f"| recall@10 | {pct(s.get('recall_at_10'))} | "
        + " | ".join("" for _ in r["by_lang"])
        + " |",
        f"| MRR | {num(s.get('mrr'))} | " + " | ".join("" for _ in r["by_lang"]) + " |",
        f"| citation coverage | {pct(s.get('citation_coverage'))} | "
        + " | ".join("" for _ in r["by_lang"])
        + " |",
        f"| abstention rate | {pct(s.get('abstention_rate'))} | "
        + " | ".join("" for _ in r["by_lang"])
        + " |",
        "",
        "| by category | n | recall@5 | citation precision | faithfulness | correctness |",
        "|---|---|---|---|---|---|",
    ]
    for cat, g in r["by_category"].items():
        lines.append(
            f"| {cat} | {g['questions']} | {pct(g.get('recall_at_5'))} | "
            f"{pct(g.get('citation_precision'))} | {pct(g.get('faithfulness'))} | "
            f"{pct(g.get('correctness'))} |"
        )
    lines += [
        "",
        f"**Latency** p50 **{num(s['latency_p50_s'])} s**, p95 **{num(s['latency_p95_s'])} s** "
        f"(mean: retrieval {num(s['retrieval_mean_s'])} s · rerank {num(s['rerank_mean_s'])} s · "
        f"LLM {num(s['llm_mean_s'])} s). Tokens per question ≈ {s['tokens_in_mean']} in / "
        f"{s['tokens_out_mean']} out. Cost per 1,000 questions: ${num(s['cost_usd_per_1000'])}.",
    ]
    return "\n".join(lines) + "\n"


def backends_section(r: dict) -> str:
    s, c = r["summary"], r["config"]
    tin, tout = s["tokens_in_mean"] or 0, s["tokens_out_mean"] or 0
    azure_cost = (tin * 0.15 + tout * 0.60) / 1_000_000 * 1000  # gpt-4o-mini list price
    return (
        f"| | Ollama `{c['llm_model']}` (RTX 3050 6 GB, measured) | Azure gpt-4o-mini (not measured — no key) |\n"
        "|---|---|---|\n"
        f"| faithfulness / correctness | {pct(s['faithfulness'])} / {pct(s['correctness'])} | run `LLM_BACKEND=azure` to fill |\n"
        f"| recall@5 / citation precision | {pct(s['recall_at_5'])} / {pct(s['citation_precision'])} | same retrieval → identical |\n"
        f"| latency p50 / p95 | {num(s['latency_p50_s'])} s / {num(s['latency_p95_s'])} s | typically 1.5–3 s end to end |\n"
        f"| cost per 1,000 questions | $0 (≈ {tin}+{tout} tokens/question, local GPU) | ≈ ${azure_cost:.2f} at list price |\n"
        "| data leaves the machine | no | yes (chosen Azure region) |\n"
    )


def failures_section(r: dict) -> str:
    rows = r["results"]
    misses = [x for x in rows if x.get("recall_at_5") == 0.0]
    wrong = [
        x
        for x in rows
        if x.get("correctness") is not None
        and x["correctness"] < 0.5
        and x.get("recall_at_5") == 1.0
    ]
    halluc = [x for x in rows if x["category"] == "unanswerable" and not x["abstained"]]
    lines = [
        f"From the baseline run: {len(misses)} retrieval misses (correct page not in top 5), "
        f"{len(wrong)} wrong answers despite correct retrieval, {len(halluc)} of "
        f"{sum(1 for x in rows if x['category'] == 'unanswerable')} unanswerable questions answered "
        "instead of abstaining.",
        "",
    ]
    for x in misses[:3]:
        exp = ", ".join(f"{e['file']} p.{e['page']}" for e in x["expected"])
        got = ", ".join(f"{e['file']} p.{e['page']}" for e in x["retrieved_top"][:3])
        lines.append(
            f"- **Retrieval miss — `{x['id']}`** ({x['lang']}): *{x['question']}* — expected {exp}; "
            f"top-3 retrieved: {got}."
        )
    for x in wrong[:2]:
        lines.append(
            f"- **Wrong answer with the right page — `{x['id']}`** ({x['lang']}): *{x['question']}* → "
            f"answered “{x['answer'][:160]}…”; reference: “{x['reference_answer'][:120]}”."
        )
    for x in halluc[:2]:
        lines.append(
            f"- **Did not abstain — `{x['id']}`** ({x['lang']}): *{x['question']}* → "
            f"“{x['answer'][:160]}…” (nothing in the corpus answers this)."
        )
    return "\n".join(lines) + "\n"


def replace(readme: str, marker: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {marker}:BEGIN -->\n).*?(<!-- {marker}:END -->)", re.DOTALL)
    assert pattern.search(readme), marker
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), readme)


def main(paths: list[str]) -> int:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    baseline = reports[0]
    readme = README.read_text(encoding="utf-8")
    readme = replace(readme, "RESULTS", results_section(baseline))
    readme = replace(readme, "BACKENDS", backends_section(baseline))
    readme = replace(readme, "FAILURES", failures_section(baseline))
    exp = table(sorted(reports, key=lambda r: r["created_at"]))
    readme = replace(readme, "EXPERIMENTS", exp)
    README.write_text(readme, encoding="utf-8")
    print(f"README updated from {len(reports)} report(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
