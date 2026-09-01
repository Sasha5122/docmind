"""Run the golden set through the pipeline and score it.

    python -m docmind.eval                       # full run, default config
    python -m docmind.eval --limit 20            # CI smoke set
    python -m docmind.eval --mode vector         # experiment: vector-only retrieval
    python -m docmind.eval --no-rerank           # experiment: reranker off
    python -m docmind.eval --no-judge            # deterministic metrics only (no LLM judge)

Writes reports/eval_<timestamp>.json (every question, every number) and a short
markdown summary next to it, and prints the summary.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from docmind.eval.golden import GoldenItem
from docmind.eval.metrics import (
    citation_coverage,
    citation_precision,
    judge_correctness,
    judge_faithfulness,
    mean,
    mrr,
    recall_at_k,
)
from docmind.ingest.embedder import Embedder
from docmind.llm.base import LLM
from docmind.rag import RagConfig, answer_question
from docmind.retrieval.reranker import Reranker

log = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    id: str
    lang: str
    category: str
    question: str
    answer: str
    reference_answer: str
    expected: list[dict]
    citations: list[dict]
    retrieved_top: list[dict]  # first 10 (file, page) the retriever returned
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None
    citation_precision: float | None
    citation_coverage: float
    abstained: bool  # answered "not in the documents"
    # Scored on the k chunks actually handed to the LLM (AFTER reranking), unlike
    # recall@k / MRR which score the fused candidate list BEFORE reranking.
    context_hit: float | None = None
    context_mrr: float | None = None
    faithfulness: float | None = None
    correctness: float | None = None
    judge_detail: dict = field(default_factory=dict)
    latency_s: float = 0.0
    retrieval_s: float = 0.0
    rerank_s: float = 0.0
    llm_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class EvalReport:
    created_at: str
    label: str
    config: dict
    n: int
    summary: dict
    by_lang: dict
    by_category: dict
    results: list[QuestionResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


ABSTAIN_MARKERS = ("keine angaben", "aucune information", "could not find", "non trovo")


class EvalAborted(RuntimeError):
    """Raised when several questions in a row fail: the backend is down, not the questions."""


class NullLLM:
    """Retrieval-only evaluation: answers nothing, so only recall@k / MRR are meaningful."""

    name = "none"
    model = "retrieval-only"

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0):
        from docmind.llm.base import LLMResponse

        return LLMResponse("", self.model, 0, 0, 0.0, 0.0)


def _is_abstention(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in ABSTAIN_MARKERS)


def evaluate_item(
    session: Session,
    item: GoldenItem,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLM,
    config: RagConfig,
    judge: LLM | None,
    document_ids: list[int] | None = None,
) -> QuestionResult:
    try:
        result = answer_question(
            session,
            item.question,
            embedder,
            reranker,
            llm,
            config=config,
            lang=item.lang,
            document_ids=document_ids,
        )
    except Exception as exc:  # noqa: BLE001 - one broken question must not kill the run
        log.exception("question %s failed", item.id)
        return QuestionResult(
            item.id, item.lang, item.category, item.question, "", item.reference_answer,
            [asdict(s) for s in item.sources], [], [], None, None, None, None, 0.0, False,
            error=f"{exc.__class__.__name__}: {exc}",
        )  # fmt: skip

    # `retrieved_ids` are candidate ids in fused order; map them back to chunks we know.
    by_id = {c.chunk_id: c for c in result.contexts}
    from docmind.models import Chunk, Document  # local import keeps module import light

    missing = [cid for cid in result.retrieved_ids if cid not in by_id]
    if missing:
        rows = (
            session.query(Chunk.id, Document.filename, Chunk.page)
            .join(Chunk.document)
            .filter(Chunk.id.in_(missing))
            .all()
        )
        from docmind.retrieval.search import RetrievedChunk

        for cid, filename, page in rows:
            by_id[cid] = RetrievedChunk(cid, 0, filename, None, page, None, "", 0.0)
    retrieved = [by_id[cid] for cid in result.retrieved_ids if cid in by_id]

    answerable = item.answerable
    abstained = _is_abstention(result.answer)
    qr = QuestionResult(
        id=item.id,
        lang=item.lang,
        category=item.category,
        question=item.question,
        answer=result.answer,
        reference_answer=item.reference_answer,
        expected=[asdict(s) for s in item.sources],
        citations=[{"file": c.filename, "page": c.page, "n": c.n} for c in result.citations],
        retrieved_top=[{"file": c.filename, "page": c.page} for c in retrieved[:10]],
        recall_at_5=recall_at_k(retrieved, item.sources, 5) if answerable else None,
        recall_at_10=recall_at_k(retrieved, item.sources, 10) if answerable else None,
        mrr=mrr(retrieved, item.sources) if answerable else None,
        context_hit=recall_at_k(result.contexts, item.sources, config.k) if answerable else None,
        context_mrr=mrr(result.contexts, item.sources) if answerable else None,
        citation_precision=citation_precision(result.citations, item.sources)
        if answerable
        else None,
        citation_coverage=citation_coverage(result.answer) if not abstained else 1.0,
        abstained=abstained,
        latency_s=result.timings.total_s,
        retrieval_s=result.timings.retrieval_s,
        rerank_s=result.timings.rerank_s,
        llm_s=result.timings.llm_s,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )
    if not answerable:
        # For unanswerable questions the "correct" behaviour is to abstain.
        qr.correctness = 1.0 if abstained else 0.0
        qr.faithfulness = 1.0 if abstained else None

    if judge is not None and answerable and not abstained:
        try:
            faith = judge_faithfulness(judge, result.answer, [c.text for c in result.contexts])
            corr = judge_correctness(judge, item.question, result.answer, item.reference_answer)
            qr.faithfulness, qr.correctness = faith.score, corr.score
            qr.judge_detail = {"faithfulness": faith.detail, "correctness": corr.detail}
        except Exception as exc:  # noqa: BLE001
            qr.judge_detail = {"error": f"{exc.__class__.__name__}: {exc}"}
    elif judge is not None and answerable and abstained:
        qr.correctness = 0.0  # the reference says the answer exists; abstaining is wrong
        qr.faithfulness = 1.0  # but nothing was invented

    return qr


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return round(ordered[index], 3)


def summarise(results: list[QuestionResult]) -> dict:
    ok = [r for r in results if r.error is None]
    lat = [r.latency_s for r in ok]
    return {
        "questions": len(results),
        "errors": len(results) - len(ok),
        "recall_at_5": mean([r.recall_at_5 for r in ok]),
        "recall_at_10": mean([r.recall_at_10 for r in ok]),
        "mrr": mean([r.mrr for r in ok]),
        "context_hit": mean([r.context_hit for r in ok]),
        "context_mrr": mean([r.context_mrr for r in ok]),
        "citation_precision": mean([r.citation_precision for r in ok]),
        "citation_coverage": mean([r.citation_coverage for r in ok]),
        "faithfulness": mean([r.faithfulness for r in ok]),
        "correctness": mean([r.correctness for r in ok]),
        "abstention_rate": mean([float(r.abstained) for r in ok]),
        "latency_p50_s": _percentile(lat, 50),
        "latency_p95_s": _percentile(lat, 95),
        "latency_mean_s": round(statistics.fmean(lat), 3) if lat else None,
        "retrieval_mean_s": round(statistics.fmean([r.retrieval_s for r in ok]), 3) if ok else None,
        "rerank_mean_s": round(statistics.fmean([r.rerank_s for r in ok]), 3) if ok else None,
        "llm_mean_s": round(statistics.fmean([r.llm_s for r in ok]), 3) if ok else None,
        "cost_usd_total": round(sum(r.cost_usd for r in ok), 6),
        "cost_usd_per_1000": round(sum(r.cost_usd for r in ok) / len(ok) * 1000, 4) if ok else None,
        "tokens_in_mean": round(statistics.fmean([r.input_tokens for r in ok])) if ok else None,
        "tokens_out_mean": round(statistics.fmean([r.output_tokens for r in ok])) if ok else None,
    }


def _grouped(results: list[QuestionResult], key: str) -> dict:
    groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        groups.setdefault(getattr(r, key), []).append(r)
    keep = (
        "questions",
        "recall_at_5",
        "context_hit",
        "citation_precision",
        "faithfulness",
        "correctness",
    )
    return {
        name: {k: v for k, v in summarise(group).items() if k in keep}
        for name, group in sorted(groups.items())
    }


def run_eval(
    session: Session,
    items: list[GoldenItem],
    embedder: Embedder,
    reranker: Reranker,
    llm: LLM,
    config: RagConfig,
    judge: LLM | None,
    label: str,
    extra_config: dict | None = None,
    document_ids: list[int] | None = None,
    max_consecutive_errors: int = 5,
) -> EvalReport:
    started = time.perf_counter()
    results: list[QuestionResult] = []
    streak = 0  # errors in a row; a long streak means Ollama/DB died, so stop early
    for i, item in enumerate(items, 1):
        qr = evaluate_item(session, item, embedder, reranker, llm, config, judge, document_ids)
        results.append(qr)
        streak = streak + 1 if qr.error else 0
        if max_consecutive_errors and streak >= max_consecutive_errors:
            raise EvalAborted(
                f"{streak} questions in a row failed (last: {qr.error}); "
                "aborting instead of writing a junk report. Is the LLM backend / database up?"
            )
        log.info(
            "%3d/%d %-22s r@5=%s cite=%s faith=%s corr=%s %.1fs",
            i,
            len(items),
            item.id,
            _fmt(qr.recall_at_5),
            _fmt(qr.citation_precision),
            _fmt(qr.faithfulness),
            _fmt(qr.correctness),
            qr.latency_s,
        )
    report = EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        label=label,
        config={
            **asdict(config),
            "llm_backend": llm.name,
            "llm_model": llm.model,
            "reranker": reranker.__class__.__name__,
            "judge": f"{judge.name}:{judge.model}" if judge else None,
            "embedder": getattr(embedder, "model_name", embedder.__class__.__name__),
            "machine": platform.node(),
            "wall_time_s": round(time.perf_counter() - started, 1),
            **(extra_config or {}),
        },
        n=len(results),
        summary=summarise(results),
        by_lang=_grouped(results, "lang"),
        by_category=_grouped(results, "category"),
        results=results,
    )
    return report


def _fmt(value: float | None) -> str:
    return "  - " if value is None else f"{value:.2f}"


def write_report(report: EvalReport, out_dir: Path = Path("reports")) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = "".join(ch if ch.isalnum() else "-" for ch in report.label).strip("-") or "eval"
    json_path = out_dir / f"eval_{stamp}_{slug}.json"
    md_path = out_dir / f"eval_{stamp}_{slug}.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(markdown_summary(report), encoding="utf-8")
    # Small summary-only file for the CI regression gate (committed; the full report is not).
    latest = {"created_at": report.created_at, "label": report.label, "n": report.n,
              "config": report.config, "summary": report.summary}  # fmt: skip
    (out_dir / "latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return json_path, md_path


def markdown_summary(report: EvalReport) -> str:
    s = report.summary
    c = report.config

    def pct(v: float | None) -> str:
        return "–" if v is None else f"{100 * v:.0f} %"

    lines = [
        f"# Eval: {report.label}",
        "",
        f"{report.created_at} · {report.n} questions · LLM `{c['llm_backend']}:{c['llm_model']}` · "
        f"retrieval `{c['retrieval_mode']}` k={c['k']} · reranker `{c['reranker']}` · "
        f"judge `{c['judge']}`",
        "",
        "| metric | value |",
        "|---|---|",
        f"| recall@5 | {pct(s['recall_at_5'])} |",
        f"| recall@10 | {pct(s['recall_at_10'])} |",
        f"| MRR | {'–' if s['mrr'] is None else f'{s["mrr"]:.2f}'} |",
        f"| context hit@k (correct page among the k chunks given to the LLM, after reranking) | "
        f"{pct(s.get('context_hit'))} |",
        f"| context MRR | {'–' if s.get('context_mrr') is None else f'{s["context_mrr"]:.2f}'} |",
        f"| citation precision | {pct(s['citation_precision'])} |",
        f"| citation coverage | {pct(s['citation_coverage'])} |",
        f"| faithfulness (judge) | {pct(s['faithfulness'])} |",
        f"| answer correctness (judge) | {pct(s['correctness'])} |",
        f"| abstention rate | {pct(s['abstention_rate'])} |",
        f"| latency p50 / p95 | {s['latency_p50_s']} s / {s['latency_p95_s']} s |",
        f"| of which retrieval / rerank / LLM (mean) | {s['retrieval_mean_s']} / "
        f"{s['rerank_mean_s']} / {s['llm_mean_s']} s |",
        f"| cost per 1000 questions | ${s['cost_usd_per_1000']} |",
        f"| errors | {s['errors']} |",
        "",
        "## By language",
        "",
        "| lang | n | recall@5 | citation precision | faithfulness | correctness |",
        "|---|---|---|---|---|---|",
    ]
    for lang, g in report.by_lang.items():
        lines.append(
            f"| {lang} | {g['questions']} | {pct(g['recall_at_5'])} | "
            f"{pct(g['citation_precision'])} | {pct(g['faithfulness'])} | {pct(g['correctness'])} |"
        )
    lines += [
        "",
        "## By category",
        "",
        "| category | n | recall@5 | citation precision | faithfulness | correctness |",
        "|---|---|---|---|---|---|",
    ]
    for cat, g in report.by_category.items():
        lines.append(
            f"| {cat} | {g['questions']} | {pct(g['recall_at_5'])} | "
            f"{pct(g['citation_precision'])} | {pct(g['faithfulness'])} | {pct(g['correctness'])} |"
        )
    misses = [r for r in report.results if r.recall_at_5 == 0.0]
    if misses:
        lines += ["", "## Retrieval misses (recall@5 = 0)", ""]
        for r in misses[:15]:
            exp = ", ".join(f"{e['file']} p.{e['page']}" for e in r.expected)
            got = ", ".join(f"{e['file']} p.{e['page']}" for e in r.retrieved_top[:3])
            lines.append(f"- **{r.id}** ({r.lang}) expected {exp} — got {got}")
    return "\n".join(lines) + "\n"
