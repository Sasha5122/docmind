"""CLI: `python -m docmind.eval [--limit N] [--mode hybrid|vector|keyword] [--no-rerank] ...`"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from docmind.config import get_settings
from docmind.db import get_session
from docmind.eval.golden import DEFAULT_GOLDEN, load_golden
from docmind.eval.runner import NullLLM, markdown_summary, run_eval, write_report
from docmind.ingest.embedder import get_embedder
from docmind.llm.backends import get_llm
from docmind.rag import RagConfig
from docmind.retrieval.reranker import get_reranker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docmind.eval")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "keyword"])
    parser.add_argument("--k", type=int, default=5, help="chunks given to the LLM")
    parser.add_argument("--candidates", type=int, default=20, help="candidates per retriever")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="no LLM at all: measures recall@k / MRR only (fast retrieval ablations)",
    )
    parser.add_argument("--label", default=None, help="name for the report files")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for noisy in ("httpx", "presidio-analyzer", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = get_settings()
    items = load_golden(args.golden, args.limit)
    llm = NullLLM() if args.retrieval_only else get_llm(settings)
    judge = None if (args.no_judge or args.retrieval_only) else llm  # same backend judges
    config = RagConfig(k=args.k, candidates=args.candidates, retrieval_mode=args.mode)
    label = args.label or (
        f"{llm.name}-{args.mode}-{'norerank' if args.no_rerank else 'rerank'}"
        f"-k{args.k}{'-smoke' if args.limit else ''}"
    )

    with get_session() as session:
        report = run_eval(
            session,
            items,
            embedder=get_embedder(settings),
            reranker=get_reranker(enabled=not args.no_rerank, model_name=settings.reranker_model),
            llm=llm,
            config=config,
            judge=judge,
            label=label,
        )
    json_path, md_path = write_report(report, args.out)
    print(markdown_summary(report))
    print(f"written: {json_path}\n         {md_path}")
    return 0 if report.summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
