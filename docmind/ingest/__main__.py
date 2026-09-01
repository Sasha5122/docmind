"""Command line entry point: `python -m docmind.ingest data/raw/`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from docmind.db import get_session
from docmind.ingest.embedder import get_embedder
from docmind.ingest.pipeline import IngestReport, ingest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docmind.ingest",
        description="Parse, redact, chunk, embed and store PDFs.",
    )
    parser.add_argument("target", type=Path, help="a PDF file or a directory of PDFs")
    parser.add_argument("--max-tokens", type=int, default=500, help="chunk size (default 500)")
    parser.add_argument("--overlap", type=int, default=50, help="overlap tokens (default 50)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

    if not args.target.exists():
        print(f"error: {args.target} does not exist", file=sys.stderr)
        return 2

    embedder = get_embedder()
    with get_session() as session:
        reports = ingest_path(args.target, session, embedder, args.max_tokens, args.overlap)
    print(format_summary(reports))
    return 1 if any(r.status == "failed" for r in reports) else 0


def format_summary(reports: list[IngestReport]) -> str:
    lines = [f"{'file':50} {'status':9} {'chunks':>6} {'pii':>4} {'sec':>6}"]
    for r in reports:
        lines.append(
            f"{r.filename[:50]:50} {r.status:9} {r.chunks:>6} {r.pii_total:>4} {r.seconds:>6.1f}"
        )
    ingested = [r for r in reports if r.status == "ingested"]
    lines.append(
        f"\n{len(ingested)} ingested, "
        f"{sum(r.status == 'skipped' for r in reports)} skipped, "
        f"{sum(r.status == 'failed' for r in reports)} failed; "
        f"{sum(r.chunks for r in ingested)} chunks, "
        f"{sum(r.pii_total for r in ingested)} PII spans redacted"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
