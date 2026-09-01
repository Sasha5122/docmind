"""Write one audit row per /ask call. Failures to audit are logged, never raised."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from docmind.models import AuditLog
from docmind.rag import Answer

log = logging.getLogger(__name__)


def record_answer(session: Session, username: str, result: Answer) -> int | None:
    row = AuditLog(
        username=username,
        question=result.question,
        lang=result.lang,
        answer=result.answer,
        backend=result.backend,
        model=result.model,
        retrieval_mode=str(result.meta.get("retrieval_mode", "")),
        retrieved_chunk_ids=list(result.retrieved_ids),
        cited_chunk_ids=[c.chunk_id for c in result.citations],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_s=result.timings.total_s,
        retrieval_s=result.timings.retrieval_s,
        rerank_s=result.timings.rerank_s,
        llm_s=result.timings.llm_s,
        status="ok",
    )
    return _save(session, row)


def record_error(session: Session, username: str, question: str, error: str) -> int | None:
    row = AuditLog(username=username, question=question, status="error", error=error[:2000])
    return _save(session, row)


def _save(session: Session, row: AuditLog) -> int | None:
    try:
        session.rollback()  # a failed query may have left the session in an aborted state
        session.add(row)
        session.commit()
        return row.id
    except Exception:  # noqa: BLE001
        log.exception("audit write failed")
        session.rollback()
        return None
