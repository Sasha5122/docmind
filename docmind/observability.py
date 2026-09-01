"""Langfuse tracing behind a no-op switch.

A trace = one /ask call; inside it three spans (retrieval, rerank, generation) with their
inputs, outputs, timings, token counts and cost. When LANGFUSE_PUBLIC_KEY is empty every
call here is a cheap no-op, so the code path is identical with and without observability.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from docmind.config import Settings, get_settings
from docmind.rag import Answer

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, client) -> None:  # client: langfuse.Langfuse | None
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def record(self, result: Answer, username: str = "anonymous", audit_id: int | None = None):
        """Send one finished answer as a trace. Never raises."""
        if self._client is None:
            return None
        try:
            trace = self._client.trace(
                name="ask",
                user_id=username,
                input={"question": result.question, "lang": result.lang},
                output={"answer": result.answer, "citations": [c.label for c in result.citations]},
                metadata={
                    "backend": result.backend,
                    "model": result.model,
                    "audit_id": audit_id,
                    **result.meta,
                },
                tags=[result.backend, result.lang, result.meta.get("retrieval_mode", "hybrid")],
            )
            t = result.timings
            trace.span(
                name="retrieval",
                input={"question": result.question},
                output={"chunk_ids": result.retrieved_ids},
                metadata={"seconds": round(t.retrieval_s, 3)},
            )
            trace.span(
                name="rerank",
                output={"chunk_ids": [c.chunk_id for c in result.contexts]},
                metadata={"seconds": round(t.rerank_s, 3)},
            )
            trace.generation(
                name="answer",
                model=result.model,
                input=[{"role": "user", "content": result.question}],
                output=result.answer,
                usage={
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "unit": "TOKENS",
                    "total_cost": result.cost_usd,
                },
                metadata={"seconds": round(t.llm_s, 3)},
            )
            return trace.id
        except Exception:  # noqa: BLE001 - observability must never break the request
            log.exception("langfuse trace failed")
            return None

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                log.exception("langfuse flush failed")


def build_tracer(s: Settings) -> Tracer:
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        return Tracer(None)
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
        )
        log.info("Langfuse tracing enabled (%s)", s.langfuse_host)
        return Tracer(client)
    except Exception:  # noqa: BLE001
        log.exception("Langfuse init failed; tracing disabled")
        return Tracer(None)


@lru_cache
def get_tracer() -> Tracer:
    """Process-wide tracer built from the environment settings."""
    return build_tracer(get_settings())
