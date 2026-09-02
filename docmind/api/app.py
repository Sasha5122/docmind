"""FastAPI service: POST /ask, GET /health, GET /metrics.

Heavy objects (embedder, reranker, LLM client) are built once at startup and kept on
`app.state`; tests replace them with fakes through `build_app(...)`.
"""

from __future__ import annotations

import logging
import secrets
import statistics
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from docmind.api.audit import record_answer, record_error
from docmind.config import get_settings
from docmind.db import get_engine, get_session
from docmind.ingest.embedder import Embedder, get_embedder
from docmind.llm.backends import get_llm
from docmind.llm.base import LLM
from docmind.models import Chunk, Document
from docmind.observability import get_tracer
from docmind.rag import RagConfig, answer_question
from docmind.retrieval.reranker import Reranker, get_reranker

log = logging.getLogger(__name__)
_LATENCY_WINDOW = 500  # last N request latencies kept in memory for /metrics
_basic = HTTPBasic(auto_error=False)
_STATIC = Path(__file__).resolve().parent.parent.parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    lang: str | None = Field(default=None, pattern="^(de|fr|en|it)$")
    document_ids: list[int] | None = None
    k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: str = Field(default="hybrid", pattern="^(hybrid|vector|keyword)$")


class CitationOut(BaseModel):
    n: int
    filename: str
    page: int
    chunk_id: int
    text: str


class AskResponse(BaseModel):
    audit_id: int | None
    answer: str
    lang: str
    citations: list[CitationOut]
    backend: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    retrieval_s: float
    rerank_s: float
    llm_s: float
    retrieved_chunk_ids: list[int]


def build_app(
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLM | None = None,
) -> FastAPI:
    """Create the app. Pass fakes in tests; pass nothing in production."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        app.state.embedder = embedder or get_embedder(settings)
        app.state.reranker = reranker or get_reranker(enabled=settings.reranker_enabled)
        app.state.llm = llm or get_llm(settings)
        app.state.latencies = deque(maxlen=_LATENCY_WINDOW)
        app.state.requests = 0
        app.state.errors = 0
        app.state.started = time.time()
        app.state.tracer = get_tracer()
        log.info(
            "DocMind ready: llm=%s embedder=%s", settings.llm_backend, settings.embedding_backend
        )
        yield
        app.state.tracer.flush()

    app = FastAPI(title="DocMind", version="0.1.0", lifespan=lifespan)

    def db() -> Session:
        session = get_session()
        try:
            yield session
        finally:
            session.close()

    def current_user(credentials: HTTPBasicCredentials | None = Depends(_basic)) -> str:
        """HTTP Basic auth when BASIC_AUTH_USER/PASSWORD are set; 'anonymous' otherwise."""
        settings = get_settings()
        if not settings.basic_auth_user:
            return "anonymous"
        if credentials is None or not (
            secrets.compare_digest(credentials.username, settings.basic_auth_user)
            and secrets.compare_digest(credentials.password, settings.basic_auth_password)
        ):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict:
        try:
            with get_engine().connect() as conn:
                conn.execute(text("select 1"))
            database = "ok"
        except Exception as exc:  # noqa: BLE001
            database = f"error: {exc.__class__.__name__}"
        return {
            "status": "ok" if database == "ok" else "degraded",
            "database": database,
            "llm_backend": get_settings().llm_backend,
            "tracing": request.app.state.tracer.enabled,
        }

    @app.get("/metrics")
    def metrics(request: Request) -> dict:
        lat = sorted(request.app.state.latencies)
        return {
            "requests_total": request.app.state.requests,
            "errors_total": request.app.state.errors,
            "uptime_s": round(time.time() - request.app.state.started, 1),
            "latency_p50_s": _percentile(lat, 50),
            "latency_p95_s": _percentile(lat, 95),
            "latency_mean_s": round(statistics.fmean(lat), 3) if lat else None,
        }

    @app.get("/documents")
    def documents(session: Session = Depends(db)) -> list[dict]:
        """The indexed library: one row per PDF with its chunk count (feeds the web UI sidebar)."""
        rows = session.execute(
            select(
                Document.id,
                Document.filename,
                Document.lang,
                Document.page_count,
                func.count(Chunk.id).label("chunks"),
            )
            .join(Chunk, Chunk.document_id == Document.id, isouter=True)
            .group_by(Document.id)
            .order_by(Document.filename)
        ).all()
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "lang": r.lang,
                "page_count": r.page_count,
                "chunks": r.chunks,
            }
            for r in rows
        ]

    @app.post("/ask", response_model=AskResponse)
    def ask(
        body: AskRequest,
        request: Request,
        session: Session = Depends(db),
        username: str = Depends(current_user),
    ) -> AskResponse:
        state = request.app.state
        state.requests += 1
        try:
            result = answer_question(
                session,
                body.question,
                embedder=state.embedder,
                reranker=state.reranker,
                llm=state.llm,
                config=RagConfig(k=body.k, retrieval_mode=body.retrieval_mode),
                lang=body.lang,
                document_ids=body.document_ids,
            )
        except Exception as exc:
            state.errors += 1
            log.exception("ask failed")
            record_error(session, username, body.question, f"{exc.__class__.__name__}: {exc}")
            raise HTTPException(status_code=502, detail=f"{exc.__class__.__name__}: {exc}") from exc
        state.latencies.append(result.timings.total_s)
        audit_id = record_answer(session, username, result)
        state.tracer.record(result, username=username, audit_id=audit_id)
        return AskResponse(
            audit_id=audit_id,
            answer=result.answer,
            lang=result.lang,
            citations=[
                CitationOut(
                    n=c.n, filename=c.filename, page=c.page, chunk_id=c.chunk_id, text=c.text
                )
                for c in result.citations
            ],
            backend=result.backend,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_s=round(result.timings.total_s, 3),
            retrieval_s=round(result.timings.retrieval_s, 3),
            rerank_s=round(result.timings.rerank_s, 3),
            llm_s=round(result.timings.llm_s, 3),
            retrieved_chunk_ids=result.retrieved_ids,
        )

    return app


def _percentile(sorted_values: list[float], pct: int) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1)))
    return round(sorted_values[index], 3)


app = build_app()
