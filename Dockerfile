# DocMind API image. Models (bge-m3, reranker, spaCy) are NOT baked in: the two
# sentence-transformers models are ~4.5 GB and are downloaded on first start into the
# HF_HOME volume; spaCy models are small wheels and come with `uv sync`.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR /app

# Dependencies first (cached layer); Linux resolves to the CPU torch wheel.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY docmind ./docmind
COPY migrations ./migrations
COPY alembic.ini ./
COPY static ./static
RUN uv sync --frozen --no-dev

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Migrate, then serve. One worker: the embedder/reranker live in process memory.
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn docmind.api.app:app --host 0.0.0.0 --port 8000 --workers 1"]
