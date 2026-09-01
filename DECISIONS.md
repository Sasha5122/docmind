# Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-08-31 | Separate git repo per portfolio project (`docmind`), not a monorepo | A recruiter should land on a README that is only about this project |
| 2026-08-31 | `uv` for dependency + Python version management (`.python-version` = 3.12) | 10-100x faster than pip, one lockfile, and it downloads the exact Python the brief asks for even though the laptop has 3.13 |
| 2026-08-31 | `ruff` for lint + format (rules E, W, F, I, UP, B) | One fast tool replaces flake8 + isort + black; same config runs locally and in CI |
| 2026-08-31 | `hatchling` build backend, `pyproject.toml` only | Modern standard (PEP 621), no setup.py; enough for a package that is deployed, not published |
| 2026-08-31 | `pgvector/pgvector:pg16` Docker image for the database | Official Postgres 16 with the extension already compiled; one `docker compose up` and vector search works |
| 2026-08-31 | All configuration through one `pydantic-settings` `Settings` class (`docmind/config.py`) | Typed, validated (`LLM_BACKEND=openai` fails fast), documented in one place; no scattered `os.environ` reads |
| 2026-08-31 | `postgresql+psycopg://` (psycopg 3) driver | Modern, async-capable, maintained; psycopg2 is legacy |
| 2026-08-31 | SQLAlchemy 2 + Alembic for tables and migrations; `pgvector` Python package for the `Vector` column type | Tables as typed Python classes; migrations replay the same schema steps on laptop, CI and cloud so they never drift |
| 2026-08-31 | `pymupdf` for PDF text extraction; `langdetect` for per-document language | pymupdf is fast, keeps reading order and page numbers, no system deps; langdetect is tiny and good enough for de/fr/en/it at document level |
| 2026-08-31 | Microsoft Presidio for PII; start with pattern recognizers only (IBAN, email, phone), add spaCy-based name detection as a separate step | Pattern rules need no model download and are deterministic; name detection needs ~500 MB models per language and deserves its own evaluation |
| 2026-08-31 | spaCy `xx_ent_wiki_sm` (multilingual, 10 MB) for name detection in all four languages | One small model instead of three ~500 MB per-language models; enough to demonstrate the mechanism, swappable via one config dict; accuracy is a documented experiment for later |
| 2026-08-31 | `tiktoken` (cl100k_base) to count tokens in the chunker | Fast, tiny, no model download; bge-m3 uses a different tokenizer but sizes are within ~15 %, and 300-800 is far below its 8192 limit — verify as an experiment in M3 |
| 2026-09-01 | One `Embedder` protocol with `LocalEmbedder` (bge-m3), `AzureEmbedder` (text-embedding-3-large at 1024 dims) and `FakeEmbedder` (hash-based) | Same 1024-wide column serves both real backends, so switching is a config flag, not a migration; the fake keeps tests offline and fast |
| 2026-09-01 | Every embedding is normalised to unit length before storage | Cosine similarity then equals a dot product; pgvector `<=>` works identically for every backend and a future HNSW index can use `vector_cosine_ops` |
| 2026-09-01 | Model loading is lazy (first `embed()` call), never at import | The API, tests and CLI help can start in milliseconds; only the ingest run pays the ~2 GB download / ~10 s load |
