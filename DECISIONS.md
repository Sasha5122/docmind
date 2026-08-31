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
