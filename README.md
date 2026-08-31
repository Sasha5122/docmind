# 01-docmind

Portfolio project — see [CLAUDE.md](CLAUDE.md) for the full brief, milestones and working rules.

## Status
- [ ] M1
- [ ] M2
- [ ] M3
- [ ] M4
- [ ] M5

## How to run (local)
```bash
cp .env.example .env            # settings + secrets (never committed)
docker compose up -d            # Postgres 16 + pgvector in a container
uv sync                         # install Python 3.12 + all dependencies
uv run alembic upgrade head     # create the tables
uv run pytest                   # unit tests + migration tests against the DB
```
