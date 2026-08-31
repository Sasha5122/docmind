# Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-08-31 | Separate git repo per portfolio project (`docmind`), not a monorepo | A recruiter should land on a README that is only about this project |
| 2026-08-31 | `uv` for dependency + Python version management (`.python-version` = 3.12) | 10-100x faster than pip, one lockfile, and it downloads the exact Python the brief asks for even though the laptop has 3.13 |
| 2026-08-31 | `ruff` for lint + format (rules E, W, F, I, UP, B) | One fast tool replaces flake8 + isort + black; same config runs locally and in CI |
| 2026-08-31 | `hatchling` build backend, `pyproject.toml` only | Modern standard (PEP 621), no setup.py; enough for a package that is deployed, not published |
