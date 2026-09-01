# DocMind — Underwriting & Compliance Document Assistant

[![ci](https://github.com/Sasha5122/docmind/actions/workflows/ci.yml/badge.svg)](https://github.com/Sasha5122/docmind/actions/workflows/ci.yml)

**Multilingual RAG over regulated documents, with citations, PII redaction, an audit trail, a
cloud ↔ local LLM switch — and an evaluation harness that produces real numbers.**

An underwriter or compliance officer asks a question in German, French, English or Italian
about insurance general conditions (AVB/CGA), FINMA circulars or annual reports. DocMind
retrieves the relevant passages from Postgres + pgvector (hybrid vector + full-text search,
then a cross-encoder reranker), lets a language model answer **only from those passages**,
and returns the answer with a citation `[document, page]` after every factual sentence.
Personal data is removed before anything is indexed; every question is written to an audit
table; the answering model can be Azure OpenAI, Anthropic or a local Ollama model with one
config flag (data-residency option). A 96-question golden set measures retrieval recall,
citation accuracy, faithfulness and correctness, and CI fails on regressions.

> Plain-language explanation of every module and decision: [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).
> One line per architectural decision with the reason: [DECISIONS.md](DECISIONS.md).

---

## Architecture

```mermaid
flowchart LR
  subgraph Ingest["python -m docmind.ingest data/raw/"]
    A[PDF] --> B[pymupdf<br/>pages + page numbers]
    B --> C[Presidio + spaCy<br/>PII → &lt;PERSON&gt; &lt;IBAN&gt; …]
    C --> D[Chunker<br/>≤500 tokens, 50 overlap,<br/>keeps page]
    D --> E[bge-m3 embeddings<br/>1024-d, fp16 on GPU]
    E --> F[(Postgres 16<br/>pgvector HNSW + tsvector GIN)]
  end
  subgraph Ask["POST /ask"]
    Q[question] --> V[vector search<br/>cosine top-20]
    Q --> K[keyword search<br/>OR-ed stemmed terms top-20]
    V & K --> R[Reciprocal Rank Fusion]
    R --> X[bge-reranker-v2-m3<br/>top-5]
    X --> P[prompt with numbered sources]
    P --> L{{LLM_BACKEND<br/>azure | anthropic | ollama}}
    L --> ANS[answer + citations<br/>+ tokens, cost, latency]
    ANS --> AUD[(audit_log)]
    ANS --> LF[Langfuse trace]
  end
  F -.-> V
  F -.-> K
  subgraph Eval["python -m docmind.eval"]
    G[golden.jsonl<br/>96 Q, DE/FR/EN] --> M[recall@k, MRR,<br/>citation precision/coverage,<br/>LLM-judge faithfulness + correctness]
    M --> REP[reports/*.json + .md] --> GATE[CI gate:<br/>fail on >5-point drop]
  end
```

---

## Run it locally (3 commands + a model)

```bash
cp .env.example .env                      # settings; keep LLM_BACKEND=ollama for a fully local run
docker compose up -d                      # Postgres 16 + pgvector
uv sync && uv run alembic upgrade head    # Python 3.12 env (GPU torch on Windows, CPU elsewhere) + tables
```

Then either install [Ollama](https://ollama.com) and `ollama pull qwen2.5:7b`, or put Azure
OpenAI / Anthropic keys in `.env` and set `LLM_BACKEND` accordingly.

```bash
uv run python scripts/download_corpus.py  # 18 public PDFs → data/raw/ (FINMA, AVB, annual reports)
uv run python -m docmind.ingest data/raw/ # parse → redact → chunk → embed → store  (~10 min on a laptop GPU)
uv run uvicorn docmind.api.app:app --reload
```

Open <http://localhost:8000> for the one-page UI, <http://localhost:8000/docs> for the API.

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Wie hoch ist der Selbstbehalt bei Elementarschäden in der Zurich Haushaltversicherung?"}' | jq
```

```bash
uv run pytest                              # 85 tests; DB tests skip automatically if Docker is off
uv run python -m docmind.eval --limit 20   # smoke evaluation → reports/
bash scripts/run_experiments.sh            # the full experiment matrix
docker compose --profile observability up -d   # optional: Langfuse UI on :3000
docker compose --profile api up --build        # optional: the API itself in a container
```

### Configuration (`.env`)

| variable | values | purpose |
|---|---|---|
| `LLM_BACKEND` | `azure` \| `anthropic` \| `ollama` | who writes the answer — the data-residency switch |
| `AZURE_OPENAI_*`, `ANTHROPIC_*`, `OLLAMA_*` | keys, endpoint, deployment/model | per-backend settings |
| `EMBEDDING_BACKEND` | `local` (bge-m3) \| `azure` (text-embedding-3-large @1024) | who turns text into vectors |
| `RERANKER_ENABLED` | `true` \| `false` | second-stage cross-encoder on/off |
| `BASIC_AUTH_USER/PASSWORD` | strings | HTTP Basic auth for `/ask`; empty = open (dev only) |
| `LANGFUSE_*` | keys + host | tracing; empty = disabled |
| `DATABASE_URL` | SQLAlchemy URL | Postgres with pgvector |

---

## Evaluation

### Golden set — `data/eval/golden.jsonl`
96 questions written against the actual corpus pages, each with the evidence sentence and
the PDF page index (machine-checked to be on that page):

| | DE | FR | EN | total |
|---|---|---|---|---|
| fact (one passage answers it) | | | | 70 |
| cross-lingual (question in one language, evidence in another) | | | | 12 |
| multi-doc (two documents needed, e.g. AXA vs Mobiliar deductible) | | | | 6 |
| unanswerable (correct behaviour: abstain) | | | | 8 |
| **total** | **45** | **19** | **32** | **96** |

### Metrics
- **recall@5 / @10** — was a correct page among the first k retrieved chunks? (±1 page: a chunk is cited by the page it starts on)
- **MRR** — 1 / rank of the first correct chunk
- **citation precision** — share of the answer's citations that point at a correct page
- **citation coverage** — share of factual sentences that carry a citation
- **faithfulness** — share of answer statements supported by the retrieved passages (RAGAS definition, LLM-as-judge)
- **answer correctness** — 0 / 0.5 / 1 agreement with the reference answer (LLM-as-judge)
- **abstention rate** — how often the system says "not in the documents"

### Results

<!-- RESULTS:BEGIN -->
*Pending: the tables below are filled from `reports/` by the first full run (see commit history).*
<!-- RESULTS:END -->

### Experiments

<!-- EXPERIMENTS:BEGIN -->
*Pending.*
<!-- EXPERIMENTS:END -->

### Azure vs local (Ollama)

The code path is identical (`LLM_BACKEND`); the same golden set runs against either. This
repository was developed without cloud credentials, so the measured column is the local one.
To fill the Azure column: set `AZURE_OPENAI_API_KEY/ENDPOINT` in `.env`, `LLM_BACKEND=azure`,
and run `uv run python -m docmind.eval --label azure-baseline` (≈ USD 0.05 for 96 questions
with gpt-4o-mini; cost is computed per call from token counts in `docmind/llm/base.py`).

<!-- BACKENDS:BEGIN -->
| | Ollama qwen2.5:7b (RTX 3050 6 GB) | Azure gpt-4o-mini |
|---|---|---|
| faithfulness / correctness | pending | not measured (no key) |
| latency p50 / p95 | pending | — |
| cost per 1,000 questions | $0 (electricity) | ≈ $0.5 at 1.3k input + 150 output tokens |
| data leaves the machine | no | yes (Azure region of choice) |
<!-- BACKENDS:END -->

---

## Failure modes (honest list)

<!-- FAILURES:BEGIN -->
*Filled from the eval "retrieval misses" and judge details after the first full run.*
<!-- FAILURES:END -->

Known limitations independent of the numbers:
- **IBAN across a PDF line break** is not detected by the pattern rule.
- **Single-word names** are deliberately not redacted (the name filter requires 2–4 Title-Case tokens); on this corpus the multilingual model without the filter flagged 63 % of chunks — precision was the right trade for regulatory text, but customer correspondence would need a different setting.
- **Tables in annual reports** extract as scrambled columns; questions whose evidence sits only in a dense table are the hardest category.
- **French over-redaction**: "Jean Dupont de Lausanne" is redacted as one name (particle handling).

---

## Security & compliance notes

- **PII never reaches the index**: redaction runs per page before chunking, embedding and storage; placeholders are stored, originals are not.
- **Audit trail**: `audit_log` has one row per `/ask` — user, question, language, retrieved and cited chunk ids, model, tokens, cost, latency, status. Reproduce any answer later.
- **Data residency**: `LLM_BACKEND=ollama` + `EMBEDDING_BACKEND=local` keeps every byte on the machine; the Azure path pins a region.
- **Auth**: HTTP Basic on the API (swap the `current_user` dependency for OIDC/SSO in a real deployment); username is part of the audit row.
- **What a real FINMA-grade deployment still needs**: retention and deletion policy for `audit_log`, role-based document access (the `document_ids` filter is the hook), encryption at rest and key management, a human-review loop for low-confidence answers, and an approval process for the golden set.

---

## Deploy

`deploy/azure/deploy.sh` creates a resource group, builds the image in Azure Container
Registry, provisions Postgres Flexible Server with `vector`, and runs the API as a Container
App in **Switzerland North** with secrets for the database URL and API keys
(`az login` required; ≈ USD 15/month idle with scale-to-zero, ≈ 80/month always on).
The same image runs locally with `docker compose --profile api up --build`.

---

## Repository layout

```
docmind/
  ingest/      parser.py · chunker.py · embedder.py · pipeline.py · __main__.py (CLI)
  pii/         redactor.py (Presidio + per-language spaCy + name filter)
  retrieval/   search.py (vector + keyword) · hybrid.py (RRF) · reranker.py
  llm/         base.py (contract, prices) · backends.py (azure/anthropic/ollama/fake) · prompt.py
  rag.py       question → answer with citations, timings, cost
  api/         app.py (FastAPI) · audit.py
  eval/        golden.py · metrics.py · runner.py · __main__.py · gate.py · compare.py
  observability.py  Langfuse tracer
migrations/    Alembic: tables → tsv + HNSW/GIN → audit_log
data/eval/golden.jsonl · data/raw/ (git-ignored PDFs) · reports/ · static/index.html
scripts/       download_corpus.py · run_experiments.sh
deploy/azure/deploy.sh · Dockerfile · docker-compose.yml · .github/workflows/
tests/         one file per module (unit + DB-backed; model-download tests opt-in via DOCMIND_RUN_SLOW=1)
```

## Status

- [x] M1 Ingestion & storage — 18 documents, page-level citations, PII redaction, bge-m3 in pgvector
- [x] M2 Retrieval & API — hybrid + reranker, citations, `/ask` `/health` `/metrics`, web page
- [x] M3 Evaluation — 96-question golden set, metrics, runner, experiments (numbers below once run)
- [x] M4 Backend switch, audit, observability — Ollama end to end, `audit_log`, Langfuse tracer
- [x] M5 CI/CD + deploy — GitHub Actions with eval gate, Dockerfile, Azure deploy script (deployment itself needs an Azure account)
