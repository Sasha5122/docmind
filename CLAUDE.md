# DocMind — Underwriting & Compliance Document Assistant (RAG with evaluation)

## Why this project exists
Portfolio project #1 of 3, targeting AI Engineer roles in Swiss/EU financial services (banks, insurers, consultancies serving them). It must demonstrate **production-grade RAG**: multilingual documents, citations, PII handling, audit trail, switchable LLM backend (cloud ↔ local), and above all an **evaluation harness with real numbers**. Reviewers will look for: deployed URL, eval metrics, cost/latency figures, and a README that explains failure modes honestly.

Reference job requirements this project must cover: Python, LLM APIs (Azure OpenAI + Anthropic), RAG, embeddings, vector DB (pgvector), FastAPI, Docker, CI/CD, cloud deploy, evaluation (RAGAS / LLM-as-judge), observability (Langfuse), prompt engineering, SQL, PII/guardrails, local LLMs (Ollama).

## The product (one paragraph)
A web service where an underwriter or compliance officer uploads or selects regulated documents — insurance general conditions (AVB/CGA), FINMA circulars, annual reports, policy wordings — in **German, French, English (Italian optional)** — and asks questions in any of those languages. Answers cite the exact document, page and passage; every query is logged to an audit table; personal data is redacted before indexing; the LLM backend can be switched between Azure OpenAI/Anthropic and a local Ollama model with one config flag (data-residency story).

## Data (all public, no licensing issues)
- FINMA circulars (PDF, DE/FR/EN): https://www.finma.ch/en/documentation/circulars/
- Swiss insurer general conditions (AVB) — pick 5–10 public PDFs from AXA, Zurich, Helvetia, Mobiliar, Baloise websites
- 2–3 annual reports (Julius Baer, Swiss Re, Swiss Life) — long, multilingual, tables
- Store raw files in `data/raw/`, never commit anything with personal data.

## Architecture
```
[Upload/ingest CLI] → [Parser: PDF→text+layout (pymupdf / unstructured)] → [PII redaction (presidio)]
   → [Chunker: layout-aware, 300–800 tokens, overlap, keeps page refs]
   → [Embeddings: multilingual (bge-m3 or text-embedding-3-large)] → [Postgres + pgvector]
                                                                      ↑ BM25 (Postgres full-text) for hybrid
[FastAPI /ask] → [Hybrid retrieval top-k] → [Reranker (bge-reranker or cohere)] → [LLM answer with citations]
   → [Audit log table] → [Langfuse trace] → response {answer, citations[], confidence, cost, latency}
[Eval harness] runs a golden Q&A set → RAGAS metrics + retrieval recall@k → JSON report; CI fails on regression.
```

## Tech stack (fixed — do not substitute without a reason written in DECISIONS.md)
- Python 3.12, FastAPI, Pydantic v2, SQLAlchemy, Alembic
- Postgres 16 + pgvector (Docker); full-text search for BM25 side
- Embeddings: `BAAI/bge-m3` via sentence-transformers (local) OR Azure `text-embedding-3-large` — both supported, config flag
- LLM backends behind one interface: Azure OpenAI (gpt-4o-mini default), Anthropic (claude-sonnet), Ollama (llama3.1 / qwen2.5) — `LLM_BACKEND=azure|anthropic|ollama`
- PII: Microsoft Presidio (DE/FR/EN recognizers)
- Reranker: `BAAI/bge-reranker-v2-m3`
- Eval: RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) + custom recall@k + LLM-as-judge for citation correctness
- Observability: Langfuse (self-hosted via docker-compose or cloud free tier)
- Tests: pytest; CI: GitHub Actions (lint, tests, eval on a 20-question smoke set)
- Deploy: Docker Compose locally; cloud target = Azure Container Apps (preferred, matches Swiss market) or AWS ECS; Terraform optional
- Frontend: minimal — a single HTML page or Streamlit is enough; API is the product

## Milestones (build in this order; each has acceptance criteria)
### M1 — Ingestion & storage (week 1)
- `python -m docmind.ingest data/raw/` parses PDFs, redacts PII, chunks with page refs, embeds, stores in pgvector
- Acceptance: 10 documents ingested; `SELECT count(*) FROM chunks` > 2,000; a chunk row contains doc_id, page, lang, text, embedding; PII test doc has names/IBANs replaced
### M2 — Retrieval & answer API (week 1–2)
- `POST /ask {question, lang?, filters?}` → hybrid retrieval → rerank → answer with citations `[doc, page]`
- Acceptance: answers in the question's language; every factual sentence carries a citation; p95 latency < 4 s with Azure, measured and recorded
### M3 — Evaluation harness (week 2) ← the most important milestone
- `data/eval/golden.jsonl`: 60–100 questions across DE/FR/EN with reference answers and the source pages (write ~30 by hand, generate the rest with an LLM and REVIEW them manually)
- `python -m docmind.eval` → RAGAS metrics + recall@5/@10 + citation accuracy → `reports/eval_<date>.json` + markdown summary
- Acceptance: baseline numbers recorded in README; at least one documented experiment (e.g. chunk size 300 vs 600, hybrid vs vector-only, reranker on/off) with before/after table
### M4 — Backend switch, audit, observability (week 3)
- `LLM_BACKEND=ollama` works end to end on a laptop; audit table logs user, question, retrieved chunk ids, answer, cost, latency; Langfuse shows traces
- Acceptance: same golden set evaluated on Azure vs Ollama, table in README (quality vs cost vs latency)
### M5 — CI/CD + deploy (week 3–4)
- GitHub Actions: ruff + pytest + 20-question eval smoke; fails on faithfulness drop > 5 points
- Deployed public URL with basic auth; `/health`, `/metrics`
- Acceptance: link in README, 2-minute demo video, architecture diagram (Mermaid), cost per 1,000 queries computed

## Definition of done (what the README must contain)
1. Architecture diagram and the one-paragraph product statement
2. Eval table: faithfulness, answer relevancy, context precision/recall, recall@5, citation accuracy — Azure vs Ollama
3. Latency (p50/p95) and cost per query per backend
4. Experiments table (what you changed, what moved)
5. "Failure modes" section: 5 concrete questions it gets wrong and why
6. Security/compliance section: PII redaction, audit log, data residency option
7. How to run locally in 3 commands; public demo link; video link

## Working rules for Claude Code (important — read before acting)
- **Explain before building.** For each milestone, first write a short plan (files, functions, data flow) and wait for the user's OK.
- **Learning mode is on.** The user is closing a hands-on-coding gap. The following modules are **HAND-WRITTEN BY THE USER**: `chunker.py`, `hybrid_retrieval.py` (merging BM25 + vector scores), and the `recall_at_k` function in the eval harness. For these: explain the approach, give the function signature and tests, then STOP and let the user write the body. Review their code afterwards; do not rewrite it unless asked.
- After any non-trivial change, run the tests and the 20-question smoke eval; report the numbers.
- Never commit secrets; use `.env` + `.env.example`. Never commit `data/raw/`.
- Record every architectural decision in `DECISIONS.md` (one line each: date, decision, why).
- Keep functions small, typed, and tested. Prefer boring, readable code over clever code.
- Languages: code and comments in English. README in English.

## Repo layout
```
docmind/            package (ingest/, retrieval/, llm/, eval/, api/, pii/)
data/raw/           source PDFs (gitignored)     data/eval/golden.jsonl
tests/              pytest                       reports/    eval outputs
docker-compose.yml  postgres+pgvector, langfuse, api
.github/workflows/  ci.yml
README.md  DECISIONS.md  CLAUDE.md
```

## Interview talking points this project should produce
- Why hybrid retrieval beats pure vector on regulatory text (numbers)
- How chunking choices changed recall (numbers)
- Trade-off table cloud LLM vs local LLM for a Swiss bank
- How you evaluate faithfulness and what "citation accuracy" means
- What you'd need to make it FINMA-compliant for real (data residency, retention, access control)
