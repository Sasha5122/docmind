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
uv run pytest                              # 90 tests; DB tests skip automatically if Docker is off
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

## Corpus (public documents, fetched by `scripts/download_corpus.py`)

| group | documents | languages | pages | chunks |
|---|---|---|---|---|
| FINMA circulars 2023/1 (operational risks), 2026/1 (nature-related risks), 2025/1 (conduct rules) | 8 | DE, FR, EN | 99 | 160 |
| Insurers' general conditions: Zurich household (4 languages), Mobiliar (2), AXA (1) | 7 | DE, FR, EN, IT | 278 | 921 |
| Annual reports: Swiss Life AR 2024, Swiss Re financial statements + financial condition report 2024 | 3 | EN | 737 | 1,091 |
| **total** | **18** | 4 | **1,114** | **2,172** |

Ingest of the whole corpus takes ~4 min on the laptop GPU (RTX 3050, fp16) and ~1.5 h on
CPU; 413 PII spans were replaced (273 of them in the Swiss Life report — board members,
auditors' signatures), 7.6 % of chunks carry a `<PERSON>` placeholder.

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
- **context hit@k** — recall measured on the k chunks actually handed to the LLM, *after* reranking (recall@k is measured before the reranker, so this is the only number the reranker can move)
- **citation precision** — share of the answer's citations that point at a correct page
- **citation coverage** — share of factual sentences that carry a citation
- **faithfulness** — share of answer statements supported by the retrieved passages (RAGAS definition, LLM-as-judge)
- **answer correctness** — 0 / 0.5 / 1 agreement with the reference answer (LLM-as-judge)
- **abstention rate** — how often the system says "not in the documents"

### Results

<!-- RESULTS:BEGIN -->
Baseline run `baseline` — 96 questions, LLM `ollama:qwen2.5:7b`, retrieval `hybrid` (k=5, 20 candidates), reranker `CrossEncoderReranker`, judge `ollama:qwen2.5:7b`, 2026-09-01, machine `DESKTOP-F3N5QRF`.

| metric | all | de | en | fr |
|---|---|---|---|---|
| recall@5 | 82 % | 86 % | 66 % | 100 % |
| context hit@k (after reranking) | 94 % | 98 % | 86 % | 100 % |
| citation precision | 60 % | 73 % | 40 % | 61 % |
| faithfulness (judge) | 95 % | 95 % | 95 % | 96 % |
| answer correctness (judge) | 82 % | 85 % | 73 % | 88 % |
| recall@10 | 92 % |  |  |  |
| MRR | 0.65 |  |  |  |
| citation coverage | 36 % |  |  |  |
| abstention rate | 6 % |  |  |  |

| by category | n | recall@5 | citation precision | faithfulness | correctness |
|---|---|---|---|---|---|
| cross-lingual | 12 | 75 % | 66 % | 95 % | 95 % |
| fact | 70 | 87 % | 58 % | 97 % | 84 % |
| multi-doc | 6 | 33 % | 69 % | 69 % | 67 % |
| unanswerable | 8 | – | – | 100 % | 62 % |

**Latency** p50 **8.45 s**, p95 **14.66 s** (mean: retrieval 1.05 s · rerank 0.79 s · LLM 8.24 s). Tokens per question ≈ 2636 in / 65 out. Cost per 1,000 questions: $0.00.
<!-- RESULTS:END -->

**Reading the numbers**

- Acceptance criteria of milestone 2: *answer in the language of the question* — met (judged
  answers, spot-checked); *every fact cited* — partly: 60 % of the citations point at a correct
  page, but citation **coverage** is only 36 % because the 7B model tends to put one `[n]` at the
  end of a paragraph instead of after every sentence; *p95 < 4 s* — **not met** with a local 7B
  model on a 6 GB laptop GPU: retrieval + reranking take 1.8 s, the remaining ~8 s is token
  generation at ~29 tok/s. A hosted model (Azure gpt-4o-mini, typically 1.5–3 s) meets the
  target with identical retrieval numbers — see the backend table below.
- The reranker is what makes the system work: before it the correct page is in the top 5 for
  82 % of the questions, after it for 94 % (context hit@k).
- French is easiest (100 % recall, all 19 questions are about policy conditions), English
  hardest (66 %): every English question targets a 400–700-page annual report whose tables
  extract as scrambled text.
- Unanswerable questions: 5 of 8 correctly declined, 3 answered — see failure modes.
- One question (`zurich-012`) is excluded from the latency statistics: the laptop went into
  standby during that call (84 min wall-clock). The report records the exclusion in
  `config.latency_outliers_excluded`; all other metrics include the question.

### Experiments

<!-- EXPERIMENTS:BEGIN -->
| run | n | LLM | recall@5 | recall@10 | MRR | ctx hit@k | cit. precision | faithfulness | correctness | abstain | p50 s | p95 s | $/1000 q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 96 | ollama:qwen2.5:7b | 82 % | 92 % | 0.65 | 94 % | 60 % | 95 % | 82 % | 6 % | 8.45 | 14.66 | 0.00 |
| retrieval-hybrid | 96 | none:retrieval-only | 82 % | 92 % | 0.65 | 94 % | – | – | – | – | 0.79 | 2.45 | 0.00 |
| retrieval-vector-only | 96 | none:retrieval-only | 81 % | 89 % | 0.64 | 91 % | – | – | – | – | 0.72 | 0.84 | 0.00 |
| retrieval-keyword-only | 96 | none:retrieval-only | 70 % | 84 % | 0.48 | 85 % | – | – | – | – | 0.68 | 0.87 | 0.00 |
| retrieval-no-rerank | 96 | none:retrieval-only | 82 % | 92 % | 0.65 | 82 % | – | – | – | – | 0.13 | 0.18 | 0.00 |
| retrieval-40-candidates | 96 | none:retrieval-only | 84 % | 88 % | 0.67 | 94 % | – | – | – | – | 1.36 | 1.59 | 0.00 |
| k3 | 96 | ollama:qwen2.5:7b | 82 % | 92 % | 0.65 | 89 % | 62 % | 95 % | 79 % | 5 % | 9.07 | 13.21 | 0.00 |
<!-- EXPERIMENTS:END -->

Retrieval-only rows use no model (`--retrieval-only`), so only retrieval metrics and latency
apply. What the table says:

- **Hybrid vs a single retriever.** Hybrid (82 % / 92 % recall@5 / @10) beats keyword-only
  clearly (70 % / 84 %) and vector-only slightly (81 % / 89 %); after reranking, 94 % vs 91 %
  context hit. On the 20-question smoke set the gap to vector-only was 20 points — a reminder
  not to draw conclusions from 20 questions. Hybrid is kept: it costs 0.1 s, cross-lingual
  questions depend on the vector side (exact terms are useless across languages), and codes
  like "RS 23/1" depend on the keyword side.
- **Reranker on / off.** recall@5 is identical by construction (measured before the reranker);
  context hit@k drops from 94 % to 82 % without it. Worth its 0.8 s.
- **40 instead of 20 candidates per retriever.** recall@5 +2 points, recall@10 −4 (more
  candidates dilute the fused ranking), context hit unchanged, +0.6 s. Not adopted.
- **k = 3 instead of 5 chunks in the prompt.** Context hit 89 % vs 94 %, correctness 79 % vs
  82 %, citation precision +2 points: less to cite wrongly, but also less to answer from. k = 5
  is kept.

### Azure vs local (Ollama)

The code path is identical (`LLM_BACKEND`); the same golden set runs against either. This
repository was developed without cloud credentials, so the measured column is the local one.
To fill the Azure column: set `AZURE_OPENAI_API_KEY/ENDPOINT` in `.env`, `LLM_BACKEND=azure`,
and run `uv run python -m docmind.eval --label azure-baseline` (≈ USD 0.05 for 96 questions
with gpt-4o-mini; cost is computed per call from token counts in `docmind/llm/base.py`).

<!-- BACKENDS:BEGIN -->
| | Ollama `qwen2.5:7b` (RTX 3050 6 GB, measured) | Azure gpt-4o-mini (not measured — no key) |
|---|---|---|
| faithfulness / correctness | 95 % / 82 % | run `LLM_BACKEND=azure` to fill |
| recall@5 / citation precision | 82 % / 60 % | same retrieval → identical |
| latency p50 / p95 | 8.45 s / 14.66 s | typically 1.5–3 s end to end |
| cost per 1,000 questions | $0 (≈ 2636+65 tokens/question, local GPU) | ≈ $0.43 at list price |
| data leaves the machine | no | yes (chosen Azure region) |
<!-- BACKENDS:END -->

---

## Failure modes (honest list)

<!-- FAILURES:BEGIN -->
From the baseline run: 16 retrieval misses (correct page not in top 5), 3 wrong answers despite correct retrieval, 3 of 8 unanswerable questions answered instead of abstaining.

- **Retrieval miss — `finma-003`** (de): *Wie definiert das FINMA-RS 23/1 die Recovery Time Objective (RTO) und die Recovery Point Objective (RPO)?* — expected finma-rs-2023-01-oprisk-de.pdf p.4; top-3 retrieved: swissre-financial-condition-report-2024-en.pdf p.95, finma-rs-2023-01-oprisk-en.pdf p.4, swissre-financial-condition-report-2024-en.pdf p.22.
- **Retrieval miss — `zurich-008`** (de): *Bis zu welchem Betrag übernimmt Zurich bei Schlüsselverlust die Kosten für das Ändern oder Ersetzen von Schlössern an den versicherten Standorten?* — expected zurich-avb-haushalt-de.pdf p.30; top-3 retrieved: zurich-avb-haushalt-de.pdf p.53, zurich-avb-haushalt-de.pdf p.17, zurich-cga-economia-domestica-it.pdf p.51.
- **Retrieval miss — `zurich-017`** (en): *Within what time window are successive earthquakes or volcanic eruptions treated as a single loss event under the Zurich household insurance?* — expected zurich-avb-haushalt-de.pdf p.14, zurich-cga-menage-fr.pdf p.14; top-3 retrieved: zurich-gtc-household-en.pdf p.20, zurich-gtc-household-en.pdf p.4, zurich-gtc-household-en.pdf p.12.
- **Wrong answer with the right page — `finma-017`** (en): *Under FINMA Circular 2023/1, which two elements make up the definition of critical functions?* → answered “The critical functions and their associated tolerances for disruption according to margin no. 14 must be approved at least annually by the board of directors. […”; reference: “Critical functions include (a) the activities, processes and services, including their underlying resources, whose disru”.
- **Wrong answer with the right page — `avb-014`** (de): *Gemäss den Mobiliar AVB Minima: Bis zu welchem Anteil der Versicherungssumme und mindestens bis zu welchem Betrag sind Kosten (z. B. zusätzliche Lebenshaltungskosten, Räumung, Schlossänderung) bei einem versicherten Hausratschaden gedeckt?* → answered “Bis zu 20 % der Versicherungssumme der Hausrat-Grunddeckung, mindestens CHF 30 000 [1].…”; reference: “Die effektiven Kosten sind bis zu 10 % der Versicherungssumme der Hausrat-Grunddeckung gedeckt, mindestens jedoch CHF 50”.
- **Did not abstain — `none-001`** (de): *Wie hoch ist die Jahresprämie für die Zurich Haushaltversicherung für eine 3-Zimmer-Wohnung in Bern?* → “Die Höhe der Prämie hängt von den versicherten Risiken und dem gewünschten Versicherungsschutz ab. Es wird keine spezifische Prämienrate für eine 3-Zimmer-Wohnu…” (nothing in the corpus answers this).
- **Did not abstain — `none-003`** (fr): *Quel est le montant maximal remboursé par la Mobilière pour un vol de vélo électrique à l'étranger ?* → “Le montant maximal remboursé par la Mobilière pour un vol de vélo électrique à l'étranger est de CHF 2 millions. [4][5]…” (nothing in the corpus answers this).
<!-- FAILURES:END -->

What the misses have in common:

- **The same clause exists in four language editions** (3 of the 16 misses: `zurich-017/019/021`).
  The Zurich household conditions are in the corpus in DE, FR, IT and EN. The golden set lists
  the DE and FR pages; the system retrieved the EN edition, answered correctly (judge: correct
  and faithful) and cited the EN page — the page-based metric still counts a miss. The fix
  belongs in the golden set (list every edition), not in the system.
- **Annual-report tables** (7 of the 16 misses are `ar-*` questions in the 400–700-page Swiss
  Life / Swiss Re reports). Figures such as a solvency ratio live in tables that pymupdf
  extracts as scrambled columns, and the same figure appears in several sections (summary,
  segment report, notes), so a "wrong" page is often a page that states the same number.
  Layout-aware parsing (e.g. table extraction to Markdown) is the next step.
- **Multi-document questions** (recall@5 33 %, context hit 83 %): "deductible at AXA vs
  Mobiliar" fills the fused top 5 with chunks of the insurer named first; the reranker rescues
  three of the four, but `ar-017` (Swiss Life vs Swiss Re) is missed entirely. The standard fix
  — split into one sub-query per insurer and merge — is not implemented.
- **Citations per paragraph, not per sentence** (coverage 36 %): the local 7B model cites
  correctly but sparsely. A stricter prompt or a post-check that rejects uncited sentences
  would raise coverage at the cost of latency.
- **Abstentions are paraphrased.** The prompt asks for one fixed sentence ("Dazu finde ich in den
  Dokumenten keine Angaben"); the model writes "wird in den Quellen nicht erwähnt" or "the
  provided sources do not contain…". The first scoring pass therefore reported 0 % abstention;
  the check is now a set of multilingual patterns (`metrics.is_abstention`) and stored answers
  were re-scored — a lesson about validating the metric before trusting the number.

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
