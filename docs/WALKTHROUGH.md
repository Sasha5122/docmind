# DocMind — the complete walkthrough

*Written for the project owner, in plain language. Every technical word is explained the
first time it appears. Read this top to bottom once and you will be able to explain every
file, every number and every decision in the repository in an interview.*

---

## 0. What DocMind is, in one breath

DocMind is a small web service. You give it a pile of regulated PDF documents (insurance
conditions, FINMA circulars, annual reports) in German, French, English or Italian. You ask
it a question in any of those languages. It answers in your language and **every factual
sentence ends with a citation** `[n]` that points to a specific document and page. Personal
data was removed from the documents before they were stored. Every question is written to an
audit table. The language model that writes the answer can be a cloud model (Azure OpenAI,
Anthropic) or a model running on your own machine (Ollama) — one setting switches it.
And, most importantly, there is an **evaluation harness** that measures how good the answers
are with numbers, so we can prove improvements instead of claiming them.

The technique is called **RAG — Retrieval-Augmented Generation**: instead of asking a
language model to answer from memory (where it invents things), we first *retrieve* the
relevant passages from our own documents and then let the model *generate* an answer only
from those passages.

---

## 1. The pipeline, stage by stage

```
PDF ──parse──▶ pages ──redact PII──▶ clean pages ──chunk──▶ chunks ──embed──▶ vectors ──▶ Postgres (+pgvector, +full-text)

question ──▶ hybrid retrieval (vector + keyword, fused) ──▶ reranker ──▶ prompt with numbered sources ──▶ LLM
         ──▶ answer + citations + cost + latency ──▶ audit_log table ──▶ Langfuse trace ──▶ JSON response
```

### 1.1 Parse — `docmind/ingest/parser.py`
**What:** opens a PDF with `pymupdf` and returns one text block per page, plus the file's
`sha256` fingerprint and its language (`langdetect`).
**Why page by page:** a citation must say "page 12". If we lost page boundaries here we could
never get them back.
**Why sha256:** a *hash* is a fingerprint of the file bytes. Two identical files have the
same hash, so re-running ingestion skips files we already stored.

### 1.2 Redact PII — `docmind/pii/redactor.py`
**What:** finds personal data and replaces it with placeholders: `<PERSON>`, `<IBAN>`,
`<EMAIL>`, `<PHONE>`. Uses Microsoft **Presidio**, a toolkit that combines *pattern rules*
(an IBAN always looks like `CH93 0076 …`) with a *language model* (spaCy) for names.
**Why before chunking and storing:** nothing personal must ever reach the database or the
language model. Doing it per page means a placeholder never straddles a chunk boundary.
**The lesson we learned (documented failure + fix):** the first version used one small
multilingual spaCy model. On our corpus it marked **63 % of all chunks** as containing a
person — it thought "Number of shares", "Schadenfalls" and "Cash" were people. That would
have poisoned search results. We switched to one small model per language and added a
filter: a span only counts as a name if it is 2–4 Title-Case words without digits ("Hans
Muster" yes, "Cash" no). We accept missing lone surnames in exchange for precision.
**Known limitation:** an IBAN broken over a PDF line break is not detected (the pattern
does not span lines).

### 1.3 Chunk — `docmind/ingest/chunker.py` *(interview module)*
**What:** cuts each document into pieces of at most 500 *tokens* with a 50-token overlap.
A **token** is a word piece — roughly ¾ of a word in English; we count them with
`tiktoken`.
**Why chunk at all:** one embedding represents one "meaning". A whole PDF is far too much
meaning for one vector; one sentence has too little context. A few hundred tokens is the
standard compromise.
**The algorithm, step by step:**
1. Split each page into paragraphs (blank lines).
2. Walk the paragraphs in order and pack them into the current chunk until adding the next
   one would exceed 500 tokens; then close the chunk and start a new one.
3. A single paragraph longer than the limit is split by words.
4. Each new chunk starts with the last ~50 tokens of the previous chunk (**overlap**), so a
   sentence cut at a boundary is whole in at least one chunk.
5. A chunk remembers the page its first *original* paragraph came from; chunks may run over
   a page boundary and cite the page they start on. (That is why the evaluation tolerates
   ±1 page.)

### 1.4 Embed — `docmind/ingest/embedder.py`
**What:** turns each chunk into an **embedding**: a list of 1,024 numbers produced by a
neural network such that texts with similar meaning get lists pointing in similar
directions. Similarity is measured with the **cosine** of the angle between two lists.
**Model:** `BAAI/bge-m3`, a multilingual model — a German chunk and a French question about
the same thing land close together. That is what makes cross-language questions work.
**Alternative wired in:** Azure `text-embedding-3-large`, asked for 1,024 dimensions so it
fits the same database column. `EMBEDDING_BACKEND=local|azure` switches.
**Why all vectors are normalised to length 1:** then cosine similarity is just a dot
product and one database operator (`<=>`) works for both backends.
**Hardware note:** on the laptop CPU, embedding ran at ~1.6 s per chunk (≈1.5 h per full
ingest). On the RTX 3050 GPU in half precision (`float16`) it is ~0.08 s per chunk. The
project pins the CUDA build of PyTorch on Windows and the small CPU build on Linux (CI).

### 1.5 Store — `docmind/models.py`, `migrations/`
**Tables:** `documents` (one row per PDF), `chunks` (text, page, token count, the
1,024-number `embedding`, and `tsv`), `audit_log` (one row per question).
**pgvector** is a Postgres extension that adds a `vector` column type and fast
nearest-neighbour search. We chose Postgres + pgvector over a dedicated vector database
because one boring database that also holds the audit log and full-text index is easier
to run, back up and explain.
**`tsv`** is a *generated column*: Postgres itself turns each chunk into a bag of
stemmed words using the stemmer of the chunk's language (german/french/italian/english),
so "Schäden" and "Schaden" match. A **GIN** index makes it fast.
**HNSW index** on the embedding column makes vector search take milliseconds.
**Alembic migrations** are versioned SQL steps that create/alter these tables the same
way on the laptop, in CI and in the cloud, so schemas never drift.

### 1.6 Retrieve — `docmind/retrieval/search.py` and `hybrid.py` *(interview module)*
Two searches run for every question:
- **Vector search**: the question is embedded and the 20 nearest chunks by cosine are
  fetched. Great at paraphrases and cross-language, weak at exact tokens.
- **Keyword search**: Postgres full-text search on `tsv`, ranked by `ts_rank_cd`.
  Great at exact tokens ("Art. 12", "FINMA-RS 2023/1"), blind to meaning.
  *Bug we fixed:* Postgres' default query builder ANDs every word, so "Wie lang ist die
  Kündigungsfrist?" only matched chunks containing *lang* as well. We OR the stemmed
  terms; ranking still rewards chunks that match more terms.
- **Fusion — Reciprocal Rank Fusion (RRF):** each chunk's score is
  `Σ 1/(60 + rank)` over the lists it appears in. Only *ranks* are used, never raw scores,
  so we never have to calibrate a cosine (0–1) against a `ts_rank` (0–∞). A chunk found by
  both searches naturally floats to the top.
- A `mode` switch (`hybrid | vector | keyword`) exists purely so the evaluation can prove
  hybrid is worth it.

### 1.7 Rerank — `docmind/retrieval/reranker.py`
**What:** a **cross-encoder** (`BAAI/bge-reranker-v2-m3`) reads the question *and* each
candidate chunk together and outputs a relevance score. The first-stage searches compared
the question to each chunk separately; the cross-encoder sees them jointly and is far more
precise — but slow per pair, so it only runs on the top 20 candidates and keeps 5.
A `NoopReranker` implements the "reranker off" experiment.

### 1.8 Answer — `docmind/llm/`, `docmind/rag.py`
- `llm/base.py` defines one contract: `complete(system, user) -> text, tokens, cost,
  latency`. Every backend implements it: `AzureOpenAILLM`, `AnthropicLLM`, `OllamaLLM`
  (`llm/backends.py`). `LLM_BACKEND=azure|anthropic|ollama` picks one. This is the
  **data-residency switch**: with `ollama`, no document text ever leaves the machine.
- `llm/prompt.py` builds the prompt: the five chunks are listed as `[1] (file, page) …`,
  and the model is told to end every factual sentence with `[n]`, to answer in the
  question's language, and to say so when the sources do not contain the answer.
  We map `[n]` back to `(file, page, passage)` in code, so a citation can only ever point
  at something that was actually retrieved. Numbers are used because small local models
  reproduce `[2]` reliably but garble file names.
- `rag.py::answer_question` is the whole flow with timings for each stage and the
  cost, and it is written so that the embedder, reranker and LLM are *passed in* — tests
  use fakes and the evaluation can swap one piece at a time.

### 1.9 Serve — `docmind/api/app.py`
FastAPI service: `POST /ask`, `GET /health`, `GET /metrics` (p50/p95 latency), and a
one-page web UI at `/` (`static/index.html`). HTTP Basic auth is enforced when
`BASIC_AUTH_USER/PASSWORD` are set. Each answer writes an `audit_log` row
(`docmind/api/audit.py`) and a Langfuse trace (`docmind/observability.py`) when keys
are configured.

---

## 2. Evaluation — the part reviewers care about most

### 2.1 The golden set — `data/eval/golden.jsonl`
96 questions: 45 German, 32 English, 19 French. Categories: 70 *fact*, 12 *cross-lingual*
(question in one language, evidence in another), 6 *multi-doc* (needs two documents),
8 *unanswerable* (the correct behaviour is to say "not in the documents"). Each answerable
item carries the exact evidence sentence and the PDF page index; a script verified that
the evidence really appears on the cited page.

### 2.2 The metrics — `docmind/eval/metrics.py`
| metric | meaning in plain words | needs an LLM? |
|---|---|---|
| **recall@5 / @10** | Was the right page among the first 5 / 10 chunks retrieved? If not, no model can answer correctly. | no |
| **MRR** | How high up was the first correct chunk (1 = top, ½ = second …)? | no |
| **context hit@k** | Same question as recall, but asked about the k chunks that were actually handed to the LLM *after* reranking. Recall@k looks *before* the reranker, so only this metric can show whether the reranker helps or hurts. | no |
| **citation precision** | Of the citations in the answer, what share point at a correct page? | no |
| **citation coverage** | What share of factual sentences carry a citation at all? | no |
| **faithfulness** | Of the statements in the answer, what share is supported by the retrieved passages? (RAGAS definition, LLM-as-judge) | yes |
| **answer correctness** | Does the answer agree with the reference answer? 0 / 0.5 / 1 (LLM-as-judge) | yes |
| **abstention rate** | How often did it say "not in the documents"? Should be high on unanswerable questions and low elsewhere. | no |

Why we did not import the RAGAS library: it pulls in LangChain and expects an OpenAI-style
judge; our judge runs through the same `LLM` contract, so evaluation works offline with
Ollama too. The definitions follow RAGAS.

### 2.3 Running it — `python -m docmind.eval`
Flags: `--limit 20` (smoke set), `--mode vector|keyword`, `--no-rerank`, `--no-judge`,
`--k`. Writes `reports/eval_<stamp>_<label>.json` (every question, every number), a
markdown summary, and `reports/latest.json` (summary only, committed) for the CI gate.

### 2.4 The regression gate — `docmind/eval/gate.py`
CI compares `reports/latest.json` with `reports/baseline.json` and fails when
faithfulness, correctness, recall@5 or citation precision drop by more than 5 points.

### 2.5 Results
*(filled in from the report files once the runs are complete — see README.)*

---

## 3. Operations

- **Docker Compose**: `db` (Postgres + pgvector); profile `api` builds and runs the service;
  profile `observability` runs a self-hosted Langfuse v2 with pre-created keys.
- **Dockerfile**: `python:3.12-slim` + `uv`; dependencies first for layer caching; the two
  large models download on first start into a volume (they would make the image 5 GB).
- **CI (`.github/workflows/ci.yml`)**: ruff lint + format check, migrations against a
  pgvector service container, the full pytest suite, and the eval gate.
- **`eval-smoke.yml`**: on demand / weekly, ingests the public corpus and runs the
  20-question smoke eval against a cloud model using repository secrets.
- **Secrets**: only in `.env` (git-ignored); `.env.example` documents every setting.

---

## 4. Tech stack — and why each piece

| piece | what it is | why this one |
|---|---|---|
| Python 3.12 + `uv` | language + very fast package manager with a lockfile | reproducible installs on laptop, CI and container in seconds |
| FastAPI + Pydantic v2 | web framework with typed, validated requests | request validation for free, OpenAPI docs at `/docs` |
| SQLAlchemy 2 + Alembic | typed tables + versioned migrations | schema lives in code and replays identically everywhere |
| Postgres 16 + pgvector | relational DB + vector search | one system for vectors, full-text, audit log |
| pymupdf, langdetect | PDF text + language | fast, keeps page numbers; tiny language detector |
| Presidio + spaCy | PII detection | rules for IBAN/e-mail/phone, statistical model for names |
| tiktoken | token counting | fast, no model download |
| sentence-transformers: bge-m3, bge-reranker-v2-m3 | multilingual embeddings + reranker | strong open models covering DE/FR/EN/IT; run locally |
| openai / anthropic SDKs, httpx (Ollama) | LLM clients | three backends behind one interface |
| Ollama + qwen2.5:7b | local LLM runtime + model | fits a 6 GB laptop GPU, good German/French |
| Langfuse | LLM tracing UI | per-request traces with tokens, cost, latency |
| pytest, ruff | tests, lint/format | 80+ tests; DB tests skip automatically when Docker is off |
| GitHub Actions | CI | lint + tests + eval gate on every push |
| Docker | packaging | same image locally and in the cloud |

---

## 5. Decision log
See `DECISIONS.md` — one line per decision with the reason. The most consequential ones:
per-page PII redaction; OR-ed keyword terms; RRF fusion; numbered-source citations;
native eval metrics; GPU float16 loading; per-language spaCy models with a name filter.

## 6. Where to look for what
```
docmind/ingest/     parser, redactor (in pii/), chunker, embedder, pipeline, CLI
docmind/retrieval/  search (vector + keyword), hybrid (RRF), reranker
docmind/llm/        base contract, backends, prompt + citation parsing
docmind/rag.py      the question → answer flow
docmind/api/        FastAPI app, audit writer
docmind/eval/       golden loader, metrics, runner, CLI, CI gate
docmind/observability.py  Langfuse tracer
migrations/         3 Alembic revisions (tables; tsv + indexes; audit_log)
scripts/download_corpus.py  fetches the 18 public PDFs into data/raw/
data/eval/golden.jsonl      the 96 evaluation questions
reports/            eval outputs (baseline.json + latest.json committed)
tests/              one test file per module
```
