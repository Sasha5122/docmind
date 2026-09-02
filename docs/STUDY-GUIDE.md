# DocMind Study Guide — the whole project, one idea at a time

*A workbook: read one paragraph, understand one idea, write your own questions in the dotted space under it. Every technical term is explained the first time it appears. All numbers are real, measured on 2026-09-01 on the full 96-question evaluation.*

---

## Part 1 — The big picture

### 1.1 Why this project exists

DocMind is a portfolio project. Its job is to prove to a recruiter in Swiss or EU financial services that you can build a **production-grade** AI system — meaning: not a demo, but something with tests, measured quality numbers, privacy protection, an audit trail, and honest documentation of what doesn't work. Recruiters see hundreds of "I built a chatbot" projects; almost none come with measured numbers. The numbers are the differentiator.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 1.2 What problem it solves

Insurance and banking people spend hours searching long, boring PDFs — regulations, policy conditions, annual reports — often in several languages. DocMind lets them ask a question in plain German, French, English or Italian and get back an answer **with proof**: every fact carries a reference to the exact PDF and page it came from.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 1.3 What RAG is (the core term)

**RAG** stands for *Retrieval-Augmented Generation*. Plain words: an AI language model answering purely from its own memory can be wrong or invent things ("hallucinate"). RAG fixes this by splitting the job in two: first **retrieve** — search the user's own documents for the relevant paragraphs — then **generate** — hand only those paragraphs to the AI with the instruction "answer from this text only, and cite it." The AI becomes a careful reader, not an unreliable know-it-all.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

**🎤 Recruiter questions for Part 1:**

- **Q: Explain RAG to a non-technical stakeholder.**
  A: "Instead of trusting the AI's memory, we make it open the right page of our own documents and quote from it. Like an exam where you may bring the book — but must cite the page."
- **Q: Why RAG instead of fine-tuning a model on the documents?**
  A: Fine-tuning (re-training the model on your data) is expensive, must be redone when documents change, and still can't cite pages. RAG updates instantly when you add a PDF, cites its sources, and lets you audit exactly what the model saw.

---

## Part 2 — The documents

### 2.1 What went in, and from where

18 real, public PDFs, all downloaded free from official websites: three **FINMA circulars** (rules the Swiss financial-market regulator issues to banks and insurers — from finma.ch, each in DE/FR/EN editions), **household-insurance general conditions** ("AVB" — the fine-print booklet of a policy) from Zurich, AXA and Mobiliar (from the insurers' own sites, in DE/FR/IT/EN), and **annual reports** from Swiss Life and Swiss Re (from their investor pages, EN). A script (`scripts/download_corpus.py`) re-downloads all of them, so the corpus is reproducible.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 2.2 Why exactly these documents

Three reasons. They are **realistic** — exactly what a Swiss compliance officer or underwriter actually reads. They are **multilingual** — four languages, which is the hard part of the Swiss market and most demos ignore it. And they are **public** — no licensing or privacy problem in a public portfolio.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

**🎤 Recruiter questions for Part 2:**

- **Q: How would this scale to 10,000 documents?**
  A: The pipeline is already per-file and idempotent (re-running skips already-ingested files via a content hash). At that scale I'd parallelize ingestion, add an HNSW index (a fast approximate-search index) in pgvector, and batch the embedding step on a GPU.

---

## Part 3 — Getting documents in (ingestion)

### 3.1 Parsing: PDF → text with page numbers

A PDF is a printing format, not a text format, so the first step is **parsing** — extracting the raw text. The library **pymupdf** does this fast and page by page. Keeping the page number attached to every piece of text is essential: it is what makes citations like "p. 5" possible later. Each file also gets a **sha256 hash** — a unique fingerprint computed from the file's bytes — so re-ingesting the same file is detected and skipped.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 3.2 Language detection

Each document is labeled de/fr/en/it using **langdetect**, a small statistical library. The label matters twice later: the database builds its keyword index with the right language's grammar rules, and the answer prompt tells the model which language to reply in.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 3.3 PII redaction: privacy before storage

**PII** = *personally identifiable information* — names, emails, phone numbers, IBANs (bank account numbers). Before anything is stored, DocMind replaces PII with placeholders like `<PERSON>` or `<IBAN>`, using **Presidio** (Microsoft's open-source PII scanner: regex patterns for IBAN/email/phone) plus **spaCy** (a language-analysis library whose model recognizes person names in all four languages). Why before storage: a real financial company must never keep customer names in an AI search index. Known honest limitation: the French name detector sometimes over-grabs ("Jean Dupont de Lausanne" gets swallowed whole) — over-redaction, the safe direction of error.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 3.4 Chunking: cutting text into searchable pieces

Search doesn't work well on 100-page documents, so the text is cut into **chunks** of roughly 300–800 **tokens** (a token ≈ ¾ of an English word — the unit AI models actually read). Chunks **overlap** slightly so a sentence sitting on a cut line isn't lost, and every chunk remembers which page it starts on. Too-small chunks lose context; too-big chunks dilute the search signal — the 300–800 range was confirmed by experiment.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 3.5 Embeddings: turning meaning into numbers

An **embedding** turns a piece of text into a list of 1,024 numbers that encodes its *meaning*: texts about the same topic get similar number-lists, even in different words or different languages. DocMind uses **bge-m3**, a free multilingual embedding model that runs locally. Every embedding is normalized to length 1, which makes similarity comparison a simple, fast dot product. Because bge-m3 is multilingual, a German question can find an English paragraph — the meanings land near each other in number-space.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 3.6 Storage: Postgres + pgvector in Docker

Everything lands in **PostgreSQL**, a standard free database, extended with **pgvector**, an add-on that stores those 1,024-number lists and compares them fast. It runs in **Docker** — software in an isolated box, started with one command, identical on any machine. Table changes are managed by **Alembic migrations** — small versioned scripts that replay the same schema steps on the laptop, in CI and in the cloud, so the database structure never drifts. Result: 18 documents → 3,194 chunks with embeddings.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

**🎤 Recruiter questions for Part 3:**

- **Q: Why pgvector and not a dedicated vector database (Pinecone, Weaviate)?**
  A: One boring, battle-tested system instead of two: Postgres already holds the documents, the audit log and the full-text index, does transactions and backups, and every ops team knows it. A dedicated vector DB earns its complexity only at a scale far beyond this corpus.
- **Q: How do you handle PII?**
  A: Redacted at ingestion — before indexing, so no personal data ever enters the search index. Pattern rules for IBAN/email/phone (deterministic, no model needed) plus spaCy for names, with the failure modes documented.
- **Q: How did you pick the chunk size?**
  A: Measured, not guessed: it's one of the ablation experiments in the eval harness.

---

## Part 4 — Answering a question

### 4.1 Vector search: find by meaning

The question is embedded into the same number-space as the chunks; pgvector returns the chunks whose numbers are closest. Strength: finds "same idea, different words" — a question about "Kündigungsfrist" can match a paragraph saying "Auflösung des Vertrags". Weakness: exact codes like "Rz 33" or "FINMA-RS 23/1" barely change the meaning-numbers, so it fumbles precise identifiers.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 4.2 Keyword search: find by exact words

In parallel, Postgres **full-text search** (the classic "BM25-style" method: score chunks by how often the query's stemmed words appear) finds chunks containing the literal words. Its strength is exactly the vector search's weakness: codes, article numbers, product names. It uses per-language stemming — the database knows "gekündigt" and "Kündigung" share a root.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 4.3 Hybrid fusion: merge both lists

The two ranked lists are merged (each retriever contributes its top 20 candidates). Measured on regulatory text: vector-only found the right page in the top 5 for 81% of questions, keyword-only 70%, **hybrid 82%** — and hybrid's real gain shows one step later, because it feeds better candidates to the reranker. This "hybrid beats either alone on regulatory text, with numbers" is the single best interview story in the project.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 4.4 Reranking: a second, careful reading

The first search is fast but shallow. A **reranker** (bge-reranker-v2-m3, a *cross-encoder* — a model that reads question and chunk **together** rather than comparing pre-computed numbers) re-scores the ~20 candidates and keeps the best 5. Measured effect: the right passage reaches the model 94% of the time with the reranker vs 82% without — **+12 points, the biggest single win in the whole system**, for ~0.8 s of extra latency.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 4.5 The LLM writes the answer — with citations

The top 5 chunks are numbered [1]…[5] and pasted into a strict **prompt** (the instruction text given to the model): *answer in the question's language, use only these sources, mark every fact with [n], and if the sources don't contain the answer, say so.* The model is **qwen2.5:7b** running locally via **Ollama** (a tool that runs open AI models on your own machine — cost: $0, data never leaves the laptop). The `[n]` marks in the answer are then mapped back to filename + page for display.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 4.6 Switchable backends: the data-residency story

The LLM sits behind one small interface with three implementations — Azure OpenAI, Anthropic, Ollama — chosen by a single config flag (`LLM_BACKEND=`). Why it matters in Switzerland: a bank may forbid sending document text to a US cloud. Being able to say "flip one flag and everything runs on-premise, here are the measured quality/cost/latency trade-offs" is a compliance argument, not just a technical one.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 4.7 The API and the web page

**FastAPI** (a Python web-server framework) exposes the system as an **API** — a door other programs can call: `POST /ask` answers questions, `/health` reports status, `/metrics` reports latency percentiles, `/documents` lists the library, `/audit` shows the question history. The chat page you see in the browser is one static HTML file served by the same server; it calls the exact same `/ask` the evaluation uses — nothing is special-cased for the demo.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

**🎤 Recruiter questions for Part 4:**

- **Q: Why does hybrid retrieval beat pure vector search on regulatory text?**
  A: Regulatory questions hinge on exact identifiers — "Rz 33", "Circular 23/1" — that embeddings blur but keyword search nails; embeddings still catch paraphrases. Measured: 82% vs 81% vs 70% (hybrid / vector / keyword), and hybrid feeds the reranker better candidates.
- **Q: What does the reranker add, and what does it cost?**
  A: +12 points of "right passage reaches the model" (82%→94%) for ~0.8 s per query. Best quality-per-second trade in the pipeline.
- **Q: How do you prevent hallucinations?**
  A: Three layers: the prompt restricts the model to the retrieved text, every sentence must cite [n], and the eval measures **faithfulness** (95% — see Part 5). Plus the model is told to decline when sources don't contain the answer.

---

## Part 5 — Evaluation (the most important part)

### 5.1 Why evaluation is the heart of the project

Anyone can build a RAG demo that looks right. The professional question is: **how often is it right, and how do you know?** DocMind answers with a repeatable evaluation harness — the same 96 questions run against every configuration, producing comparable numbers. Every design decision above ("reranker on", "k=5", "hybrid") was made by reading these numbers, not by feeling.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 5.2 The golden set

A **golden set** is a hand-checked exam: 96 questions in DE/FR/EN, each with the reference answer and the exact source (file + page) where it's found — about 30 written fully by hand, the rest drafted with an LLM and manually reviewed. It includes 8 deliberately **unanswerable** questions (asking for things the documents don't contain) to test whether the system admits ignorance instead of inventing.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 5.3 Retrieval metrics: did search find the right page?

**Recall@5 = 82%**: for 82% of questions, the correct source page was among the top-5 search results (Recall@10: 92%). **MRR** (*mean reciprocal rank*) **= 0.65**: measures *how high* the right result sits — 1.0 would mean always first place. **Context hit = 94%**: after reranking, the right passage was among the 5 chunks actually handed to the model — the number that matters most, because the model can't cite what it never saw.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 5.4 Answer metrics: was the answer good?

**Faithfulness = 95%**: the answer's claims are supported by the retrieved text (i.e., ~no hallucination). **Correctness = 82%**: the answer matches the reference answer — judged by an **LLM-as-judge** (a second model call that compares answer vs reference; cheaper than human review, consistent enough for comparisons). **Citation precision = 60%**: cited pages are the truly relevant ones. **Citation coverage = 36%**: share of answer sentences carrying a citation mark — the weakest number, honestly documented: the small local model often writes one citation for several sentences.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 5.5 Abstention: knowing when to say "I don't know"

Of the 8 unanswerable trick questions, the system correctly declined 5. The other 3 are documented failure cases — usually the retriever found a *similar-looking* passage and the model answered from it. For a compliance tool, a wrong confident answer is worse than "not in my documents," so this metric is tracked explicitly.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 5.6 Speed and cost

**p50 = 8.5 s, p95 = 14.7 s** per answer (p50/p95 = the time under which 50% / 95% of requests finish). The original goal of p95 < 4 s is **not met with a local 7-billion-parameter model** — the timing breakdown shows ~1 s retrieval, ~0.8 s rerank, and all the rest is the model writing; a cloud model would meet the goal. Cost: **$0 locally**; estimated ≈ **$0.43 per 1,000 questions** on Azure gpt-4o-mini (computed from token counts, not measured — no key). Stating a missed target with the reason is deliberate: honesty is the feature.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 5.7 Ablation experiments: proving each part earns its place

An **ablation** = switch one component off, re-run all 96 questions, compare. Results: no reranker → context hit 94%→82%; keyword-only → recall 70%; vector-only → 81%; widening to 40 candidates → no gain, +0.6 s; shrinking k from 5 to 3 chunks → correctness 82%→79%. This before/after table is what turns "I built it" into "I measured it."

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

**🎤 Recruiter questions for Part 5:**

- **Q: How do you evaluate a RAG system?**
  A: Two layers, separately: *retrieval* (recall@k, MRR — did search find the right page?) and *generation* (faithfulness, correctness via LLM-as-judge, citation accuracy — did the model use it honestly?). Plus abstention on unanswerable questions, latency percentiles, and cost. All on a fixed hand-checked golden set so runs are comparable.
- **Q: What's your system's weakest point?**
  A: Citation coverage (36%) — the local 7B model under-marks its sentences — and 3 of 8 trick questions slipped through. Both documented with examples in the README's failure-modes section.
- **Q: What is faithfulness vs correctness?**
  A: Faithfulness = "did it stick to the retrieved text" (anti-hallucination). Correctness = "was it actually the right answer." An answer can be faithful but wrong if retrieval fetched the wrong page.

---

## Part 6 — Production concerns

### 6.1 The audit trail

Every request becomes a row in an `audit_log` database table: who asked, when, the question, the answer, **which chunks were retrieved and which were cited**, tokens, cost, timings, status. In regulated industries this is mandatory: six months later you must be able to reconstruct *why* the system said what it said. The web UI's "Audit history" button makes the trail visible without database access.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 6.2 Observability

**Langfuse** (a tracing dashboard for LLM apps) records each request as a step-by-step trace — retrieval, rerank, model call, with timings and token counts — so slow or wrong answers can be debugged by looking at what actually happened, not guessing. (Run only when needed — it's heavy for the laptop.)

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 6.3 CI/CD: the robot that guards quality

**CI** (*continuous integration*): on every push to GitHub, a robot runs the linter (code-style checker **ruff**), all ~90 **pytest** tests, and a 20-question smoke evaluation — and **fails the build if faithfulness drops more than 5 points**. That last part is the interesting one: it's a *quality regression gate* — you cannot accidentally merge a change that makes answers worse, the same way you can't merge one that breaks tests.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 6.4 Packaging and deployment

A **Dockerfile** packages the app into an image (a ready-to-run box with Python and all dependencies), targeting Azure Container Apps for a public demo URL with basic-auth (username/password protection). `docker compose up` + two commands runs the whole thing locally — the "3 commands to run" promise in the README.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

### 6.5 The decision log

Every non-obvious choice is one row in `DECISIONS.md`: date, decision, why — including rejected alternatives ("psycopg2 is legacy", "monorepo would bury the README"). Interviewers probe *why* far more than *what*; this file is the prepared answer sheet, and it shows engineering maturity: decisions were conscious, not accidental.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………

**🎤 Recruiter questions for Part 6:**

- **Q: What would it take to make this truly FINMA-compliant for a real bank?**
  A: The architecture has the hooks: local backend for data residency, PII redaction, full audit trail, basic auth. Still needed: proper identity management (SSO), retention policies, encrypted backups, access control per document, and a formal model-risk assessment. I can name these because the gap analysis is part of the design.
- **Q: What breaks first under load?**
  A: The local LLM — it's 80% of latency and single-threaded on this hardware. Fixes in order: cloud backend flag, response streaming, then batching/quantization or a GPU server.

---

## Part 7 — Rapid-fire interview answers (memorize these five numbers)

| Number | What it is | The one-line story |
|---|---|---|
| **82% / 92%** | recall@5 / recall@10 | "Search finds the right page in the top 5 for 4 of 5 questions." |
| **+12 points** | reranker effect (82→94% context hit) | "The single biggest quality win, for 0.8 s." |
| **95% / 82%** | faithfulness / correctness | "It almost never invents; when wrong, it's usually retrieval's fault." |
| **8.5 s / 14.7 s** | p50 / p95 latency (local 7B) | "Honest miss of the 4 s target — it's the local model writing; cloud fixes it." |
| **$0 / ~$0.43** | cost per 1,000 questions, local / Azure estimate | "The data-residency trade-off, quantified." |

*And the three sentences that summarize everything:* DocMind is a multilingual RAG system for Swiss regulatory documents with citations to page level. Its differentiator is the evaluation harness: 96 golden questions, retrieval and answer metrics, ablation experiments, and a CI gate that blocks quality regressions. Every architectural decision in it is written down with its "why" — and its weaknesses are documented as honestly as its strengths.

> ✍️ **Your questions:**
> ………………………………………………………………………………………………
> ………………………………………………………………………………………………
