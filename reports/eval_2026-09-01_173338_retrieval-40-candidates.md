# Eval: retrieval-40-candidates

2026-09-01T14:33:38+00:00 · 96 questions · LLM `none:retrieval-only` · retrieval `hybrid` k=5 · reranker `CrossEncoderReranker` · judge `None`

| metric | value |
|---|---|
| recall@5 | 84 % |
| recall@10 | 88 % |
| MRR | 0.67 |
| context hit@k (correct page among the k chunks given to the LLM, after reranking) | 94 % |
| context MRR | 0.80 |
| citation precision | – |
| citation coverage | 100 % |
| faithfulness (judge) | – |
| answer correctness (judge) | 0 % |
| abstention rate | 0 % |
| latency p50 / p95 | 1.36 s / 1.591 s |
| of which retrieval / rerank / LLM (mean) | 0.328 / 1.308 / 0.0 s |
| cost per 1000 questions | $0.0 |
| errors | 0 |

## By language

| lang | n | recall@5 | citation precision | faithfulness | correctness |
|---|---|---|---|---|---|
| de | 45 | 90 % | – | – | 0 % |
| en | 32 | 66 % | – | – | 0 % |
| fr | 19 | 100 % | – | – | 0 % |

## By category

| category | n | recall@5 | citation precision | faithfulness | correctness |
|---|---|---|---|---|---|
| cross-lingual | 12 | 75 % | – | – | – |
| fact | 70 | 90 % | – | – | – |
| multi-doc | 6 | 33 % | – | – | – |
| unanswerable | 8 | – | – | – | 0 % |

## Retrieval misses (recall@5 = 0)

- **zurich-008** (de) expected zurich-avb-haushalt-de.pdf p.30 — got zurich-avb-haushalt-de.pdf p.53, zurich-avb-haushalt-de.pdf p.51, zurich-avb-haushalt-de.pdf p.52
- **zurich-017** (en) expected zurich-avb-haushalt-de.pdf p.14, zurich-cga-menage-fr.pdf p.14 — got zurich-gtc-household-en.pdf p.20, zurich-gtc-household-en.pdf p.4, zurich-gtc-household-en.pdf p.12
- **zurich-019** (en) expected zurich-avb-haushalt-de.pdf p.7, zurich-cga-menage-fr.pdf p.7 — got zurich-gtc-household-en.pdf p.6, zurich-gtc-household-en.pdf p.5, zurich-gtc-household-en.pdf p.35
- **zurich-021** (en) expected zurich-avb-haushalt-de.pdf p.11, zurich-cga-menage-fr.pdf p.11 — got zurich-gtc-household-en.pdf p.6, zurich-gtc-household-en.pdf p.6, zurich-gtc-household-en.pdf p.5
- **avb-015** (de) expected mobiliar-avb-hausrat-minima-de.pdf p.13 — got zurich-avb-haushalt-de.pdf p.15, zurich-avb-haushalt-de.pdf p.29, zurich-avb-haushalt-de.pdf p.29
- **avb-019** (de) expected axa-avb-haushalt-de.pdf p.6, mobiliar-avb-hausrat-junge-de.pdf p.14 — got mobiliar-avb-hausrat-junge-de.pdf p.1, mobiliar-avb-hausrat-minima-de.pdf p.13, mobiliar-avb-hausrat-junge-de.pdf p.2
- **avb-021** (de) expected axa-avb-haushalt-de.pdf p.9, mobiliar-avb-hausrat-junge-de.pdf p.17 — got mobiliar-avb-hausrat-junge-de.pdf p.2, mobiliar-avb-hausrat-junge-de.pdf p.1, axa-avb-haushalt-de.pdf p.12
- **ar-003** (en) expected swisslife-annual-report-2024-en.pdf p.7 — got swisslife-annual-report-2024-en.pdf p.226, swisslife-annual-report-2024-en.pdf p.227, swisslife-annual-report-2024-en.pdf p.239
- **ar-004** (en) expected swisslife-annual-report-2024-en.pdf p.17 — got swisslife-annual-report-2024-en.pdf p.5, swisslife-annual-report-2024-en.pdf p.426, swisslife-annual-report-2024-en.pdf p.70
- **ar-005** (en) expected swisslife-annual-report-2024-en.pdf p.77 — got swisslife-annual-report-2024-en.pdf p.10, swisslife-annual-report-2024-en.pdf p.382, swisslife-annual-report-2024-en.pdf p.5
- **ar-014** (en) expected swissre-annual-report-2024-financial-statements-en.pdf p.171 — got swissre-annual-report-2024-financial-statements-en.pdf p.161, swissre-financial-condition-report-2024-en.pdf p.10, swisslife-annual-report-2024-en.pdf p.413
- **ar-016** (en) expected swissre-annual-report-2024-financial-statements-en.pdf p.38 — got swissre-annual-report-2024-financial-statements-en.pdf p.36, swissre-annual-report-2024-financial-statements-en.pdf p.36, swissre-financial-condition-report-2024-en.pdf p.58
- **ar-017** (en) expected swisslife-annual-report-2024-en.pdf p.77, swissre-annual-report-2024-financial-statements-en.pdf p.128 — got swissre-financial-condition-report-2024-en.pdf p.6, swissre-financial-condition-report-2024-en.pdf p.26, swisslife-annual-report-2024-en.pdf p.382
- **ar-018** (en) expected swisslife-annual-report-2024-en.pdf p.7, swissre-annual-report-2024-financial-statements-en.pdf p.182 — got swisslife-annual-report-2024-en.pdf p.226, swisslife-annual-report-2024-en.pdf p.5, swissre-annual-report-2024-financial-statements-en.pdf p.130
