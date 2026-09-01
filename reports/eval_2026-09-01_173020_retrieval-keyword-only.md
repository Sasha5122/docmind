# Eval: retrieval-keyword-only

2026-09-01T14:30:20+00:00 · 96 questions · LLM `none:retrieval-only` · retrieval `keyword` k=5 · reranker `CrossEncoderReranker` · judge `None`

| metric | value |
|---|---|
| recall@5 | 70 % |
| recall@10 | 84 % |
| MRR | 0.48 |
| context hit@k (correct page among the k chunks given to the LLM, after reranking) | 85 % |
| context MRR | 0.78 |
| citation precision | – |
| citation coverage | 100 % |
| faithfulness (judge) | – |
| answer correctness (judge) | 0 % |
| abstention rate | 0 % |
| latency p50 / p95 | 0.681 s / 0.868 s |
| of which retrieval / rerank / LLM (mean) | 0.054 / 0.794 / 0.0 s |
| cost per 1000 questions | $0.0 |
| errors | 0 |

## By language

| lang | n | recall@5 | citation precision | faithfulness | correctness |
|---|---|---|---|---|---|
| de | 45 | 74 % | – | – | 0 % |
| en | 32 | 62 % | – | – | 0 % |
| fr | 19 | 76 % | – | – | 0 % |

## By category

| category | n | recall@5 | citation precision | faithfulness | correctness |
|---|---|---|---|---|---|
| cross-lingual | 12 | 67 % | – | – | – |
| fact | 70 | 74 % | – | – | – |
| multi-doc | 6 | 33 % | – | – | – |
| unanswerable | 8 | – | – | – | 0 % |

## Retrieval misses (recall@5 = 0)

- **finma-001** (de) expected finma-rs-2023-01-oprisk-de.pdf p.5 — got swissre-financial-condition-report-2024-en.pdf p.80, swissre-annual-report-2024-financial-statements-en.pdf p.6, swisslife-annual-report-2024-en.pdf p.304
- **finma-003** (de) expected finma-rs-2023-01-oprisk-de.pdf p.4 — got finma-rs-2023-01-oprisk-en.pdf p.4, finma-rs-2026-01-nature-risks-de.pdf p.8, finma-rs-2026-01-nature-risks-en.pdf p.8
- **finma-008** (de) expected finma-rs-2026-01-nature-risks-de.pdf p.6 — got finma-rs-2025-01-verhaltenspflichten-de.pdf p.4, axa-avb-haushalt-de.pdf p.24, finma-rs-2025-01-verhaltenspflichten-de.pdf p.4
- **finma-009** (fr) expected finma-rs-2023-01-oprisk-fr.pdf p.3 — got finma-rs-2026-01-nature-risks-fr.pdf p.8, finma-rs-2026-01-nature-risks-fr.pdf p.4, finma-rs-2023-01-oprisk-fr.pdf p.5
- **zurich-003** (de) expected zurich-avb-haushalt-de.pdf p.9 — got axa-avb-haushalt-de.pdf p.6, mobiliar-avb-hausrat-minima-de.pdf p.7, zurich-avb-haushalt-de.pdf p.7
- **zurich-008** (de) expected zurich-avb-haushalt-de.pdf p.30 — got zurich-avb-haushalt-de.pdf p.17, zurich-avb-haushalt-de.pdf p.51, zurich-avb-haushalt-de.pdf p.52
- **zurich-011** (fr) expected zurich-cga-menage-fr.pdf p.10 — got zurich-cga-menage-fr.pdf p.29, zurich-cga-menage-fr.pdf p.12, zurich-cga-menage-fr.pdf p.29
- **zurich-013** (fr) expected zurich-cga-menage-fr.pdf p.28 — got zurich-cga-menage-fr.pdf p.8, zurich-cga-menage-fr.pdf p.35, zurich-cga-menage-fr.pdf p.35
- **zurich-017** (en) expected zurich-avb-haushalt-de.pdf p.14, zurich-cga-menage-fr.pdf p.14 — got zurich-gtc-household-en.pdf p.26, zurich-gtc-household-en.pdf p.4, zurich-gtc-household-en.pdf p.20
- **zurich-018** (fr) expected zurich-avb-haushalt-de.pdf p.14, zurich-cga-menage-fr.pdf p.14 — got zurich-cga-menage-fr.pdf p.12, zurich-cga-menage-fr.pdf p.12, zurich-cga-menage-fr.pdf p.29
- **zurich-019** (en) expected zurich-avb-haushalt-de.pdf p.7, zurich-cga-menage-fr.pdf p.7 — got zurich-gtc-household-en.pdf p.26, zurich-gtc-household-en.pdf p.31, zurich-gtc-household-en.pdf p.4
- **zurich-021** (en) expected zurich-avb-haushalt-de.pdf p.11, zurich-cga-menage-fr.pdf p.11 — got swisslife-annual-report-2024-en.pdf p.244, swissre-annual-report-2024-financial-statements-en.pdf p.24, zurich-gtc-household-en.pdf p.26
- **avb-007** (de) expected mobiliar-avb-hausrat-junge-de.pdf p.12 — got zurich-avb-haushalt-de.pdf p.51, zurich-avb-haushalt-de.pdf p.12, axa-avb-haushalt-de.pdf p.23
- **avb-012** (de) expected mobiliar-avb-hausrat-junge-de.pdf p.29 — got mobiliar-avb-hausrat-junge-de.pdf p.1, mobiliar-avb-hausrat-junge-de.pdf p.2, mobiliar-avb-hausrat-junge-de.pdf p.6
- **avb-014** (de) expected mobiliar-avb-hausrat-minima-de.pdf p.12 — got zurich-avb-haushalt-de.pdf p.17, zurich-avb-haushalt-de.pdf p.51, axa-avb-haushalt-de.pdf p.13
