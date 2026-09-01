# Eval: retrieval-vector-only

2026-09-01T14:28:51+00:00 · 96 questions · LLM `none:retrieval-only` · retrieval `vector` k=5 · reranker `CrossEncoderReranker` · judge `None`

| metric | value |
|---|---|
| recall@5 | 81 % |
| recall@10 | 89 % |
| MRR | 0.64 |
| context hit@k (correct page among the k chunks given to the LLM, after reranking) | 91 % |
| context MRR | 0.76 |
| citation precision | – |
| citation coverage | 100 % |
| faithfulness (judge) | – |
| answer correctness (judge) | 0 % |
| abstention rate | 0 % |
| latency p50 / p95 | 0.718 s / 0.837 s |
| of which retrieval / rerank / LLM (mean) | 0.353 / 0.724 / 0.0 s |
| cost per 1000 questions | $0.0 |
| errors | 0 |

## By language

| lang | n | recall@5 | citation precision | faithfulness | correctness |
|---|---|---|---|---|---|
| de | 45 | 86 % | – | – | 0 % |
| en | 32 | 72 % | – | – | 0 % |
| fr | 19 | 82 % | – | – | 0 % |

## By category

| category | n | recall@5 | citation precision | faithfulness | correctness |
|---|---|---|---|---|---|
| cross-lingual | 12 | 92 % | – | – | – |
| fact | 70 | 80 % | – | – | – |
| multi-doc | 6 | 67 % | – | – | – |
| unanswerable | 8 | – | – | – | 0 % |

## Retrieval misses (recall@5 = 0)

- **finma-003** (de) expected finma-rs-2023-01-oprisk-de.pdf p.4 — got swissre-financial-condition-report-2024-en.pdf p.22, swissre-financial-condition-report-2024-en.pdf p.95, swisslife-annual-report-2024-en.pdf p.70
- **finma-007** (de) expected finma-rs-2026-01-nature-risks-de.pdf p.4 — got zurich-cga-economia-domestica-it.pdf p.19, zurich-cga-economia-domestica-it.pdf p.22, zurich-cga-economia-domestica-it.pdf p.21
- **finma-009** (fr) expected finma-rs-2023-01-oprisk-fr.pdf p.3 — got finma-rs-2023-01-oprisk-fr.pdf p.1, finma-rs-2023-01-oprisk-en.pdf p.1, finma-rs-2023-01-oprisk-de.pdf p.1
- **finma-013** (fr) expected finma-rs-2026-01-nature-risks-fr.pdf p.6 — got finma-rs-2026-01-nature-risks-fr.pdf p.1, finma-rs-2026-01-nature-risks-en.pdf p.1, finma-rs-2026-01-nature-risks-fr.pdf p.4
- **finma-020** (en) expected finma-rs-2026-01-nature-risks-en.pdf p.4 — got finma-rs-2026-01-nature-risks-en.pdf p.1, finma-rs-2026-01-nature-risks-fr.pdf p.1, finma-rs-2023-01-oprisk-en.pdf p.1
- **zurich-008** (de) expected zurich-avb-haushalt-de.pdf p.30 — got zurich-cga-economia-domestica-it.pdf p.51, zurich-gtc-household-en.pdf p.23, zurich-gtc-household-en.pdf p.6
- **zurich-009** (de) expected zurich-avb-haushalt-de.pdf p.36 — got swissre-annual-report-2024-financial-statements-en.pdf p.103, swisslife-annual-report-2024-en.pdf p.331, swisslife-annual-report-2024-en.pdf p.357
- **zurich-018** (fr) expected zurich-avb-haushalt-de.pdf p.14, zurich-cga-menage-fr.pdf p.14 — got zurich-gtc-household-en.pdf p.12, zurich-gtc-household-en.pdf p.17, zurich-avb-haushalt-de.pdf p.23
- **avb-015** (de) expected mobiliar-avb-hausrat-minima-de.pdf p.13 — got zurich-cga-economia-domestica-it.pdf p.28, zurich-gtc-household-en.pdf p.26, zurich-cga-economia-domestica-it.pdf p.28
- **avb-017** (de) expected mobiliar-avb-hausrat-minima-de.pdf p.14 — got swisslife-annual-report-2024-en.pdf p.351, swisslife-annual-report-2024-en.pdf p.340, swissre-annual-report-2024-financial-statements-en.pdf p.69
- **ar-003** (en) expected swisslife-annual-report-2024-en.pdf p.7 — got swisslife-annual-report-2024-en.pdf p.226, swisslife-annual-report-2024-en.pdf p.11, swisslife-annual-report-2024-en.pdf p.227
- **ar-005** (en) expected swisslife-annual-report-2024-en.pdf p.77 — got swisslife-annual-report-2024-en.pdf p.382, swisslife-annual-report-2024-en.pdf p.10, swisslife-annual-report-2024-en.pdf p.226
- **ar-011** (en) expected swissre-annual-report-2024-financial-statements-en.pdf p.128 — got swissre-financial-condition-report-2024-en.pdf p.6, swissre-annual-report-2024-financial-statements-en.pdf p.130, swissre-financial-condition-report-2024-en.pdf p.26
- **ar-014** (en) expected swissre-annual-report-2024-financial-statements-en.pdf p.171 — got swissre-annual-report-2024-financial-statements-en.pdf p.1, swissre-financial-condition-report-2024-en.pdf p.1, swissre-financial-condition-report-2024-en.pdf p.10
- **ar-015** (en) expected swissre-annual-report-2024-financial-statements-en.pdf p.12 — got swissre-financial-condition-report-2024-en.pdf p.5, swissre-annual-report-2024-financial-statements-en.pdf p.180, swissre-annual-report-2024-financial-statements-en.pdf p.176
