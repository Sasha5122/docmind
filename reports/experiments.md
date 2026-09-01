| run | n | LLM | recall@5 | recall@10 | MRR | ctx hit@k | cit. precision | faithfulness | correctness | abstain | p50 s | p95 s | $/1000 q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 96 | ollama:qwen2.5:7b | 82 % | 92 % | 0.65 | 94 % | 60 % | 95 % | 82 % | 6 % | 8.45 | 14.66 | 0.00 |
| retrieval-hybrid | 96 | none:retrieval-only | 82 % | 92 % | 0.65 | 94 % | – | – | – | – | 0.79 | 2.45 | 0.00 |
| retrieval-vector-only | 96 | none:retrieval-only | 81 % | 89 % | 0.64 | 91 % | – | – | – | – | 0.72 | 0.84 | 0.00 |
| retrieval-keyword-only | 96 | none:retrieval-only | 70 % | 84 % | 0.48 | 85 % | – | – | – | – | 0.68 | 0.87 | 0.00 |
| retrieval-no-rerank | 96 | none:retrieval-only | 82 % | 92 % | 0.65 | 82 % | – | – | – | – | 0.13 | 0.18 | 0.00 |
| retrieval-40-candidates | 96 | none:retrieval-only | 84 % | 88 % | 0.67 | 94 % | – | – | – | – | 1.36 | 1.59 | 0.00 |
| k3 | 96 | ollama:qwen2.5:7b | 82 % | 92 % | 0.65 | 89 % | 62 % | 95 % | 79 % | 5 % | 9.07 | 13.21 | 0.00 |
