#!/usr/bin/env bash
# The M3 experiment matrix. Each run writes reports/eval_<stamp>_<label>.{json,md}.
# Retrieval-only runs (--no-judge) are fast; the judged baseline is the slow one.
#
#   bash scripts/run_experiments.sh            # full golden set
#   LIMIT=20 bash scripts/run_experiments.sh   # quick pass
set -euo pipefail
cd "$(dirname "$0")/.."
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
PY=${PY:-uv run python}

# 1. Baseline: hybrid + reranker + LLM judge
$PY -m docmind.eval $LIMIT_ARG --label baseline

# 2. Retrieval ablations: no LLM at all, so each run takes ~1 s per question.
#    recall@5 / MRR score the fused candidate list BEFORE reranking (so the reranker cannot
#    change them); "ctx hit@k" scores the k chunks that WOULD be handed to the LLM, AFTER
#    reranking -- that is the number the --no-rerank row is about.
$PY -m docmind.eval $LIMIT_ARG --retrieval-only --label retrieval-hybrid
$PY -m docmind.eval $LIMIT_ARG --retrieval-only --mode vector   --label retrieval-vector-only
$PY -m docmind.eval $LIMIT_ARG --retrieval-only --mode keyword  --label retrieval-keyword-only
$PY -m docmind.eval $LIMIT_ARG --retrieval-only --no-rerank     --label retrieval-no-rerank
$PY -m docmind.eval $LIMIT_ARG --retrieval-only --candidates 40 --label retrieval-40-candidates

# 3. Answer-quality ablation that needs the LLM: fewer / more chunks in the prompt
$PY -m docmind.eval $LIMIT_ARG --k 3 --label k3

$PY -m docmind.eval.compare reports/eval_*.json --out reports/experiments.md
