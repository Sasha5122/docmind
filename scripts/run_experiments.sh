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

# 2. Retrieval ablations (deterministic metrics only; the LLM still answers so citation
#    precision is comparable, but no judge -> faster)
$PY -m docmind.eval $LIMIT_ARG --mode vector   --label vector-only  --no-judge
$PY -m docmind.eval $LIMIT_ARG --mode keyword  --label keyword-only --no-judge
$PY -m docmind.eval $LIMIT_ARG --no-rerank     --label no-rerank    --no-judge
$PY -m docmind.eval $LIMIT_ARG --k 3           --label k3           --no-judge
$PY -m docmind.eval $LIMIT_ARG --k 8           --label k8           --no-judge

$PY -m docmind.eval.compare reports/eval_*.json --out reports/experiments.md
