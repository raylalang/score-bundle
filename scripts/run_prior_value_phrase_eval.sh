#!/bin/bash
# Sharded driver for the phrase-scale prior-value experiment.
#   bash scripts/run_prior_value_phrase_eval.sh [N_SHARDS=12]
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts
N="${1:-12}"
DIR=results/phase3_cells
mkdir -p logs "$DIR"
rm -f "$DIR"/priorval_phrase.shard*.pkl
pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_prior_value_phrase_dev.py run "$K/$N" \
    > "logs/priorval_phrase_eval.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_prior_value_phrase_dev.py report | tee "logs/priorval_phrase_eval.report.log"
