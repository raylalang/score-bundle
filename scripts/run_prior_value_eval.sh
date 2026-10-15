#!/bin/bash
# Sharded driver for the prior-value experiment (eval_prior_value_dev.py).
# Long-run rule: run the `smoke` verb first and check the projection.
#
#   bash scripts/run_prior_value_eval.sh [N_SHARDS=8] [TUNING_SD_CENTS]
# With the second argument the tuning-scalar follow-up runs instead
# (separate cell files + results file; the first run is never clobbered).
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts

N="${1:-8}"
case "$N" in (*[!0-9]*|"")
  echo "usage: run_prior_value_eval.sh [N_SHARDS=8] [TUNING_SD_CENTS]" >&2; exit 2 ;;
esac
DIR=results/phase3_cells
mkdir -p logs "$DIR"
if [ -n "${2:-}" ]; then
  export PRIORVAL_TUNING_SD="$2"
  PREFIX=priorval_tuning; TAG=priorval_tuning_eval
else
  PREFIX=priorval; TAG=priorval_eval
fi
rm -f "$DIR"/${PREFIX}.shard*.pkl

pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_prior_value_dev.py run "$K/$N" \
    > "logs/${TAG}.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_prior_value_dev.py report | tee "logs/${TAG}.report.log"
