#!/bin/bash
# Sharded driver for the prior-value experiment (eval_prior_value_dev.py).
# Long-run rule: run the `smoke` verb first and check the projection.
#
#   bash scripts/run_prior_value_eval.sh [N_SHARDS=8] [TUNING_SD_CENTS|-] [v2]
# Second argument: the tuning-scalar follow-up ("-" = none). Third
# argument "v2": protocol v2 (hidden note acoustically absent). Each
# variant has its own cell files + results file; nothing is clobbered.
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
PREFIX=priorval
if [ "${3:-}" = "v2" ]; then
  export PRIORVAL_DROP_HIDDEN=1; PREFIX=${PREFIX}_v2
fi
if [ -n "${2:-}" ] && [ "${2}" != "-" ]; then
  export PRIORVAL_TUNING_SD="$2"; PREFIX=${PREFIX}_tuning
fi
TAG=${PREFIX}_eval
rm -f "$DIR"/${PREFIX}.shard*.pkl

pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_prior_value_dev.py run "$K/$N" \
    > "logs/${TAG}.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_prior_value_dev.py report | tee "logs/${TAG}.report.log"
