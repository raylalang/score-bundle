#!/bin/bash
# Sharded driver for the Phase-3 chain/boundary dev study
# (eval_phase3_chain_dev.py).  Long-run rule: run the `smoke` verb first
# and check the printed projection before launching this.
#
#   bash scripts/run_phase3_chain_eval.sh [N_SHARDS=8]
#
# Cleans stale chain.shard*.pkl fragments (the report stage merges by
# glob), launches N background shard processes at OMP_NUM_THREADS=2 with
# per-shard logs, waits, then reports.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts

N="${1:-8}"
case "$N" in (*[!0-9]*|"")
  echo "usage: run_phase3_chain_eval.sh [N_SHARDS=8] — arg must be a number" >&2
  exit 2 ;;
esac
DIR=results/phase3_cells
mkdir -p logs "$DIR"
rm -f "$DIR"/chain.shard*.pkl

pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_phase3_chain_dev.py run "$K/$N" \
    > "logs/phase3_chain_eval.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_phase3_chain_dev.py report | tee "logs/phase3_chain_eval.report.log"
