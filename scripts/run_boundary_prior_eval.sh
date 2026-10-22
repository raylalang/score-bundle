#!/bin/bash
# Sharded driver for the boundary-prior experiment (eval_boundary_prior_dev.py).
# Long-run rule: run the `smoke` verb first and check the projection.
#   bash scripts/run_boundary_prior_eval.sh [N_SHARDS=8]
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts
N="${1:-8}"
DIR=results/phase3_cells
PREFIX=boundprior; TAG=boundprior_eval
if [ "${2:-}" = "chain" ]; then
  export BOUNDPRIOR_SIGMA=chain; PREFIX=boundprior_chainsig; TAG=boundprior_chainsig_eval
fi
mkdir -p logs "$DIR"
rm -f "$DIR"/${PREFIX}.shard*.pkl
pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_boundary_prior_dev.py run "$K/$N" \
    > "logs/${TAG}.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_boundary_prior_dev.py report | tee "logs/${TAG}.report.log"
