#!/bin/bash
# Sustain-overlap edge A/B (motif-graph borrow, task 2026-07-14): c_graph vs
# c_overlap, guard-on, on the published 40%-hidden anchor masks — paired cells
# isolate the overlap relation's marginal value (c_overlap = c_graph + one
# learned overlap weight).  DEV pieces only; confirmation never read.
# Waits for the theory-feature A/B shards to finish before starting so the two
# jobs never contend.
set -euo pipefail
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src
mkdir -p logs results/graphgp_overlap

# wait for the 24 theory-feature shard pickles (12 shards x 2 configs)
while [ "$(ls results/graphgp_theoryfeat/*.shard*.pkl 2>/dev/null | wc -l)" -lt 24 ]; do
  sleep 120
done
echo "[overlap-ab] theoryfeat done, starting $(date)"

pids=()
for K in $(seq 0 11); do
  OMP_NUM_THREADS=2 python scripts/eval_graphgp.py --stage run --guard \
    --configs c_graph,c_overlap --out-dir results/graphgp_overlap \
    --shard "$K/12" > "logs/overlap_run.shard$K.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "OVERLAP AB DONE $(date)"
