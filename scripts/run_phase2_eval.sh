#!/bin/bash
# Sharded driver for the Phase-2 evaluation (plain bundle or the tonal study).
# Replaces the single-process 'eval' path for real runs: the 2026-08-13/17
# monolithic runs took 21h/31h silently; sharded 8 ways this finishes in a
# few hours with per-cell progress in logs/phase2_eval*.shard*.log.
#
#   bash scripts/run_phase2_eval.sh [N_SHARDS=8] [tonal]
#
# Cleans stale fragments for the chosen tag first (the report stage refuses
# mixed shard counts), launches N background shard processes at
# OMP_NUM_THREADS=2, waits, then reports.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts

N="${1:-8}"
TAG="${2:-plain}"
case "$N" in (*[!0-9]*|"")
  echo "usage: run_phase2_eval.sh [N_SHARDS=8] [tonal] — first arg must be a number" >&2
  exit 2 ;;
esac
if [ "$TAG" = "tonal" ]; then
  RUN=run-tonal; REPORT=report-tonal; DIR=results/phase2_cells/tonal
else
  RUN=run; REPORT=report; DIR=results/phase2_cells/plain
fi
mkdir -p logs "$DIR"
rm -f "$DIR"/cells.shard*.pkl

pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_phase2_real.py "$RUN" "$K/$N" \
    > "logs/phase2_eval_${TAG}.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_phase2_real.py "$REPORT" | tee "logs/phase2_eval_${TAG}.report.log"
