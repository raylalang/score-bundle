#!/usr/bin/env bash
# Motif study, pilot two (DEVELOPMENT ONLY, 2026-11-17): the per-channel motif
# weight, four seeds, both masking schemes, 21 pieces, 672 fits on N shards.
# Hypotheses H2 in scripts/eval_motif_pilot_dev.py (committed before the run).
#   bash scripts/run_motif_pilot2.sh [N=14]
set -euo pipefail
N=${1:-14}
OUT=results/motif_pilot2
ARMS=time,time_motif,time_motif_pc,time_pitch_motif_pc
PY=/home/ray/miniconda3/envs/score-bundle/bin/python
export OMP_NUM_THREADS=2 PYTHONPATH=src:scripts
[ -f $OUT/pieces.pkl ] || $PY scripts/eval_motif_pilot_dev.py --stage align --out $OUT
rm -f $OUT/cells.shard*.pkl
echo "HEAD $(git rev-parse --short HEAD)  start $(date)  load $(uptime | sed 's/.*load/load/')"
for k in $(seq 0 $((N-1))); do
  $PY scripts/eval_motif_pilot_dev.py --stage run --out $OUT --arms $ARMS --seeds 4 \
      --shard $k/$N > logs/motif_pilot2.shard$k.log 2>&1 &
done
wait
echo "shards done $(date)"
$PY scripts/eval_motif_pilot_dev.py --stage report --out $OUT --arms $ARMS --seeds 4 \
    > logs/motif_pilot2.report.log 2>&1
echo "report done $(date)"
