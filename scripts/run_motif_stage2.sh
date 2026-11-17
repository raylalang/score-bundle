#!/usr/bin/env bash
# Motif study, stage two (DEVELOPMENT ONLY, 2026-11-18): the edge family from
# the ISMIR 2026 model's held-out predicted occurrences (camera-ready sweep,
# quantile 0.86 = the paper's configured tau_occ_quantile), masks and cells on
# the annotated occurrences as in pilots one and two.  Hypothesis H3 in
# scripts/eval_motif_pilot_dev.py.   bash scripts/run_motif_stage2.sh [N=14]
set -euo pipefail
N=${1:-14}
OUT=results/motif_stage2
ARMS=time,time_motif_pred,time_motif_pc_pred
PY=/home/ray/miniconda3/envs/score-bundle/bin/python
export OMP_NUM_THREADS=2 PYTHONPATH=src:scripts
COMMON="--out $OUT --occ-source pred --pred-q 86 --arms $ARMS --schemes occ --seeds 4"
[ -f $OUT/pieces.pkl ] || $PY scripts/eval_motif_pilot_dev.py --stage align $COMMON
rm -f $OUT/cells.shard*.pkl
echo "HEAD $(git rev-parse --short HEAD)  start $(date)  load $(uptime | sed 's/.*load/load/')"
for k in $(seq 0 $((N-1))); do
  $PY scripts/eval_motif_pilot_dev.py --stage run $COMMON --shard $k/$N \
      > logs/motif_stage2.shard$k.log 2>&1 &
done
wait
echo "shards done $(date)"
$PY scripts/eval_motif_pilot_dev.py --stage report $COMMON > logs/motif_stage2.report.log 2>&1
echo "report done $(date)"
