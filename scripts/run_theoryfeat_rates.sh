#!/bin/bash
# Theory-feature configs at the four non-anchor rates (queued after the
# kernel-variant runs to avoid oversubscription). Validation only.
set -euo pipefail
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src
until [ "$(ls results/kernels_ms_obs0.50/tonal.pkl results/kernels_ms_obs0.70/tonal.pkl results/kernels_ms_obs0.80/tonal.pkl results/kernels_ms_obs0.90/tonal.pkl 2>/dev/null | wc -l)" -ge 4 ]; do sleep 300; done
for OF in 0.50 0.70 0.80 0.90; do
  pids=()
  for K in $(seq 0 11); do
    OMP_NUM_THREADS=2 python scripts/eval_graphgp.py --stage run --guard \
      --configs b_theoryfeat,b_theoryfeatlm \
      --inputs .cache/masksweep_inputs_obs$OF.pkl \
      --emb-dump .cache/masksweep_emb_obs$OF.pkl \
      --out-dir results/graphgp_theoryfeat_obs$OF --shard "$K/12" \
      > "logs/theoryfeat_obs$OF.shard$K.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "[theoryfeat] obs$OF done $(date)"
done
echo THEORYFEAT_RATES_DONE
