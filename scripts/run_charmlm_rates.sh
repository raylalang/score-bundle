#!/bin/bash
# Harmonic-edge deciding check, remaining rates: c_harm_lm at obs 0.50/0.70/0.80
# (obs 0.90 already run, results/graphgp_charmlm_obs0.90 — the tie breaks there).
# Sequential rates, 12 shards each; pair against results/graphgp_masksweep/obsX/
# b_featlm via scripts/report_charmlm_rates.py. Validation only.
set -euo pipefail
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src
mkdir -p logs
for OF in 0.50 0.70 0.80; do
  pids=()
  for K in $(seq 0 11); do
    OMP_NUM_THREADS=2 python scripts/eval_graphgp.py --stage run --guard \
      --configs c_harm_lm \
      --inputs .cache/masksweep_inputs_obs$OF.pkl \
      --emb-dump .cache/masksweep_emb_obs$OF.pkl \
      --out-dir results/graphgp_charmlm_obs$OF --shard "$K/12" \
      > "logs/charmlm_obs$OF.shard$K.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "[charmlm] obs$OF done $(date)"
done
echo "CHARMLM_RATES_DONE $(date)"
