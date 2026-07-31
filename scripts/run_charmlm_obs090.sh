#!/bin/bash
# Harmonic-edge deciding check, first rate: c_harm_lm at obs 0.90 (10% hidden),
# where the two-stage harmonic effect is largest (docs/kernel_multirate_results.md).
# Pair against results/graphgp_masksweep/obs0.90/b_featlm. Validation only.
set -euo pipefail
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src
mkdir -p logs
pids=()
for K in $(seq 0 11); do
  OMP_NUM_THREADS=2 python scripts/eval_graphgp.py --stage run --guard \
    --configs c_harm_lm \
    --inputs .cache/masksweep_inputs_obs0.90.pkl \
    --emb-dump .cache/masksweep_emb_obs0.90.pkl \
    --out-dir results/graphgp_charmlm_obs0.90 --shard "$K/12" \
    > "logs/charmlm_obs0.90.shard$K.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "CHARMLM_OBS090_DONE $(date)"
