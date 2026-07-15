#!/bin/bash
# Masking-level sweep on the DEV pieces (requested 2026-07-13): hidden fraction
# 50 / 40 / 30 / 20 / 10 % plus the leave-one-out limit (scripts/eval_gp_loo.py).
#
# DEV ONLY: eval_start=0 everywhere, 30 pieces — the confirmation set (eval-list
# offset 30+, pieces 31-50) is never read.  Guard ON (new-run policy); the
# obs0.60_anchor block re-runs the published 40%-hidden masks under the same
# guard-on setting so every fraction is like-for-like within the sweep, with
# results/graphgp_v2 as the external cross-check.
#
# Mask-aware embeddings are recomputed per fraction (leak-free requires the LM
# input to blank exactly the hidden notes of each mask) — GPU precompute via
# scripts/eval_kernels.py --stage precompute, disjoint mask-seed bases 50xx.
set -euo pipefail
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src
mkdir -p logs results/graphgp_masksweep

CONFIGS=b_feat,b_featlm,b_featlm_nograph
SHARDS=12

run_frac () {  # $1 tag  $2 inputs pkl  $3 embedding dump
  local OUT=results/graphgp_masksweep/$1
  mkdir -p "$OUT"
  local pids=()
  for K in $(seq 0 $((SHARDS - 1))); do
    OMP_NUM_THREADS=2 python scripts/eval_graphgp.py --stage run --guard \
      --configs "$CONFIGS" --inputs "$2" --emb-dump "$3" \
      --out-dir "$OUT" --shard "$K/$SHARDS" \
      > "logs/masksweep_run_$1.shard$K.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "[sweep] $1 done $(date)"
}

# 0) anchor: published 40%-hidden masks/embeddings, guard-on
run_frac obs0.60_anchor .cache/kernel_sweep_inputs.pkl .cache/kernel_sweep_emb_ma.pkl

# 1) new fractions (observed 0.50/0.70/0.80/0.90 = hidden 50/30/20/10%)
for OF in 0.50 0.70 0.80 0.90; do
  TAG=obs$OF
  INP=.cache/masksweep_inputs_$TAG.pkl
  EMB=.cache/masksweep_emb_$TAG.pkl
  BASE=$((5000 + ${OF/0./}))
  if [ ! -f "$INP" ] || [ ! -f "$EMB" ]; then
    python scripts/eval_kernels.py --stage precompute --observed-frac "$OF" \
      --mask-seed-base "$BASE" --mean feat_lm --inputs "$INP" \
      --dump-embeddings "$EMB" > "logs/masksweep_precompute_$TAG.log" 2>&1
    echo "[sweep] precompute $TAG done $(date)"
  fi
  run_frac "$TAG" "$INP" "$EMB"
done

# 2) leave-one-out limit
OMP_NUM_THREADS=2 python scripts/eval_gp_loo.py --guard --configs "$CONFIGS" \
  --out results/graphgp_masksweep/loo.pkl > logs/masksweep_loo.log 2>&1
echo "[sweep] loo done $(date)"

echo "MASK SWEEP DONE $(date)"
