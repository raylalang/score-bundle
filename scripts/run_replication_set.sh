#!/bin/bash
# Post-hoc REPLICATION set (2026-07-16): the dev ladder re-run on 30 fresh ASAP
# pieces (eval positions 50-79 of the same seed-0 shuffle, extracted tonight to
# .cache/asap_arrays_named80.pkl; positions 0-49 verified byte-identical to the
# published cache, so the confirmation pieces 31-50 are untouched and the fresh
# pieces were never seen by any selection decision).  NOT confirmation-grade
# (no preregistration) — label: replication set.  Guard on; masks/embeddings
# from a disjoint seed base (7000), leak-free per mask.
set -euo pipefail
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src
mkdir -p logs results/graphgp_repl

INP=.cache/repl_inputs.pkl
EMB=.cache/repl_emb.pkl
if [ ! -f "$INP" ] || [ ! -f "$EMB" ]; then
  python scripts/eval_kernels.py --stage precompute \
    --arrays-cache .cache/asap_arrays_named80.pkl \
    --eval-start 50 --n-eval-pieces 30 --mask-seed-base 7000 --mean feat_lm \
    --inputs "$INP" --dump-embeddings "$EMB" \
    > logs/repl_precompute.log 2>&1
  echo "[repl] precompute done $(date)"
fi

pids=()
for K in $(seq 0 11); do
  OMP_NUM_THREADS=2 python scripts/eval_graphgp.py --stage run --guard \
    --arrays-cache .cache/asap_arrays_named80.pkl --eval-start 50 \
    --configs b_feat,b_featlm,b_featlm_nograph \
    --inputs "$INP" --emb-dump "$EMB" \
    --out-dir results/graphgp_repl --shard "$K/12" \
    > "logs/repl_run.shard$K.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "REPLICATION RUN DONE $(date)"
