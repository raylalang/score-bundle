#!/bin/bash
# =============================================================================
#  THE RE-USED-POOL ONE-SHOTS (registered 2026-10-27).  READ BEFORE RUNNING.
#
#  Two claims, each run ONCE on the 13-piece URMP confirmation pool that was
#  spent on 2026-08-27 for claims C1-C4 and is RE-USED WITH FULL DISCLOSURE:
#    corrnoise  docs/corrnoise_prereg_design.md      tag corrnoise-registration-2026-10-27
#    tonal      docs/phase2_tonal_prereg_design.md   tag phase2-tonal-registration-2026-10-27
#  No reruns, no added seeds, no post-hoc filters; every number reported
#  whatever the outcome. Inputs are the archived 2026-08-27 targets (md5
#  checked); no extraction stage runs; the spent artifacts are never touched.
#
#  Refuses unless ALL THREE are set:
#      export PHASE2_SPLIT=confirmation
#      export PHASE2_CONFIRMATION_I_AM_SURE=yes
#      export PHASE2_REUSE_I_AM_SURE=yes
#  and refuses if the claim's results file or evidence directory exists.
#
#  Usage: bash scripts/run_phase2_reuse_confirmation.sh {corrnoise|tonal} [N=6]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CLAIM="${1:-}"
N="${2:-6}"
if [ "${PHASE2_SPLIT:-}" != "confirmation" ] || \
   [ "${PHASE2_CONFIRMATION_I_AM_SURE:-}" != "yes" ] || \
   [ "${PHASE2_REUSE_I_AM_SURE:-}" != "yes" ]; then
  echo "REFUSED: this runs a registered one-shot on the re-used pool. Read the"
  echo "banner in scripts/run_phase2_reuse_confirmation.sh and set all THREE"
  echo "environment variables only when the spend has been deliberately agreed."
  exit 1
fi
case "$CLAIM" in
  corrnoise)
    TAG=corrnoise-registration-2026-10-27
    OUT_MD=results/corrnoise_confirmation_results.md
    CELLS=results/phase2_cells/conf_corrnoise ;;
  tonal)
    TAG=phase2-tonal-registration-2026-10-27
    OUT_MD=results/phase2_tonal_confirmation_results.md
    CELLS=results/phase2_cells/conf_tonal ;;
  *) echo "REFUSED: claim must be corrnoise or tonal"; exit 1 ;;
esac
EVID=evidence/phase2_reuse_confirmation/$CLAIM
if [ -e "$OUT_MD" ] || [ -e "$EVID" ]; then
  echo "REFUSED: $CLAIM was already run once ($OUT_MD or $EVID exists)."
  exit 1
fi

# frozen code state: the registration tag must match the working tree
FROZEN="docs/corrnoise_prereg_design.md docs/phase2_tonal_prereg_design.md \
scripts/eval_corrnoise_tau.py scripts/eval_phase2_real.py \
scripts/run_phase2_reuse_confirmation.sh src/score_bundle/gp.py \
src/score_bundle/graph.py src/score_bundle/baselines.py \
src/score_bundle/phase2/splits.py"
if ! git diff --quiet "$TAG" -- $FROZEN || [ -n "$(git status --porcelain -- $FROZEN)" ]; then
  echo "REFUSED: frozen files differ from tag $TAG"; exit 1
fi

source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts

# inputs: the archived 2026-08-27 targets, verbatim
ARCH=evidence/phase2_confirmation/urmp_targets_conf_inputs.pkl
MD5=6a44d902589a1a0369f551cea1b991a9
if [ ! -e .cache/urmp_targets_conf.pkl ]; then cp "$ARCH" .cache/urmp_targets_conf.pkl; fi
echo "$MD5  .cache/urmp_targets_conf.pkl" | md5sum -c - || { echo "REFUSED: input cache md5 mismatch"; exit 1; }
python - <<'EOF'
import pickle
d = pickle.load(open(".cache/urmp_targets_conf.pkl", "rb"))
keys = sorted(d)
assert len(keys) == 40, len(keys)
print("input cache: 40 unique confirmation tracks, keys", keys[0], "..", keys[-1])
EOF

mkdir -p logs "$CELLS"
rm -f "$CELLS"/cells*.pkl
LOG=logs/phase2_reuse_${CLAIM}.log
{
  echo "== $CLAIM re-use one-shot start $(date -Is)"
  echo "code state $(git rev-parse HEAD) = tag $TAG ($(git rev-parse "$TAG"))"
  echo "shards $N x OMP_NUM_THREADS=4"
} | tee "$LOG"

pids=()
for K in $(seq 0 $((N - 1))); do
  if [ "$CLAIM" = corrnoise ]; then
    OMP_NUM_THREADS=4 python scripts/eval_corrnoise_tau.py run --shard "$K/$N" \
      > "logs/phase2_reuse_${CLAIM}.shard${K}.log" 2>&1 &
  else
    OMP_NUM_THREADS=4 python scripts/eval_phase2_real.py run-tonal "$K/$N" \
      > "logs/phase2_reuse_${CLAIM}.shard${K}.log" 2>&1 &
  fi
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
if [ "$CLAIM" = corrnoise ]; then
  python scripts/eval_corrnoise_tau.py report --md "$OUT_MD" 2>&1 | tee -a "$LOG"
else
  python scripts/eval_phase2_real.py report-tonal 2>&1 | tee -a "$LOG"
fi
echo "== $CLAIM re-use one-shot done $(date -Is)" | tee -a "$LOG"
echo "Archive $LOG, the shard logs and $CELLS/cells*.pkl into $EVID and commit"
echo "IMMEDIATELY; then score the verdict by the frozen rule and append it to $OUT_MD."
