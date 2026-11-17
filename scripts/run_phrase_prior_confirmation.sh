#!/bin/bash
# =============================================================================
#  THE PHRASE-SCALE WAVEFORM ONE-SHOT (registered; docs/phrase_prior_prereg_design.md).
#  READ BEFORE RUNNING.
#
#  Runs scripts/eval_prior_value_phrase_dev.py ONCE on the 40 unique tracks of
#  the 13-piece URMP confirmation pool: spent 2026-08-27 (C1-C4), re-used with
#  disclosure 2026-10-27 (correlated noise, tonal metric), and used here a THIRD
#  time with disclosure, the first time at the waveform level.  Claim W1 and
#  the decision rule are frozen in the registration doc and the tag below.
#  No reruns, no added cells, no post-hoc filters; every number reported.
#
#  Refuses unless ALL THREE are set:
#      export PHASE2_SPLIT=confirmation
#      export PHASE2_CONFIRMATION_I_AM_SURE=yes
#      export PHASE2_REUSE_I_AM_SURE=yes
#  and refuses if the results file or the evidence directory exists, or if a
#  frozen file differs from the registration tag.
#
#  Usage: bash scripts/run_phrase_prior_confirmation.sh [N=12]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-12}"
TAG=phrase-prior-registration-2026-11-18
OUT_MD=results/phrase_prior_confirmation_results.md
CELLS=results/phase3_cells/conf_phrase
EVID=evidence/phrase_prior_confirmation
if [ "${PHASE2_SPLIT:-}" != "confirmation" ] || \
   [ "${PHASE2_CONFIRMATION_I_AM_SURE:-}" != "yes" ] || \
   [ "${PHASE2_REUSE_I_AM_SURE:-}" != "yes" ]; then
  echo "REFUSED: this runs a registered one-shot on the re-used confirmation pool."
  echo "Read the banner in scripts/run_phrase_prior_confirmation.sh and set all"
  echo "THREE environment variables only when the spend has been deliberately agreed."
  exit 1
fi
if [ -e "$OUT_MD" ] || [ -e "$EVID" ]; then
  echo "REFUSED: the phrase-scale one-shot was already run once ($OUT_MD or $EVID exists)."
  exit 1
fi
FROZEN="docs/phrase_prior_prereg_design.md scripts/eval_prior_value_phrase_dev.py \
scripts/eval_prior_value_dev.py scripts/eval_phase3_chain_dev.py \
scripts/eval_phase3_curve_dev.py scripts/eval_phase2_real.py \
scripts/run_phrase_prior_confirmation.sh src/score_bundle/phase3/curve.py \
src/score_bundle/phase3/synth.py src/score_bundle/phase3/waveform_model.py \
src/score_bundle/graph.py src/score_bundle/phase2/splits.py .cache/phase3_prior_fit.json"
if ! git diff --quiet "$TAG" -- $FROZEN || [ -n "$(git status --porcelain -- $FROZEN)" ]; then
  echo "REFUSED: frozen files differ from tag $TAG"; exit 1
fi

source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts

ARCH=evidence/phase2_confirmation/urmp_targets_conf_inputs.pkl
MD5=6a44d902589a1a0369f551cea1b991a9
if [ ! -e .cache/urmp_targets_conf.pkl ]; then cp "$ARCH" .cache/urmp_targets_conf.pkl; fi
echo "$MD5  .cache/urmp_targets_conf.pkl" | md5sum -c - || { echo "REFUSED: input cache md5 mismatch"; exit 1; }
PRIOR_MD5=d6e41db36d86031c4a1151f90cc242e6   # evidence/phrase_prior_registration/phase3_prior_fit.json
echo "$PRIOR_MD5  .cache/phase3_prior_fit.json" | md5sum -c - || { echo "REFUSED: curve prior file differs from the registered copy"; exit 1; }

mkdir -p logs "$CELLS"
rm -f "$CELLS"/${PREFIX:-priorval_phrase_conf}.shard*.pkl
LOG=logs/phrase_prior_confirmation.log
{
  echo "== phrase-scale one-shot start $(date -Is)"
  echo "code state $(git rev-parse HEAD) = tag $TAG ($(git rev-parse "$TAG"))"
  echo "shards $N x OMP_NUM_THREADS=2; load $(uptime | sed 's/.*load/load/')"
  python scripts/eval_prior_value_phrase_dev.py count
} | tee "$LOG"

pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_prior_value_phrase_dev.py run "$K/$N" \
    > "logs/phrase_prior_confirmation.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_prior_value_phrase_dev.py report 2>&1 | tee -a "$LOG"
echo "== phrase-scale one-shot done $(date -Is)" | tee -a "$LOG"
echo "Archive $LOG, the shard logs and $CELLS/*.pkl into $EVID and commit IMMEDIATELY;"
echo "then score the verdict by the frozen rule and append it to $OUT_MD."
