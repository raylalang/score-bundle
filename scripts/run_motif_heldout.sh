#!/usr/bin/env bash
# =============================================================================
#  THE MOTIF-STUDY ONE-SHOT ON HELD-OUT PERFORMANCES (registered;
#  docs/motif_prereg_design.md).  READ BEFORE RUNNING.
#
#  Runs scripts/eval_motif_pilot_dev.py ONCE with --perf-source heldout: for
#  every development piece of the motif study, the aligned ASAP performances
#  OTHER than the first (the development one), capped at four, with the
#  annotated occurrences, masks and arms of the registration.  The pieces are
#  the development pieces (disclosed); the performances, hence the targets,
#  have never been read by any run.  No reruns, no added seeds or performances,
#  no post-hoc filters; every number reported.
#
#  Refuses unless MOTIF_HELDOUT_I_AM_SURE=yes, refuses if the results file or
#  the evidence directory exists, and refuses if a frozen file differs from the
#  registration tag.      Usage: bash scripts/run_motif_heldout.sh [N=14]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-14}"
TAG=motif-registration-2026-11-20
OUT=results/motif_heldout
OUT_MD=results/motif_heldout_results.md
EVID=evidence/motif_heldout
ARMS=time,time_motif,time_motif_pc
if [ "${MOTIF_HELDOUT_I_AM_SURE:-}" != "yes" ]; then
  echo "REFUSED: this runs the registered motif one-shot on held-out performances."
  echo "Read the banner in scripts/run_motif_heldout.sh and set MOTIF_HELDOUT_I_AM_SURE=yes"
  echo "only when the spend has been deliberately agreed."
  exit 1
fi
if [ -e "$OUT_MD" ] || [ -e "$EVID" ] || [ -e "$OUT/pieces.pkl" ]; then
  echo "REFUSED: the motif one-shot was already run once ($OUT_MD, $EVID or $OUT exists)."
  exit 1
fi
FROZEN="docs/motif_prereg_design.md scripts/eval_motif_pilot_dev.py scripts/eval_graphgp.py \
scripts/run_motif_heldout.sh src/score_bundle/gp.py src/score_bundle/graph.py \
src/score_bundle/baselines.py src/score_bundle/features.py src/score_bundle/downstream.py"
if ! git diff --quiet "$TAG" -- $FROZEN || [ -n "$(git status --porcelain -- $FROZEN)" ]; then
  echo "REFUSED: frozen files differ from tag $TAG"; exit 1
fi
PY=/home/ray/miniconda3/envs/score-bundle/bin/python
export OMP_NUM_THREADS=2 PYTHONPATH=src:scripts
COMMON="--out $OUT --perf-source heldout --max-perf 4 --arms $ARMS --schemes occ --seeds 4"
mkdir -p logs "$OUT"
LOG=logs/motif_heldout.log
{
  echo "== motif held-out one-shot start $(date -Is)"
  echo "code state $(git rev-parse HEAD) = tag $TAG ($(git rev-parse "$TAG"))"
  echo "shards $N x OMP_NUM_THREADS=2; load $(uptime | sed 's/.*load/load/')"
  $PY scripts/eval_motif_pilot_dev.py --stage align $COMMON
} | tee "$LOG"
pids=()
for K in $(seq 0 $((N - 1))); do
  $PY scripts/eval_motif_pilot_dev.py --stage run $COMMON --shard "$K/$N" \
    > "logs/motif_heldout.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY scripts/eval_motif_pilot_dev.py --stage report $COMMON 2>&1 | tee -a "$LOG"
echo "== motif held-out one-shot done $(date -Is)" | tee -a "$LOG"
echo "Archive $LOG, the shard logs, $OUT/align.md, $OUT/report.md, $OUT/effects.json and"
echo "$OUT/cells.shard*.pkl into $EVID and commit IMMEDIATELY; then score the verdict by"
echo "the frozen rule and write $OUT_MD."
