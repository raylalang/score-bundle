#!/bin/bash
# =============================================================================
#  THE PHASE-2 CONFIRMATION ONE-SHOT.  READ BEFORE RUNNING.
#
#  This spends the frozen 13-piece URMP confirmation pool — ONCE, per the
#  registered protocol (docs/phase2_prereg_design.md, git tag
#  phase2-registration-2026-08-17). There are no reruns, no added seeds, no
#  post-hoc filters; every number is reported whatever the outcome. Running
#  this is a deliberate decision (Ray + professor), not a routine command.
#
#  It refuses to start unless BOTH are set:
#      export PHASE2_SPLIT=confirmation
#      export PHASE2_CONFIRMATION_I_AM_SURE=yes
#
#  Pipeline (sharded; expect an afternoon, not days): extract -> loudness ->
#  tau -> delta_vib gt/pyin -> delta -> N-sharded eval (3 registered systems)
#  -> report into results/phase2_confirmation_results.md.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${PHASE2_SPLIT:-}" != "confirmation" ] || \
   [ "${PHASE2_CONFIRMATION_I_AM_SURE:-}" != "yes" ]; then
  echo "REFUSED: this is the one-shot confirmation run. Read the banner in"
  echo "scripts/run_phase2_confirmation.sh and set BOTH environment variables"
  echo "only when the spend has been deliberately agreed."
  exit 1
fi

source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src:scripts
N="${1:-8}"
mkdir -p logs results/phase2_cells/conf
rm -f results/phase2_cells/conf/cells.shard*.pkl

echo "== confirmation pipeline start $(date -Is)" | tee logs/phase2_confirmation.log
for STAGE in extract loudness tau; do
  OMP_NUM_THREADS=2 python scripts/eval_phase2_real.py "$STAGE" \
    2>&1 | tee -a logs/phase2_confirmation.log
done
for STAGE in gt pyin; do
  OMP_NUM_THREADS=1 python scripts/eval_delta_vib.py "$STAGE" \
    2>&1 | tee -a logs/phase2_confirmation.log
done
OMP_NUM_THREADS=2 python scripts/eval_phase2_real.py delta \
  2>&1 | tee -a logs/phase2_confirmation.log

pids=()
for K in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=2 python scripts/eval_phase2_real.py run "$K/$N" \
    > "logs/phase2_confirmation.shard${K}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/eval_phase2_real.py report 2>&1 | tee -a logs/phase2_confirmation.log
echo "== confirmation pipeline done $(date -Is)" | tee -a logs/phase2_confirmation.log
echo "Archive logs + results into evidence/ and commit IMMEDIATELY (one-shot"
echo "evidence goes into version control the moment it exists)."
