#!/bin/bash
# Mean+smoothing (best: feat+LM mean, chord+VL graph) at the four non-anchor
# masking rates, on the validation pieces, using the existing per-rate
# leak-free masks/means (.cache/masksweep_inputs_obsX.pkl).  Validation only.
set -euo pipefail
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh
conda activate score-bundle
export PYTHONPATH=src
mkdir -p logs
for OF in 0.50 0.70 0.80 0.90; do
  OMP_NUM_THREADS=8 python scripts/eval_kernels.py --stage run \
    --kernels harmonic_vl --inputs .cache/masksweep_inputs_obs$OF.pkl \
    --out-dir results/kernels_ms_obs$OF \
    > logs/ms_baseline_obs$OF.log 2>&1
  echo "[baseline] obs$OF done $(date)"
done
echo "MULTIRATE BASELINE DONE"
