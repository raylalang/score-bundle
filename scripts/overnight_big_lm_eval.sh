#!/usr/bin/env bash
# Wait for the bigger-LM training (PID $1) to finish, then run the held-out ASAP eval on its
# checkpoint and plot its training curve. Chained so the ablation completes unattended.
set -u
TRAIN_PID="${1:?usage: overnight_big_lm_eval.sh <train_pid>}"
cd /home/ray/Research/score-bundle
source /home/ray/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate score-bundle 2>/dev/null

echo "[orchestrator] waiting for big-LM training pid $TRAIN_PID ..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
echo "[orchestrator] training exited; checkpoint:"
ls -la checkpoints/maestro_big/best.pt || { echo "no best.pt — aborting"; exit 1; }

# training curve for the bigger model
python scripts/plot_training_curve.py \
  --log checkpoints/maestro_big/train.log \
  --out-json checkpoints/maestro_big/history.json \
  --out-fig figures/lm_training_curve_big.png \
  --title "MAESTRO LM big (d=512, L=8, 25.6M params)"

# held-out ASAP eval with the bigger LM's embeddings (same protocol as the small LM)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -u scripts/eval_asap_calibration.py \
  --asap-root ../data/asap-dataset --maestro-root ../data/maestro-v3.0.0 \
  --checkpoint checkpoints/maestro_big/best.pt \
  > logs/eval_big.log 2>&1
echo "[orchestrator] big-LM eval done -> logs/eval_big.log"
