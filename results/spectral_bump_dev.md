# A learned spectral filter on the graph spectrum (development, exploratory)

2026-09-10. Study E of the exploration week, closing the "we only used
basic kernels" question definitively: the kernel comparison had measured
*fixed* classical profiles as ties; this study lets the evidence LEARN a
filter and asks whether it ever wants one. **Development pieces only;
same masks as every published number; exploratory; no claims.**

- Script: `scripts/eval_spectral_bump.py` (3 shards, 120 cells =
  30 dev pieces × 4 seeds, config b_feat).
- Model: additive profile plus one free Gaussian bump on the graph
  spectrum, g(ν) ∝ 1/(1+sν) + a·exp(−(ν−m)²/2w²), with (a, m, w)
  optimized jointly with all other hyperparameters by the per-piece
  evidence. a → 0 recovers the additive baseline exactly (nested).
- Paired baseline: the published b_feat cells (results/graphgp_v2),
  identical masks.

## Verdict: the evidence does not want a filter

- **The bump is switched off on 90% of cells** (fitted amplitude:
  median 0.0000, q90 = 0.006).
- Where the extra freedom wanders, it mildly **hurts**: paired
  RMSE +0.0091 [+0.0016, +0.0231]\* (median +0.0006), NLL +0.0150
  [−0.0014, +0.0316] ns — the classical signature of evidence
  overfitting from unneeded flexibility, not of missing capacity.

Together with the fixed-profile ties (kernel comparison) and the
spectral-overlay diagnosis (the families differ only where the evidence
is insensitive), the conclusion now holds against the learned case too:
**on this graph, the leverage is in $L_G$ — which notes are connected and
under what metric — not in the spectral profile.** The "basic kernels"
door is closed, with a measurement rather than an assumption.

- Reproduce: `PYTHONPATH=src:scripts python scripts/eval_spectral_bump.py
  run --shard K/3` (×3) then `report`.
