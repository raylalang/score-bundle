# Correlated τ noise row (development, exploratory)

2026-09-10. Study B of the exploration week: the failed registered claim
C4 (timing calibration, Phase-2 confirmation) names its own follow-up in
the thesis — alignment error is correlated along score time, and the
diagonal noise row provably cannot represent it. This study measures an
AR(1)-correlated τ-noise block on the development split. **Development
data only; exploratory; no claims; the registered pipeline and the spent
confirmation are untouched.**

- Script: `scripts/eval_corrnoise_tau.py` (6 shards, 149 (track, seed)
  cells, 78 dev tracks × 2 seeds, same cache/masks/system as the
  registered dev evaluation's as-given variant).
- Model: Σ_τ(ρ)_ij = √(d_i d_j)·ρ^|i−j| in note order, d_i the as-given
  LOO-warp variances; ρ profiled per cell on a 7-point grid by the
  OBSERVED block's marginal likelihood at the fitted hyperparameters
  (deploy-legal); ρ = 0 reproduces the diagonal path and is the paired
  baseline. Held-out τ predictions use the full covariance including the
  noise cross-terms — a neighbour's warp error informs mine, which is the
  modelled mechanism.

## Verdict: the named follow-up works, on both axes

| τ metric | correlated | diagonal | paired Δmean (95% CI) |
|---|---|---|---|
| NLL | −1.563 | −1.452 | **−0.111 [−0.142, −0.080]\*** |
| cov@90 | 0.903 | 0.916 | −0.013 [−0.019, −0.007]\* (toward nominal) |
| RMSE (s) | 0.0795 | 0.0987 | **−0.0192 [−0.0261, −0.0132]\*** |

And the diagnosis behind C4 is confirmed as a *detected* property of the
data, not a hypothesis: the evidence chooses ρ > 0 on **97% of cells**
(median ρ = 0.45; histogram 0:5, 0.15:10, 0.3:33, 0.45:47, 0.6:38,
0.75:12, 0.9:4). Alignment error along score time is correlated, the
model can represent it with one extra parameter, and representing it
improves timing recovery by ~20% (≈99 → 80 ms) *and* its calibration —
the exact axis the confirmation failed on.

## Notes

- The profile is at fixed fitted hyperparameters (cheap, staged design);
  a joint refit with ρ inside the evidence is the full version and can
  only do better in-model. Not run here.
- Coverage moves from mild over-coverage toward nominal; the NLL gain is
  therefore genuine sharpening, not variance inflation.
- If this is ever to become a claim, it is a natural candidate for the
  tonal-metric-style path: its own preregistered confirmation on a fresh
  pool (the Phase-2 pool is spent; a corpus decision is pending anyway).
  Nothing is adopted now.
- Reproduce: `PYTHONPATH=src:scripts python scripts/eval_corrnoise_tau.py
  run --shard K/6` (×6) then `report`.
