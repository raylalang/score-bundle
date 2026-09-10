# Deploy-time heavy-tailed timing predictive (development, exploratory)

2026-09-10. Study A of the exploration week: the cheap remedy for the one
measured Phase-1 failure mode (draft sec:guard-gp — the Gaussian log
score's quadratic tail amplifying a handful of timing outliers on a very
steady piece into the pooled NLL). **Development data only; entirely
offline from the published per-note predictions (no refit, means and RMSE
untouched by construction); exploratory; no claims. The confirmation set
was not rescored — its one-shot discipline covers evaluations, not only
fits.**

- Script: `scripts/eval_tail_predictive.py` (offline, seconds).
- Cells: the proposed model (`b_featlm`) — published dev run (40% hidden)
  plus the masking-level sweep (50/30/20/10%), 120 cells each.
- Variants (τ channel only, Gaussian elsewhere): variance-matched
  Student-t predictive (ν = 3, 5, 10); per-cell predictive-std floors
  (β·median s, β = 0.5, 1.0); the t₅+floor combination.

## Verdict

**The Student-t predictive defuses the tail completely and costs
nothing.** The documented blow-up cell (piece 28, seed 2, 30% hidden):
Gaussian cell NLL **105.16 → −1.69** under t₅ (coverage 0.95 → 0.94).
Worst-cell τ NLL across masking levels collapses from
10.0 / 105.2 / 9.1 / 18.6 (50/30/20/10%) to ≤ 0.7 under every t variant,
and pooled NLL *improves on healthy cells too* (40%: −0.82 → −1.41 under
t₅) with PIT-KS slightly better — consistent with the calibration
profile's finding that the Gaussian predictive is mildly mis-tailed
against the realized errors. Coverage at the variant's own nominal 90%
stays 0.93–0.95 throughout (t₃ trims the mild over-coverage most).

**The predictive-std floor — the other remedy Future Work names — is
measured ineffective in this form.** The blow-up cell's predictive stds
are not small (its coverage is 0.95); the damage is a few notes many
standard deviations out, which a floor at the cell median cannot reach
(floor rows ≈ identical to Gaussian everywhere). The honest reading: the
failure mode is a tail-shape problem, not a scale problem, so only the
tail-shape fix works.

## Numbers (τ channel, per-cell NLL pooled / median / worst; cov@90; PIT-KS)

| set | gauss | t₃ | t₅ | t₁₀ | floor₁₀₀ |
|---|---|---|---|---|---|
| 40% (published dev) | −0.82 / −0.85 / 2.86 · 0.947 · 0.194 | −1.53 / −1.49 / 0.32 · 0.930 · 0.133 | −1.41 / −1.36 / 0.54 · 0.943 · 0.169 | −1.31 / −1.24 / 0.66 · 0.946 · 0.184 | −0.83 / −0.85 / 2.86 · 0.948 · 0.195 |
| 50% | −0.65 / −0.86 / 10.03 | −1.53 / −1.52 / 0.33 | −1.40 / −1.37 / 0.52 | −1.31 / −1.21 / 0.67 | −0.66 / −0.86 / 9.85 |
| 30% | −0.01 / −0.92 / 105.16 | −1.54 / −1.52 / 0.33 | −1.41 / −1.38 / 0.53 | −1.31 / −1.26 / 0.67 | −0.02 / −0.92 / 104.50 |
| 20% | −0.88 / −0.93 / 9.10 | −1.55 / −1.51 / 0.33 | −1.42 / −1.36 / 0.55 | −1.33 / −1.24 / 0.72 | −0.88 / −0.93 / 8.90 |
| 10% | −0.81 / −1.06 / 18.62 | −1.55 / −1.52 / 0.33 | −1.42 / −1.37 / 0.55 | −1.32 / −1.28 / 0.72 | −0.84 / −1.05 / 15.66 |

## Scope, stated

This is a change of the deploy-time scoring distribution, not of the
inference: the fitted GP, its posterior means, its intervals' centres,
and every RMSE are bit-identical to the published run. A full Student-t
*observation model* for τ (non-conjugate, approximate inference) remains
the inference-level version and remains future work; this study shows the
predictive-level version already removes the entire measured symptom. Any
adoption into reported protocols would be a scoring-rule change requiring
its own registration; the published tables stand as they are.

- Recommendation if ever adopted: ν = 5 (worst cell ≤ 0.55, coverage
  0.94–0.95, no ν-tuning knife-edge — ν = 3 and 10 behave the same way).
- Reproduce: `PYTHONPATH=src:scripts python scripts/eval_tail_predictive.py`.
