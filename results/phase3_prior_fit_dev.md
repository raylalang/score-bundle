# Phase 3: evidence-based fit of the curve-prior hyperparameters (DEV, exploratory, no claims)

2026-10-09. `scripts/fit_phase3_prior.py` on 24 notes (up to 4 per track,
7 one-per-instrument dev tracks). Coordinate ascent on the summed
per-note Laplace evidence: MAP refit per round (`curve.fit_note`), then
Adam on the five hyperparameters with each MAP and its likelihood-only
curvature frozen (`curve.loglik_hessian` + `curve.evidence_fixed_map`;
the base point is test-pinned to `CurveFit.log_evidence`).
Hyperparameters SHARED across notes — the spectral-mixture study's
lesson (`results/sm_proper_dev.md`): per-note freedom has nothing to
discover here.

## Result

| | w1 | mu1 (Hz) | v1 | w2 | v2 | sum log evidence |
|---|---|---|---|---|---|---|
| fixed (prototype scales) | 30 | 5.5 | 0.4 | 60 | 0.3 | 1,175,898.6 |
| evidence-fitted | 96.6 | 5.66 | 0.57 | 151.9 | 0.82 | 1,176,615.6 (+717.0) |

Ascent was monotone and converged by round 2 (round deltas +496, +15,
+0.1 nats at frozen MAPs). Reading: the evidence asks for a LOOSER
prior — both component weights up ~3x, both bandwidths roughly doubled —
while the vibrato mean barely moves (5.5 → 5.66 Hz): the physical band
is confirmed, the hand-set amplitude scales were conservative.

## Gate (three prototype notes of `results/phase3_curve_proto_dev.md`, GT-frame RMSE / coverage@90)

| note | fixed | fitted |
|---|---|---|
| (1,1)#31 vn | 2.29 c / 0.14 | 2.42 c / 0.12 |
| (1,2)#0 vc | 4.80 c / 0.12 | 5.16 c / 0.10 |
| (3,2)#96 cl | 1.13 c / 0.59 | 1.05 c / 0.62 |

Mixed at n = 3 (worse on two, better on one; coverage unchanged), so by
the pre-stated gate the fitted prior is NOT adopted as the default: the
fixed prototype prior stays the baseline, and the fitted prior runs as
the second arm of the batch study (`scripts/eval_phase3_curve_dev.py`),
which measures the delta at n ≈ 370 instead of 3.

Fitted values cached in `.cache/phase3_prior_fit.json` (consumed by the
batch eval's `fitted` arm).

NB the low GT-frame coverage on all three notes (either prior) is the
August estimand gap at curve level — the Laplace band is a parameter
band while the GT frames carry their own measurement noise; committed
question (b) of the batch study, measured there, not hidden here.
