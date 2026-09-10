# Extrapolation-safe completion: coverage test + disagreement guard (development)

2026-09-10. Study C of the exploration week: the measured adaptation
boundary (per-piece feature weights explode when fit on an opening
excerpt; draft sec:boundary) and Future Work's named fix (a Mahalanobis
coverage test with fallback to the cross-piece head), measured — and
refined, because the named fix alone turned out to be half the answer.
**Development pieces only (ev[:30]); exploratory; no claims.**

Process disclosure: the first launch of this script iterated the arrays
cache's full eval list, which continues into the 20 confirmation pieces.
Those runs crashed on a metrics-key bug before any number was printed or
persisted, so nothing derived from confirmation pieces was ever observed
or recorded; the script now slices to the dev 30 explicitly.

- Script: `scripts/eval_completion_fallback.py`; prefix and random masks
  at observed fractions 0.25/0.5/0.75; proposed model (GP-featlm) and
  cross-piece head (μ_LM with the head pieces' residual SD) as brackets.
- Rules under test, all deploy-legal:
  (i) **Mahalanobis coverage** in the observed excerpt's low-rank PCA
  frame (r ≤ 10 comps at 95% var; χ²_q in-subspace + off-subspace energy
  vs the observed q-quantile). A raw high-dimensional Mahalanobis test
  degenerates (flags 100% everywhere) — the low-rank frame is required.
  (ii) **Disagreement guard**: fall back per (note, channel) where
  |m_GP − μ_LM| > 3·head_SD.
  (iii) Either.

## Verdict: the combination closes the boundary

RMSE mean / median / worst per piece-set (30 pieces):

| setting | GP-featlm | mahal q=.99 | guard 3sd | **either q=.99** | head only |
|---|---|---|---|---|---|
| prefix .25 | 16009 / 0.478 / 480266 | 6253 / 0.413 / 187565 | 0.482 / 0.468 / 0.9 | **0.423 / 0.413 / 0.6** | 0.414 / 0.421 / 0.6 |
| prefix .50 | 0.432 / 0.426 / 0.7 | 0.413 / 0.420 / 0.6 | 0.426 / 0.419 / 0.6 | **0.413 / 0.420 / 0.6** | 0.411 / 0.408 / 0.6 |
| prefix .75 | 0.380 / 0.351 / 0.7 | 0.379 / 0.352 / 0.7 | 0.380 / 0.351 / 0.7 | **0.379 / 0.352 / 0.7** | 0.410 / 0.395 / 0.7 |
| random .25 | 0.382 / 0.390 / 0.5 | 0.382 / 0.388 / 0.5 | 0.383 / 0.381 / 0.5 | **0.383 / 0.388 / 0.5** | 0.432 / 0.425 / 0.6 |
| random .50 | 54.3 / 0.364 / 1618 | 44.5 / 0.368 / 1325 | 0.372 / 0.364 / 0.6 | **0.374 / 0.368 / 0.6** | 0.431 / 0.423 / 0.7 |
| random .75 | 0.326 / 0.332 / 0.5 | 0.349 / 0.344 / 0.6 | 0.348 / 0.351 / 0.6 | **0.353 / 0.345 / 0.6** | 0.428 / 0.415 / 0.7 |

Findings, in order of importance:

1. **The combined rule is safe everywhere and gives almost nothing away.**
   Worst cell ≤ 0.7 at every setting (from 480,266); head-level under
   heavy extrapolation; the GP's genuine advantage retained where
   adaptation wins (prefix .75, random) up to a small cost at random .75
   (0.353 vs 0.326, the Mahalanobis rule's ~4% false flags).
2. **The two rules catch different failures.** The coverage test sees
   distributional extrapolation (halves the prefix-.25 damage, flags 50%
   there, 3–4% on interpolation — the control behaves) but misses
   catastrophic notes lying inside the excerpt's feature coverage. The
   disagreement guard is the catastrophe-killer (every worst tamed,
   ~0.1% healthy flags) but ignores the broad degradation. Future Work's
   "a Mahalanobis coverage test would do" should be amended: it half-does.
3. **Adaptation blow-ups are not exclusively an extrapolation
   phenomenon.** A 1,618-RMSE cell sat in *random 50% interpolation*;
   only the guard catches it. The boundary is about feature-weight
   determination, and thin coverage can happen anywhere.
4. Coverage under the combined rule sits at 0.90–0.93 everywhere
   (GP-alone drops to 0.84 at prefix .25).

Adoption would be a deploy-mode change and is not made here; the rule is
recorded for the Future Work section's update.

- Reproduce: `PYTHONPATH=src:scripts python
  scripts/eval_completion_fallback.py --fracs F` for F in 0.25/0.5/0.75.
