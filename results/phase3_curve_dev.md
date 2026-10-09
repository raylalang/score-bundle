# Phase 3: the curve-from-waveform posterior at scale (DEV, exploratory, no claims)

Batch scaling of results/phase3_curve_proto_dev.md under the development protocol of
eval_phase3_waveform_dev (same tracks, same eligibility + rng(0) subsample); model =
score_bundle.phase3.curve (gradient MAP + full joint Laplace; docs/ddsp_review.md).
Scored on corpus ground-truth frames the fit never sees.  Committed questions (stated
before the run, see the script docstring): (a) curve vs pYIN frames on GT-frame RMSE per
family; (b) curve-level coverage@90 — parameter band vs frames that carry their own
measurement noise (the August estimand gap at curve level).

376 notes, tracks: (1, 1)(vn), (1, 2)(vc), (3, 1)(fl), (3, 2)(cl), (5, 1)(tpt), (6, 1)(sax), (7, 2)(tbn)

| arm | median RMSE (cents) | q90 | median band sd | cov@90 | mean NLL | PD fits |
|---|---|---|---|---|---|---|
| fixed | 3.69 | 11.46 | 0.28 | 0.32 | 213.34 | 98% |
| fitted | 3.16 | 10.61 | 0.30 | 0.36 | 189.59 | 99% |
| pYIN frames | 4.09 | 8.62 | — | — | — | (n=376) |

curve(fixed) − pYIN paired RMSE: median -0.458 cents, curve better on 56% of notes
fitted − fixed paired: RMSE median -0.139 cents (better on 70%), NLL median -3.680

Per family, median GT-frame RMSE (fixed / fitted / pYIN):
- strings (n=114): 4.78 / 4.90 / 3.09
- winds (n=142): 2.63 / 2.26 / 5.15
- brass (n=120): 4.07 / 3.15 / 3.51

wall per note (fixed arm): median 4 s, q90 10 s

## Reading (2026-10-09)

1. **Committed question (a): yes at the median, with the August family
   split intact.** The curve posterior beats the pYIN frames overall
   (paired −0.46 cents, better on 56% of notes) and decisively on winds
   (2.3–2.6 vs 5.2 cents); strings still favour the tracker (4.8–4.9 vs
   3.1), exactly as the scalar study found for the estimator chain. The
   curve's failure tail is heavier than pYIN's (q90 ~11 vs 8.6 cents) —
   when the harmonic model is wrong it is wrong by more.
2. **The prior-fit gate verdict flips at scale.** At n = 3 the fitted
   prior looked mixed (results/phase3_prior_fit_dev.md); at n = 376 it
   beats the fixed prior on 70% of notes (RMSE −0.14 median, NLL −3.7)
   and never needed the eigen-clip more often (99% PD). The
   evidence-fitted prior becomes the DEV default for Phase-3 curve work
   from here (nothing registered is touched); the fixed prototype prior
   stays in the scripts as the documented baseline arm.
3. **Committed question (b): no — coverage 0.32/0.36.** The full-Laplace
   band is a parameter band (median 0.3 cents) and the GT annotation
   frames sit 2–4 cents away: the August estimand/measurement gap
   reproduced at curve level. The band is honest for the latent curve
   (model-true synthetic coverage is nominal, test-pinned); any
   frame-level predictive claim needs the empirically calibrated
   discrepancy floor of results/phase3_integration_dev.md. That floor is
   how the curve channel enters the bundle in the integration re-run.
4. Wall: median 4 s/note (q90 10 s) per arm on CPU — roughly a quarter
   of the Nelder-Mead prototype's per-note cost, full Laplace included,
   which is what makes a 376-note two-arm batch a minutes-scale job
   sharded 8 ways.
