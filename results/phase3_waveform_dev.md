# Phase 3: waveform-likelihood intonation at scale

**DEV ONLY — EXPLORATORY. No claims registered.** Study run 2026-08-27,
curated 2026-08-28. Thesis record: §phase3-math, subsection "The measured
constraint" (tab:phase3-study, fig:phase3-study). Companion pilot:
`results/phase3_waveform_feasibility.md` (the 2-note study that seeded
this one).

## Question

Does the raw waveform carry per-note expressive information directly —
no tracker in the loop — and is the resulting posterior honest? Four
position models test the second half, after the feasibility pilot showed
overconfidence under the constant-parameter curve.

## Data structure

- **One observation = one note's audio segment:** m samples at 16 kHz
  (48 kHz stems polyphase-resampled), cut at the annotated note
  boundaries with 20 ms guards, capped at 2 s. m ranges ~2,000–31,000.
- **Corpus slice:** 376 notes from 7 dev URMP tracks, one per instrument
  — vn, vc (strings); fl, cl, sax (winds); tpt, tbn (brass) — up to 60
  eligible notes per track (eligibility: quasi-truth centre exists,
  |c_gt| ≤ 150 cents, duration ≥ 0.25 s; rng(0) subsample).
- **Design matrix per note:** harmonic quadrature basis (K = 8 harmonics
  → cos/sin columns) × 4 equal time chunks (the eq:loudness
  segmentation, absorbing envelope at chunk resolution) = p = 64
  amplitude columns; amplitudes marginalized exactly under a broad
  N(0, 10·I) prior via the O(mp²) Woodbury form
  (`waveform_model.collapsed_loglik_lowrank`, dense-equality
  unit-pinned in `tests/test_phase3.py`).
- **Per-note record:** for each variant, (posterior mean, posterior sd,
  profiled slope, fitted rho) + gt_c (quasi-truth centre), est_c/est_sd
  (Phase-2 estimator, comparison only), instrument, duration, n_frames.

## Method

f0 curve parameterized by the Phase-2 channel variables
(eq:cents-curve/eq:vibrato); on vibrato-identifiable notes the
extent/rate/delay scaffold comes from the Phase-2 estimator — the one
estimator-supplied ingredient, held fixed across variants. Inference over
the intonation centre c is by grid, centred on the SCORE-NOMINAL pitch
(not the estimator): coarse 1-cent pass over ±50 cents, 0.03-cent
refinement around the optimum — exact 1-D inference up to discretization.
Noise variance = residual variance of the amplitude LS fit, refit at the
coarse optimum. Variants:

1. **flat** — constant c (the pilot's model).
2. **+ AR(1) noise** — per-note residual lag-1 correlation fit, audio
   and design columns whitened by it (Jacobian constant across the grid,
   so posterior comparisons stand).
3. **+ drift** — the linear within-note drift term of
   `results/phase2_drift_dev.md`; slope profiled (9-point coarse grid
   ± 40 cents/s, 13-point refinement).
4. **+ deviation prior** — 8 zero-mean Hann-bump pitch deviations,
   finite-differenced waveform Jacobian at the per-candidate LS
   amplitudes appended to Φ, prior sd 5 cents/coefficient, marginalized
   exactly like the amplitudes (zero-mean so a constant shift stays
   identified as c). Coarse location by the drift search; reported
   posterior conditional on the profiled slope.

**Scoring:** vs the quasi-truth centre (Phase-2 NLLS estimator on URMP's
ground-truth curve): absolute error, z = (mean − gt)/sd, coverage@90
(|z| ≤ 1.645). **Self-check:** 40 synthetic notes built from real notes'
fitted structure (curve at gt_c, LS amplitudes, fitted noise) — the model
true by construction — through the same flat pipeline.

## Results (n = 376)

| position model | median \|err\| (cents) | q90 | median sd | median \|z\| | cov@90 |
|---|---|---|---|---|---|
| flat | 2.81 | 8.72 | 0.078 | 36.3 | 0.03 |
| + AR(1) noise | 2.73 | 8.37 | 0.061 | 45.2 | 0.02 |
| + drift | 2.40 | 7.23 | 0.063 | 36.2 | 0.01 |
| + deviation prior | **2.29** | **6.77** | 0.063 | 31.7 | 0.03 |
| pyin+NLLS estimator chain | 2.01 | — | (own σ) | — | — |
| **model-true synthetics (self-check, n=40)** | — | — | — | **0.53** | **0.90** |

Supporting statistics:
- drift−flat paired |err|: median −0.177 cents, drift better on 57% of
  notes — the drift-study structure transfers to the waveform domain.
- AR(1) residual rho: median 0.944 (q10/q90 0.841/0.980).
- Self-check: mean z −0.19 (no bias), coverage 0.90 exactly.

Per family, median |err| (flat / drift / dev-prior / estimator):
- strings (n=114): 2.15 / 2.04 / 2.04 / 1.43
- winds (n=142): 3.53 / 2.75 / **2.67** / 3.27 ← waveform BEATS the estimator
- brass (n=120): 2.76 / 2.34 / 2.04 / 1.59

## Synthesis

1. **The waveform carries the information.** Median 2.29 cents from
   quasi-truth with no tracker, approaching the estimator chain (2.01)
   and beating it on winds — where pyin's failure modes live.
2. **Accuracy climbs with every model refinement; coverage does not
   move.** A better mean model shrinks the posterior width as fast as
   the error. The AR(1) floor fails diagnostically: the structured
   misfit is concentrated IN-BAND at the harmonics (inharmonicity,
   envelope sidebands, curve mismatch), where a stationary colored floor
   cannot reach — whitening effectively raises the local SNR and makes
   |z| worse.
3. **The machinery is internally calibrated** (self-check covers at
   0.90 exactly). Therefore the invariant overconfidence is an
   **estimand gap**: the constant the harmonic model estimates and the
   constant the NLLS estimator defines on the GT curve are different
   functionals of the same performance, ~2 cents apart. No within-model
   refinement can close a between-question gap — which is why all three
   didn't.
4. **Design resolution (in the thesis):** the waveform posterior enters
   the Phase-2 bundle the way every estimator does — a measurement whose
   noise row is its posterior variance plus an empirically calibrated
   per-note discrepancy floor at the measured ~2-cent scale — restoring
   calibration by construction under the as-given discipline.
   Waveform-native calibration (a bridge model for the estimand gap) is
   the research frontier beyond that.

## Reproduce

```bash
# main study (8 shards, ~35 min) + deviation prior (10 shards, ~30 min)
for K in $(seq 0 7);  do OMP_NUM_THREADS=2 PYTHONPATH=src:scripts \
  python scripts/eval_phase3_waveform_dev.py run    $K/8  & done; wait
for K in $(seq 0 9);  do OMP_NUM_THREADS=2 PYTHONPATH=src:scripts \
  python scripts/eval_phase3_waveform_dev.py rundev $K/10 & done; wait
PYTHONPATH=src:scripts python scripts/eval_phase3_waveform_dev.py report
PYTHONPATH=src:scripts python scripts/eval_phase3_selfcheck.py
PYTHONPATH=src:scripts python scripts/make_phase3_study.py   # figure
```

NB `report` regenerates the raw tables from the shard pickles; this file
is the curated record (tables verified identical 2026-08-28).
