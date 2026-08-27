# Phase 3: waveform-likelihood intonation at scale (DEV)

376 notes, tracks: (1, 1)(vn), (1, 2)(vc), (3, 1)(fl), (3, 2)(cl), (5, 1)(tpt), (6, 1)(sax), (7, 2)(tbn)

| variant | median abs err (cents) | q90 | median sd | median abs z | cov@90 |
|---|---|---|---|---|---|
| flat | 2.81 | 8.72 | 0.078 | 36.3 | 0.03 |
| drift | 2.40 | 7.23 | 0.063 | 36.2 | 0.01 |
| ar1 | 2.73 | 8.37 | 0.061 | 45.2 | 0.02 |

estimator |err| median 2.01 (n=373 notes with estimates)
drift-flat paired |err|: median -0.177, drift better on 57% of notes
AR(1) residual rho: median 0.944, q10/q90 0.841/0.980

Per family (median abs err flat / drift / estimator):
- strings (n=114): 2.15 / 2.04 / 1.43
- winds (n=142): 3.53 / 2.75 / 3.27
- brass (n=120): 2.76 / 2.34 / 1.59

## Reading (2026-08-27; three variants, sharded 8 x OMP=2)

**DEV ONLY — EXPLORATORY.** Scaled from the 2-note feasibility demo
(`results/phase3_waveform_feasibility.md`); inference estimator-free
(score-centred grids; the vibrato scaffold on identifiable notes is the
one estimator ingredient, recorded as such).

1. **The waveform likelihood is competitive at scale with no tracker:**
   median 2.4 cents (drift variant) from quasi-truth vs the pyin+NLLS
   chain's 2.0 — and on winds it is *better* (2.75 vs 3.27), where pyin's
   failure modes live. Phase 3's premise survives 376 real notes.
2. **The drift term improves accuracy** (paired median −0.18 cents,
   better on 57% of notes) — the drift-study structure is real in the
   waveform domain — **but does not repair calibration** (coverage@90
   1–3%, median |z| ~36, both white-noise variants): a better mean model
   shrinks the posterior width as fast as the error.
3. **The generic noise-side fix also fails, diagnostically.** AR(1)
   residual whitening (median fitted rho 0.944) leaves coverage at 2%
   and *worsens* |z| (45): the whitener suppresses out-of-band noise,
   but the structured misfit is concentrated IN-BAND, at the harmonics
   themselves (inharmonicity, envelope modulation sidebands, curve
   mismatch) — precisely where a stationary colored-noise floor cannot
   reach and where whitening effectively raises the local SNR.
4. **The sharpened Phase-3 design conclusion:** calibration requires
   residual structure correlated with the model's own directions — a
   marginalized pitch-curve deviation prior (smooth deviations delta(t)
   around the parametric curve, marginalized like the amplitudes, adding
   a J Sigma_dev J^T term along the curve Jacobian), not a richer point
   model (measured: insufficient) and not a generic colored floor
   (measured: insufficient). Phase 2 earned its error bars from
   estimator-supplied variances; Phase 3 must earn them from an explicit
   model of what the parametric curve is not.

No claims; single study, dev only. Next: the deviation prior, then the
joint prior over z across notes.
