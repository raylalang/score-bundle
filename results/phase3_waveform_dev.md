# Phase 3: waveform-likelihood intonation at scale (DEV)

376 notes, tracks: (1, 1)(vn), (1, 2)(vc), (3, 1)(fl), (3, 2)(cl), (5, 1)(tpt), (6, 1)(sax), (7, 2)(tbn)

| variant | median abs err (cents) | q90 | median sd | median abs z | cov@90 |
|---|---|---|---|---|---|
| flat | 2.81 | 8.72 | 0.078 | 36.3 | 0.03 |
| drift | 2.40 | 7.23 | 0.063 | 36.2 | 0.01 |

estimator |err| median 2.01 (n=373 notes with estimates)
drift-flat paired |err|: median -0.177, drift better on 57% of notes

Per family (median abs err flat / drift / estimator):
- strings (n=114): 2.15 / 2.04 / 1.43
- winds (n=142): 3.53 / 2.75 / 3.27
- brass (n=120): 2.76 / 2.34 / 1.59

## Reading (2026-08-27; sharded run ~35 min, 8 x OMP=2; script this file's header)

**DEV ONLY — EXPLORATORY.** Scaled from the 2-note feasibility demo
(`results/phase3_waveform_feasibility.md`) to 376 notes, 7 tracks, all
three families; inference estimator-free (score-centred grids; the only
estimator ingredient is the vibrato scaffold on identifiable notes,
recorded as such).

1. **The waveform likelihood is competitive at scale with no tracker:**
   median 2.4 cents from quasi-truth vs the pyin+NLLS chain's 2.0 — and
   on winds it is *better* (2.75 vs 3.27), presumably where pyin's
   octave/breathiness failure modes live. Phase 3's premise survives
   contact with 376 real notes.
2. **The drift term improves accuracy** (paired median −0.18 cents,
   better on 57% of notes; winds −0.8) — the drift-study structure is
   real in the waveform domain too.
3. **The drift term does NOT repair calibration.** Median |z| ~36 and
   coverage@90 of 1–3% for BOTH variants: improving the mean model
   shrinks the posterior width (0.078 → 0.063 cents) as fast as the
   error, so overconfidence is invariant to mean-side fixes. The
   diagnosis of the feasibility note sharpens into a design theorem for
   Phase 3: with 10^4-sample likelihoods, calibration must come from the
   NOISE side — a residual model that carries the structured misfit
   (colored noise / inflated marginal / explicit pitch-curve prior with
   marginalized deviations) — not from enriching the point
   parameterization. Phase 2's honest error bars came from
   estimator-supplied variances; Phase 3 has to earn its own the same
   way.

No claims; next steps in order: structured-residual noise model, then
the joint prior over z across notes.
