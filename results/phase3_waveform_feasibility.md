# Phase 3: first waveform-likelihood inference on real audio

> Superseded in scope by the 376-note study
> `results/phase3_waveform_dev.md` (same day), which scales this
> pilot, tests three calibration fixes, and closes the diagnosis
> (estimand gap). This file remains the record of the pilot.

**DEV ONLY — EXPLORATORY / FEASIBILITY.** 2026-08-27. Script:
`scripts/demo_phase3_waveform.py` (URMP dev track (1,1), violin; reads
the dev caches + audio; no tracker in the inference loop). New machinery:
`waveform_model.collapsed_loglik_lowrank` (Woodbury form of the exact
amplitude-marginalized likelihood, O(m p²); equality with the dense form
unit-pinned in `tests/test_phase3.py`).

## What was run

The Phase-3 claim in miniature: **the audio itself is the observation.**
The note's f0 curve is parameterized by the Phase-2 channel variables
(eq:cents-curve / eq:vibrato), the harmonic design matrix Φ(z) carries
phase coherence, harmonic amplitudes (piecewise-constant over the four
eq:loudness chunks × 8 harmonics in quadrature) are marginalized exactly,
and the collapsed likelihood is gridded over intonation c (coarse
0.5-cent pass ± 30 cents, fine 0.02-cent refinement) and, on the vibrato
note, over rate f at the c optimum. Grid inference is exact up to
discretization in 1-D — no Laplace machinery claimed. Audio at 16 kHz;
noise floor fit from the residual of an amplitude regression at the
estimator's curve.

## Results (two notes, track (1,1))

| | steady note 27 (0.41 s) | vibrato note 75 (1.49 s) |
|---|---|---|
| waveform posterior c | +5.95 ± 0.17 cents | +16.46 ± 0.04 cents |
| Phase-2 estimator c (tracked frames) | +8.64 ± 1.44 | +16.30 ± 0.39 |
| quasi-truth c (GT curve) | +7.53 | +15.73 |
| \|waveform − GT\| | 1.58 cents | 0.73 cents |
| \|estimator − GT\| | 1.12 cents | 0.58 cents |
| waveform posterior f_vib | — | 4.82 ± <0.005 Hz (est 4.92, GT 4.93) |
| model-residual SNR | 4.4 dB | 2.2 dB |

## Reading

1. **Feasibility: yes.** With no tracker anywhere in the loop, the
   collapsed waveform likelihood localizes intonation to within 0.7–1.6
   cents of the ground truth on real recorded violin — on par with the
   dedicated pyin + NLLS estimator chain. The phase-coherent basis is
   extraordinarily informative: one cent of centre error accumulates a
   quarter-cycle of phase drift per second at 440 Hz, and the likelihood
   sees all of it.
2. **The honest failure mode: overconfidence under model mismatch.** The
   posterior widths (±0.04–0.17 cents; rate width below the 0.002 Hz
   grid) are far smaller than the actual distances to GT (0.6–1.6 cents;
   0.11 Hz on rate) — 4–18σ. Cause: the collapsed likelihood conditions
   on the parametric curve being *exactly* right, so with 6k–23k sample
   observations every deviation between the true pitch curve and the
   constant-parameter model is either absorbed into the white-noise floor
   (deflating precision honestly) or, where it correlates with the
   c-direction, biases the peak with high confidence. This is the
   waveform-domain face of the drift-study finding
   (`results/phase2_drift_dev.md`): within-note time variation is real,
   and a likelihood this powerful cannot ignore it and stay calibrated.
3. **Consequence for the Phase-3 design (the research agenda, now with a
   measured basis):** the position model z must carry within-note
   structure (at minimum the linear drift term; properly a smooth pitch
   curve prior), and/or the noise model must be robust to structured
   residual — otherwise Phase 3 trades Phase 2's honest error bars for
   sharper wrong ones. Calibration-first, again.
4. The low model-residual SNR (2–4 dB) is expected: "noise" here is
   everything 8 harmonics × 4 chunks cannot represent (bow noise,
   inharmonicity, reverb, envelope detail within chunks) — the model
   absorbs it as σ², which is what keeps the posterior from being even
   sharper.

## Not claimed

No claims registered; single track, two notes, one instrument; the
outer-loop inference over all notes jointly (GP prior over z coupling
notes — the actual Phase-3 model) remains the open work. Next steps in
order: drift term in the curve model (removes the known mismatch), more
notes/instruments, then the joint prior.
