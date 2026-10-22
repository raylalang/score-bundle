# The prior-value experiment: score structure in direct waveform inference (protocol v2, hidden note acoustically absent; DEV, exploratory, no claims)

Three cross-note priors over per-note intonation constants at matched marginal
variance, identical likelihood/data/optimizer, boundaries fixed at annotations.
Conditions: the middle eligible note's audio MASKED from the likelihood (maximal
ambiguity) vs clean. Committed hypotheses and falsification rule in the script
docstring (`scripts/eval_prior_value_dev.py`). Scored against the reference centre.

47 cells (chains), tracks: (1, 1)(vn), (1, 2)(vc), (3, 2)(cl), (7, 2)(tbn); 25 cells have a recurring pitch.

| condition | arm | median abs err (cents) | cov@90 | median sd |
|---|---|---|---|---|
| masked | indep | 14.42 | 0.77 | 12.71 |
| masked | temporal | 6.80 | 0.94 | 11.43 |
| masked | graph | 7.62 | 0.94 | 11.24 |
| clean | indep | 2.37 | 0.91 | 6.56 |
| clean | temporal | 2.30 | 0.89 | 6.31 |
| clean | graph | 2.28 | 0.89 | 6.32 |
| masked | (deadpan reference) | 14.42 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs indep: -6.24 [-8.49, -3.79]* (better on 79%, n=47)
- temporal vs indep: -5.78 [-8.16, -3.27]* (better on 79%, n=47)
- graph vs temporal: -0.46 [-1.48, +0.51]  (better on 53%, n=47)
- graph vs temporal, recurring-pitch cells only: -0.86 [-2.04, +0.34]  (better on 56%, n=25)

## Paired contrasts, clean condition (H2: expect ties)

- graph vs indep: -0.00 [-0.02, +0.01]  (better on 53%, n=47)
- temporal vs indep: -0.00 [-0.02, +0.01]  (better on 57%, n=47)

## Cross-run, masked condition: this run minus `priorval`, per arm (negative = this run better; exactly paired cells)

- indep: -15.00 [-22.53, -8.29]* (better on 64%, n=47); first-run cov@90 0.38
- temporal: -16.39 [-23.50, -10.46]* (better on 79%, n=47); first-run cov@90 0.45
- graph: -16.61 [-24.05, -10.47]* (better on 79%, n=47); first-run cov@90 0.49

`priorval` graph vs indep, masked, same cells: -4.63 [-6.43, -2.96]* (better on 85%, n=47)

wall per fit incl. Laplace: median 151 s

## Reading (2026-10-15)

Protocol v2: the hidden note's design columns are removed from the
likelihood, so its posterior is the prior conditional given the
observed notes (the first run's masked condition was leakage-driven,
see the erratum in `results/prior_value_dev.md`).

1. **The protocol is now what it claims to be.** The independent arm
   returns exactly the prior mean (median error = deadpan 14.42 cents,
   median sd 12.71 = the prior sd), and the clean rows are bit-identical
   to the first run. Every arm improves by 15–17 cents paired against
   the leaky run (all three significant): the first run's masked numbers
   were noise, in every arm.
2. **H1' holds without the tuning scalar, and by a wide margin**: the
   structured priors halve the hidden note's error, 14.4 → 6.8
   (temporal) / 7.6 (graph) cents median; graph vs indep −6.24
   [−8.49, −3.79], temporal vs indep −5.78 [−8.16, −3.27], both better
   on 79% of 47 chains. This contrast INCLUDES tuning-offset transfer
   (a zero-mean independent prior cannot learn the track's offset; the
   structured priors transfer it from the neighbours), so it is the
   "any structure vs none" answer; the note-level attribution is the
   scalar run's question (`results/prior_value_v2_tuning_dev.md`).
3. **Score graph vs temporal chain: a tie** (−0.46 [−1.48, +0.51]);
   the pre-stated recurring-pitch stratum shows a trend in the predicted
   direction (−0.86 [−2.04, +0.34], 56% of 25) that does not reach
   significance at n = 25. The first run's starred stratum result was
   an artefact; this is the honest version.
4. **H4' (prior calibration)**: the structured conditional bands cover
   at 0.94 (nominal 0.90; sd 11.2–11.4 cents, conservative by a hair);
   the independent prior's zero-mean band covers at 0.77 — the
   zero-mean prior IS miscentred on tuning-offset tracks, which the
   scalar run addresses. The clean bands stay at 0.89–0.91.
5. **H5' holds** (clean rows tie to the second decimal; bit-identical
   to the first run, as the code path is unchanged there).
6. Scope unchanged: 47 chains, 4 development tracks, intonation
   constants only, boundaries fixed, s = 1, ~151 s per fit; development
   evidence, any claim needs its own registration.
