# The prior-value experiment, tuning-scalar follow-up: free per-chain tuning scalar (sd 50 cents) in all arms (protocol v2, hidden note acoustically absent; DEV, exploratory, no claims)

Three cross-note priors over per-note intonation constants at matched marginal
variance, identical likelihood/data/optimizer, boundaries fixed at annotations.
Conditions: the middle eligible note's audio MASKED from the likelihood (maximal
ambiguity) vs clean. Committed hypotheses and falsification rule in the script
docstring (`scripts/eval_prior_value_dev.py`). Scored against the quasi-truth centre.

47 cells (chains), tracks: (1, 1)(vn), (1, 2)(vc), (3, 2)(cl), (7, 2)(tbn); 25 cells have a recurring pitch.

| condition | arm | median abs err (cents) | cov@90 | median sd |
|---|---|---|---|---|
| masked | indep | 6.42 | 0.96 | 14.93 |
| masked | temporal | 6.55 | 0.89 | 12.24 |
| masked | graph | 5.90 | 0.89 | 11.87 |
| clean | indep | 2.35 | 0.91 | 6.87 |
| clean | temporal | 2.24 | 0.91 | 6.45 |
| clean | graph | 2.22 | 0.89 | 6.48 |
| masked | (deadpan reference) | 14.42 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs indep: -0.02 [-0.33, +0.32]  (better on 55%, n=47)
- temporal vs indep: +0.17 [-0.55, +0.91]  (better on 43%, n=47)
- graph vs temporal: -0.19 [-0.89, +0.46]  (better on 55%, n=47)
- graph vs temporal, recurring-pitch cells only: +0.01 [-0.88, +0.91]  (better on 48%, n=25)

## Paired contrasts, clean condition (H2: expect ties)

- graph vs indep: -0.00 [-0.01, +0.00]  (better on 62%, n=47)
- temporal vs indep: -0.01 [-0.02, +0.00]  (better on 60%, n=47)

## Cross-run, masked condition: this run minus `priorval_v2`, per arm (negative = this run better; exactly paired cells)

- indep: -6.85 [-9.92, -3.54]* (better on 77%, n=47); first-run cov@90 0.77
- temporal: -0.90 [-2.05, +0.24]  (better on 60%, n=47); first-run cov@90 0.94
- graph: -0.62 [-1.76, +0.49]  (better on 60%, n=47); first-run cov@90 0.94

`priorval_v2` graph vs indep, masked, same cells: -6.24 [-8.62, -3.81]* (better on 79%, n=47)

wall per fit incl. Laplace: median 149 s

## Reading (2026-10-15)

Protocol v2 (hidden note acoustically absent) with a free per-chain
tuning scalar t ~ N(0, 50²) marginalized into every arm's prior as a
rank-one term, so the independent arm returns the chain's observed
mean rather than zero. Paired against `priorval_v2` (same protocol, no
scalar) on the same 47 cells.

1. **The falsification rule of H1' fires.** With the common mode
   available to every arm, the three arms tie on masked cells: graph vs
   indep −0.02 [−0.33, +0.32] (55%), temporal vs indep +0.17 [−0.55,
   +0.91], graph vs temporal −0.19 [−0.89, +0.46]; the pre-stated
   recurring-pitch stratum is +0.01 [−0.88, +0.91] (48% of 25). Medians
   5.9–6.6 cents for all three.
2. **Attribution, resolved**: the scalar moves the independent arm by
   −6.85* (14.4 → 6.4 cents, better on 77%) and the structured arms by
   −0.9 / −0.6 (ns). The whole structured-vs-independent gain of
   `priorval_v2` (−6.24*) was the transfer of the track's shared tuning
   offset. **On 3–5-note chains, score structure (and plain temporal
   adjacency) carries no measurable note-level intonation information
   beyond tuning bookkeeping.** Reported verbatim, as committed.
3. **Calibration (H4')**: the scalar-augmented independent band covers
   at 0.96 (sd 14.9, conservative: the chain-mean posterior plus the
   full prior spread); the structured bands at 0.89 (sd 11.9–12.2), at
   nominal. The zero-mean independent band of `priorval_v2` (0.77) was
   miscentred, as the scalar run shows by repairing it.
4. **H5'**: clean rows bit-identical to `priorval_tuning` (same code
   path), ties at 2.2–2.4 cents, coverage 0.89–0.91.
5. **What the first span now says.** Positive: a cross-note prior lets
   a note whose audio is absent inherit its chain's tuning offset
   (halving the error against a zero-mean prior), with honest bands.
   Negative: the pitch-proximity coupling that distinguishes the score
   graph from time adjacency adds nothing on these chains, and so does
   adjacency itself once the common mode is shared. Where the
   methodological hypothesis could still hold, untested: longer spans
   with recurring pitches at a distance, the other channels (vibrato,
   dynamics), and the curve level — each with the common-mode term in
   every arm by design, which this experiment shows to be mandatory.
6. Scope: 47 chains, 4 development tracks, intonation constants only,
   boundaries fixed, s = 1 for both structured kernels, 149 s per fit;
   development evidence, no claims.
