# The prior-value experiment: score structure in direct waveform inference (DEV, exploratory, no claims)

Three cross-note priors over per-note intonation constants at matched marginal
variance, identical likelihood/data/optimizer, boundaries fixed at annotations.
Conditions: the middle eligible note's audio MASKED from the likelihood (maximal
ambiguity) vs clean. Committed hypotheses and falsification rule in the script
docstring (`scripts/eval_prior_value_dev.py`). Scored against the quasi-truth centre.

47 cells (chains), tracks: (1, 1)(vn), (1, 2)(vc), (3, 2)(cl), (7, 2)(tbn); 25 cells have a recurring pitch.

| condition | arm | median abs err (cents) | cov@90 | median sd |
|---|---|---|---|---|
| masked | indep | 21.23 | 0.38 | 10.88 |
| masked | temporal | 18.94 | 0.45 | 10.06 |
| masked | graph | 18.78 | 0.49 | 9.95 |
| clean | indep | 2.37 | 0.91 | 6.56 |
| clean | temporal | 2.30 | 0.89 | 6.31 |
| clean | graph | 2.28 | 0.89 | 6.32 |
| masked | (deadpan reference) | 14.42 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs indep: -4.63 [-6.29, -2.87]* (better on 85%, n=47)
- temporal vs indep: -4.39 [-6.40, -2.45]* (better on 83%, n=47)
- graph vs temporal: -0.24 [-0.88, +0.42]  (better on 55%, n=47)
- graph vs temporal, recurring-pitch cells only: -0.72 [-1.26, -0.18]* (better on 64%, n=25)

## Paired contrasts, clean condition (H2: expect ties)

- graph vs indep: -0.00 [-0.02, +0.01]  (better on 53%, n=47)
- temporal vs indep: -0.00 [-0.02, +0.01]  (better on 57%, n=47)

wall per fit incl. Laplace: median 155 s

## Reading (2026-10-14)

1. **H1 confirmed at development level: score structure adds value to
   direct waveform inference exactly where acoustic evidence is
   ambiguous.** With a note's audio removed, the score-graph prior cuts
   the hidden note's intonation error from 21.2 to 18.8 cents median
   (paired −4.63 [−6.29, −2.87], significant, better on 85% of 47
   cells); the temporal prior gains similarly (−4.39, significant). The
   independent prior can only return its mean.
2. **The graph's edge over a generic temporal prior is small overall
   (−0.24, ns) but lands exactly in the pre-stated stratum**: on the 25
   cells whose chain contains a recurring pitch, graph beats temporal
   −0.72 [−1.26, −0.18], significant, better on 64% — pitch-proximity
   coupling transfers intonation across recurrences of the same pitch.
   This is the first waveform-level evidence that *score* structure
   (not merely adjacency in time) carries information, and it was
   predicted, not found post hoc.
3. **H2 confirmed: with clean audio the three arms tie exactly**
   (differences −0.00 [−0.02, +0.01]; medians 2.28–2.37 cents; coverage
   0.89–0.91). Structure costs nothing when evidence dominates — the
   prior's value is specific to ambiguity, which is the methodological
   hypothesis of the thesis in one line.
4. **Honest decomposition caveat: much of the structured win is
   common-mode.** The deadpan reference (median |quasi-truth c| = 14.4
   cents) says the hidden notes sit far from zero largely because tracks
   carry a shared tuning offset; neighbors transfer that offset to the
   hidden note under either structured prior. That transfer is real,
   useful information — but the follow-up must separate tuning-offset
   transfer from note-level expression transfer (e.g., a free per-chain
   tuning scalar shared by all arms, or demeaning by the observed
   notes). The recurring-pitch contrast (point 2) already survives this
   concern qualitatively, since all arms share the common mode.
5. **Masked coverage is far below nominal for every arm (0.38–0.49)**
   — the zero-mean c prior is miscentred on tuning-offset tracks, a
   prior-design finding in its own right (the same tuning scalar would
   repair it). Clean-condition coverage is at nominal (0.89–0.91), so
   the Laplace machinery itself is behaving.
6. Scope honesty: 47 cells on 4 development tracks (the chain-eligible
   ones), intonation constants only, boundaries fixed at annotations,
   s = 1 for both structured kernels (sensitivity unprobed), ~155 s per
   fit including the Laplace. Development evidence; any claim needs its
   own registration.

## ERRATUM (2026-10-15): the masked condition above was contaminated by window leakage

Diagnosed while running the tuning-scalar follow-up
(`results/prior_value_tuning_dev.md`) on the same 47 cells. Facts, from
the stored cells and a profile of the objective on cell 0:

1. Under masking, every arm's hidden-note estimate has **zero
   correlation with the truth** (Pearson −0.00 / +0.01 / +0.00 for
   indep / temporal / graph) and spreads over ±85 cents (sd across cells
   31–35 cents, wider than the prior's own 12.7), and **every arm is
   worse than predicting zero** (deadpan 14.42 vs 18.8–21.2 above). A
   prior-mean-returning independent arm would have scored exactly the
   deadpan value; it did not.
2. Cause: a masked note's sigmoid window tails beyond the 50 ms mask
   guard leave ~4 samples' worth of column mass on the neighbours'
   unmasked audio, and with the amplitudes marginalized that leak
   carves a ~38-nat well along the hidden note's c across ±100 cents,
   against 4–35 nats from the prior. The masked estimates were
   leakage-driven likelihood optima, not prior conditionals (on cell 0
   the optimizer converged to the leaked optimum; the mechanism is the
   leak, not the optimizer).
3. Therefore the reading's point 1 ("the independent prior can only
   return its mean") is false for this protocol, and the headline
   "structure wins under ambiguity" is **not supported as stated**: the
   structured priors' 4.6-cent edge is an edge against leakage-driven
   noise, measured on contaminated ambiguity. Points 2 and 4 (the
   recurring-pitch stratum; the common-mode caveat) are likewise not
   interpretable from this run. The clean-condition rows (point 3) are
   unaffected.
4. Protocol v2 (`PRIORVAL_DROP_HIDDEN=1`; `fit_chain(drop_notes=)`)
   removes the hidden note's design columns from the likelihood so its
   posterior is exactly the prior conditional given the observed notes.
   Its records: `results/prior_value_v2_dev.md` and
   `results/prior_value_v2_tuning_dev.md`. The table above stays as
   measured; nothing in it should be quoted as evidence for the prior.
