# The prior-value experiment at phrase scale on the URMP CONFIRMATION tracks (registered one-shot, docs/phrase_prior_prereg_design.md; third disclosed use of the pool, first at the waveform level)

6-10-note legato runs from all confirmation tracks in which the hidden (middle) note's pitch
recurs at index distance >= 2 and not at distance 1; otherwise the machinery of
`scripts/eval_prior_value_dev.py` unchanged (hidden note dropped from the design, free per-chain
tuning scalar in every arm). Committed hypotheses and falsification rule in the script
docstring (`scripts/eval_prior_value_phrase_dev.py`). Scored against the reference centre.

48 cells, notes/cell median 8, span median 4.8 s, recurrence distance median 2 notes / 1.3 s, hidden-to-recurrence edge weight median 0.82; instruments: bn, cl, fl, hn, ob, sax, tbn, tpt, va, vc, vn

| condition | arm | median abs err (cents) | cov@90 | median sd |
|---|---|---|---|---|
| masked | indep | 4.11 | 0.98 | 11.25 |
| masked | temporal | 4.01 | 0.94 | 9.77 |
| masked | graph | 3.68 | 0.96 | 8.93 |
| clean | indep | 1.65 | 0.96 | 6.91 |
| clean | temporal | 1.67 | 0.96 | 6.51 |
| clean | graph | 1.67 | 0.96 | 6.30 |
| masked | (deadpan reference) | 15.53 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs temporal: -0.82 [-1.59, -0.05]* (better on 62%, n=48)
- graph vs indep: -0.13 [-0.44, +0.17]  (better on 56%, n=48)
- temporal vs indep: +0.68 [-0.19, +1.52]  (better on 40%, n=48)
- graph vs temporal, upper half of edge weight only: -1.07 [-2.24, +0.03]  (better on 62%, n=24)

## Paired contrasts, clean condition (H-C3: expect ties)

- graph vs indep: -0.00 [-0.01, +0.01]  (better on 46%, n=48)
- temporal vs indep: +0.00 [-0.01, +0.01]  (better on 44%, n=48)

Dose-response: Pearson r between (graph - temporal) masked error and the hidden-to-recurrence edge weight = -0.06

wall per fit incl. Laplace: median 472 s

## VERDICT against the registered claims (scored 2026-11-18, run 04:47-10:21 +09:00, 5 h 34 min, 12 shards x OMP=2, code state 6baf314 = tag phrase-prior-registration-2026-11-18; decision rule verbatim from docs/phrase_prior_prereg_design.md @ that tag)

- **W1 (the weak form, PRIMARY) — PASS.** Mean paired difference in absolute error on the hidden note, graph minus temporal, masked condition, against the reference centre: −0.82 cents [−1.59, −0.05]\*, percentile bootstrap over the 48 cells, B = 2000, rng 31 in the frozen contrast order. Negative mean, interval excludes zero. Better on 62% of cells (median −0.90, SD 2.76). Development basis −1.63 [−2.93, −0.46]\* on 48 development cells; the pre-stated expectation for this winds-heavy pool was about −0.9, and the measured mean is −0.82.
- **W2 (the strong form, report-only) — tie, as on development.** Graph minus independent −0.13 [−0.44, +0.17], better on 56%. No new evidence for the strong form: beyond the common tuning offset the score graph adds nothing measurable at phrase scale on these tracks either.
- **W3 (control) — PASS.** Clean-condition contrasts tie: graph minus independent −0.00 [−0.01, +0.01], temporal minus independent +0.00 [−0.01, +0.01], both |mean| below 0.1 cents.
- Temporal minus independent +0.68 [−0.19, +1.52], better on 40%: the temporal prior's harm, starred on development (+1.56\*), is present in sign and size but not starred here. The repaired harm is again what the graph's win consists of.
- Coverage of the 90% band on the hidden note: independent 0.98, temporal 0.94, graph 0.96; clean 0.96 on all three. Median posterior sd 11.3 / 9.8 / 8.9 cents.
- Dose-response: r = −0.06 between the contrast and the hidden-to-recurrence edge weight (development −0.03); the upper half of edge weight gives −1.07 [−2.24, +0.03]. No dose-response, as on development.
- Report-only, not pre-registered beyond the family expectation in the registration doc: by family, graph minus temporal is −1.55 [−3.32, +0.34] on 14 string cells, −0.54 [−1.32, +0.26] on 25 wind cells, −0.44 [−2.52, +1.45] on 9 brass cells (seeds 1-3, B = 2000). The ordering of the development run (strings carry the effect) repeats; no family is starred on its own.
- Cells: 48 of 48 scored from 20 tracks; no shard errors; median wall per fit 472 s.
- **Headline, by the registered rule:** at phrase scale the score graph beats a purely temporal prior on held-out tracks, confirmed once on the re-used pool (disclosed, third use, the first at the waveform level). The strong form stays unconfirmed: the graph ties the independent prior, and the win is the temporal prior's harm repaired. The disclosure stands: these 40 tracks were spent on claims C1–C4 and re-used for the correlated-noise and tonal claims at the tracked-target level; no waveform script had read them before this run.
- Evidence: evidence/phrase_prior_confirmation/ (run log with the drawn cell list, twelve shard logs, twelve cell shards).
