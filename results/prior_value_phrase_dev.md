# The prior-value experiment at phrase scale: distant pitch recurrence (protocol v2 + tuning scalar; DEV, exploratory, no claims)

6-10-note legato runs from all development tracks in which the hidden (middle) note's pitch
recurs at index distance >= 2 and not at distance 1; otherwise the machinery of
`scripts/eval_prior_value_dev.py` unchanged (hidden note dropped from the design, free per-chain
tuning scalar in every arm). Committed hypotheses and falsification rule in the script
docstring (`scripts/eval_prior_value_phrase_dev.py`). Scored against the quasi-truth centre.

48 cells, notes/cell median 8, span median 4.5 s, recurrence distance median 3 notes / 1.2 s, hidden-to-recurrence edge weight median 0.83; instruments: cl, db, hn, sax, tbn, tpt, va, vc, vn

| condition | arm | median abs err (cents) | cov@90 | median sd |
|---|---|---|---|---|
| masked | indep | 8.61 | 0.96 | 13.99 |
| masked | temporal | 9.28 | 0.90 | 12.05 |
| masked | graph | 8.16 | 0.94 | 11.42 |
| clean | indep | 3.74 | 0.94 | 7.71 |
| clean | temporal | 3.76 | 0.94 | 7.35 |
| clean | graph | 3.75 | 0.94 | 7.15 |
| masked | (deadpan reference) | 10.71 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs temporal: -1.63 [-2.93, -0.46]* (better on 65%, n=48)
- graph vs indep: -0.07 [-0.42, +0.27]  (better on 54%, n=48)
- temporal vs indep: +1.56 [+0.26, +2.92]* (better on 42%, n=48)
- graph vs temporal, upper half of edge weight only: -1.90 [-4.00, +0.10]  (better on 62%, n=24)

## Paired contrasts, clean condition (H-C3: expect ties)

- graph vs indep: -0.00 [-0.01, +0.01]  (better on 52%, n=48)
- temporal vs indep: +0.00 [-0.01, +0.01]  (better on 40%, n=48)

Dose-response: Pearson r between (graph - temporal) masked error and the hidden-to-recurrence edge weight = -0.03

wall per fit incl. Laplace: median 673 s

## Reading (2026-10-20)

1. **H-C1 passes as literally stated**: the score graph beats the
   temporal chain on the hidden note, −1.63 [−2.93, −0.46], better on
   65% of 48 cells; the upper-half-edge-weight stratum −1.90
   [−4.00, +0.10]. By the committed rule the pitch-proximity coupling
   is NOT falsified at phrase scale.
2. **But the mechanism claim does not hold up.** The graph ties the
   scalar-augmented independent arm (−0.07 [−0.42, +0.27], 54%), and
   the temporal chain is significantly WORSE than independent (+1.56
   [+0.26, +2.92]). The graph's edge over the temporal chain is
   therefore the temporal prior's harm repaired, not note-level
   information recovered: at phrase scale, adjacency pulls the hidden
   note toward neighbours that resemble it less than the chain mean
   does, and the pitch edge to the recurrence pulls it back to about
   the chain mean. The dose-response is nil (Pearson r = −0.03 between
   the graph-minus-temporal error and the hidden-to-recurrence edge
   weight; −0.19 for graph-minus-independent, post hoc).
3. **H-C2: tie**, as at chain scale. Post hoc (not pre-stated): paired
   Gaussian NLL graph vs independent −0.05 [−0.12, +0.06] (better on
   79%, ns); graph vs temporal −0.12 [−0.21, −0.04]*; the graph's band
   is the tightest (sd 11.4 vs 14.0 cents) at nominal coverage (0.94),
   a sharpness trend without a significant score advantage.
4. **H-C3 holds**: clean rows tie to the second decimal, coverage 0.94.
5. **Family split (post hoc)**: strings (n=33) carry the graph-vs-
   temporal star (−2.16*); winds (n=7) and brass (n=8) are
   uninformative at their n. Graph vs independent is a tie in every
   family.
6. **What the third span says.** Against the thesis's own stated
   baselines — independent or purely temporal — the score graph is
   never worse and, at phrase scale, beats the temporal prior; against
   the common-mode baseline that the first span showed to be mandatory,
   it adds nothing measurable for intonation at either span. The
   methodological hypothesis survives in its weak form (score structure
   beats naive temporal structure) and fails in its strong form
   (score structure recovers note-level expression a performer-wide
   offset cannot). Scope: 48 cells, 9 instruments, legato runs only,
   673 s per fit; development evidence, no claims.
