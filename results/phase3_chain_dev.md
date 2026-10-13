# Phase 3: transition-region boundaries on joint chains (DEV, exploratory, no claims)

Joint gradient MAP over 3-5-note chains: per-note curves (evidence-fitted prior) +
per-boundary fade-in/fade-out edge pair (width under a lognormal prior), amplitudes
collapsed exactly (curve.fit_chain).  Committed questions in the script docstring
(scripts/eval_phase3_chain_dev.py): (a) fade-in edge vs the pilot's point boundary +
region coverage of the annotation; (b) free vs FIXED-width control; (c) deployment
from score-nominal (tempo-line) inits; (d) per-family widths, unisons stratified out.
Pilot baseline: median 25 ms (results/phase3_boundary_pilot_dev.md).

47 chains, 154 boundaries (26 unison, excluded from the main rows), tracks: (1, 1)(vn), (1, 2)(vc), (3, 2)(cl), (7, 2)(tbn)

| arm | median edge err (ms) | q90 | median center err | region covers annot | median width (ms) |
|---|---|---|---|---|---|
| free-60 | 19.3 | 51.1 | 18.5 | 5% | 1 |
| free+60 | 17.9 | 53.2 | 17.9 | 5% | 1 |
| fixed-60 | 25.6 | 51.0 | 22.7 | 15% | 10 |
| fixed+60 | 22.6 | 54.0 | 21.1 | 8% | 10 |
| nominal | 19.5 | 52.9 | 19.6 | 8% | 1 |

free − fixed paired edge error: median -3.4 ms, free better on 77% of boundaries (n=256)
edge error vs log width (free+60): Pearson r = -0.01
recovery spread |b_lo(+60) − b_lo(−60)|: median 5.7 ms, q90 48.0 ms

deployment (score-nominal init): median edge err 19.5 ms (q90 52.9) from median init err 39.4 ms

Per family, free+60: median edge err / width (non-unison boundaries):
- strings (n=78): 17.0 ms / 1 ms
- winds (n=5): 24.1 ms / 1 ms
- brass (n=45): 19.0 ms / 1 ms
- unison stratum (n=26): median edge err 12.5 ms (expected poor — harmonic likelihood carries no unison boundary)

Curve GT-frame RMSE (cents), median over notes with >=5 frames (guarded ±30 ms at boundaries / unguarded):
- free+60: 5.19 / 9.21 (single-note batch reference 3.16)
- nominal: 4.94 / 8.96 (single-note batch reference 3.16)

window mass: min 0.69, fraction < 0.5 = 0.0% (dropout detector)
wall per fit: median 138 s, q90 337 s

## Reading (2026-10-14)

1. **The region hypothesis is rejected by the data — and the rejection
   is productive.** Free widths collapse to the 1 ms minimum on
   essentially every boundary (the likelihood overrules the lognormal
   prior, as it numerically must), so the inferred "region" is a point
   and covers the annotation only 5% of the time. But the free arm BEATS
   the fixed-10 ms control (paired −3.4 ms, better on 77% of 256
   boundaries) and width does not absorb pitch error (r = −0.01): the
   acoustic transition, as this harmonic likelihood sees it, is SHARP.
   The transition-region idea survives only as its sharp fade-in edge.
2. **Committed question (a): yes.** Median edge error 17.9–19.3 ms vs
   the pilot's 25 ms point boundary — improvement from the sharp-edge
   model + chains + the reference-noise confound fix. The ~18 ms
   residual is systematic (q90 ~51 ms) and is the acoustic-vs-musical
   onset gap restated; by the pre-named criterion (>15 ms residual),
   transient/broadband columns at the boundary are the next lever.
3. **Committed question (c) — the headline: deployment works.** From
   score-nominal (tempo-line) inits with NO annotation access, median
   boundary error drops 39.4 → 19.5 ms, matching the recovery arms. The
   first genuine Phase-3 joint inference numbers: score + waveform in,
   boundaries + curves out.
4. **Start-independence largely holds** (median recovery spread 5.7 ms)
   with a real tail (q90 48 ms) — chains occasionally bistable, as the
   pilot's eighth pair already showed.
5. **Surprise worth keeping: the unison stratum is BETTER (12.5 ms),
   not worse.** The expectation was no-signal (shared harmonics carry no
   boundary); the likely mechanism is that re-articulation amplitude
   dips are visible to the per-note chunked amplitude envelopes, whose
   chunk edges move with the boundary. Not investigated further; noted.
6. **Joint chains currently cost curve accuracy**: guarded GT-frame
   RMSE 4.9–5.2 cents vs 3.16 for single-note fits (unguarded 9.0–9.2 —
   boundary neighbourhoods are bad for everyone). Deploy-time
   recommendation stays: chains for boundaries, single-note refits for
   curves, until the transient lever is measured.
7. **Scope honesty**: selection (3–5 consecutive notes, gaps ≤ 40 ms)
   found chains on only 4 of 7 tracks (vn, vc, cl, tbn); flute, sax,
   trumpet contributed none, and winds n=5 boundaries — family rows are
   indicative only. No window-mass dropouts (min 0.69); noise refits
   logged per fit.
