# The boundary-prior experiment: score structure in note-boundary inference from the waveform (DEV, exploratory, no claims)

Three cross-joint priors over boundary offsets from the tempo-line boundaries, matched marginal
variance, free per-chain offset scalar in every arm, identical likelihood/data/optimizer, sharp
edges. Masked = the hidden (middle) joint removed from the design (hard cuts +/-100 ms, samples
masked); clean = deployment from the tempo line with full audio. Committed hypotheses and
falsification rule in the script docstring (`scripts/eval_boundary_prior_dev.py`). Scored against the
annotated onset. Unison joints are a separate stratum, excluded from the main rows.

47 cells (chains), 40 non-unison, tracks: (1, 1)(vn), (1, 2)(vc), (3, 2)(cl), (7, 2)(tbn)

| condition | arm | median abs err (ms) | q90 | cov@90 | median sd |
|---|---|---|---|---|---|
| masked | indep | 30.6 | 74.7 | 1.00 | 95.0 |
| masked | temporal | 29.1 | 72.7 | 1.00 | 73.1 |
| masked | graph | 28.1 | 69.9 | 1.00 | 75.1 |
| clean | indep | 19.6 | 47.8 | 0.05 | 0.7 |
| clean | temporal | 19.6 | 47.8 | 0.05 | 0.7 |
| clean | graph | 19.6 | 47.8 | 0.05 | 0.7 |
| masked | (deadpan = tempo line, analytic indep without the scalar) | 37.8 | 118.9 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs indep: -2.45 [-8.38, +3.32]  (better on 48%, n=40)
- temporal vs indep: -3.65 [-10.90, +3.03]  (better on 52%, n=40)
- graph vs temporal: +1.20 [-2.12, +4.45]  (better on 48%, n=40)
- indep vs deadpan (tempo line): -9.99 [-16.61, -3.38]* (better on 62%)
- temporal vs deadpan (tempo line): -13.64 [-25.43, -2.32]* (better on 60%)
- graph vs deadpan (tempo line): -12.43 [-24.34, -1.74]* (better on 60%)

## Paired contrasts, clean condition (H-B2: expect ties; deployment with a prior)

- graph vs indep: -0.00 [-0.01, -0.00]* (better on 70%, n=40)
- temporal vs indep: -0.00 [-0.01, -0.00]* (better on 70%, n=40)

## Unison stratum (n=7), masked median abs err: indep 5.9, temporal 11.1, graph 3.1

wall per fit incl. Laplace: median 156 s

## Reading (2026-10-20)

1. **H-B1: the falsification rule fires.** On the 40 non-unison masked
   cells the structured arms tie the scalar-augmented independent arm:
   graph vs indep −2.45 [−8.38, +3.32] (48%), temporal vs indep −3.65
   [−10.90, +3.03] (52%), graph vs temporal +1.20 [−2.12, +4.45]. At
   chain scale, boundary timing carries no measurable note-level
   structure beyond the chain's common offset — the same verdict the
   intonation constants gave (`results/prior_value_v2_tuning_dev.md`).
2. **The common offset itself is worth 10–14 ms against the tempo
   line.** Every arm beats the deadpan reference (|τ| median 37.8 ms):
   indep −9.99 [−16.61, −3.38]*, temporal −13.64*, graph −12.43*. A
   hidden joint inherits its chain's mean offset from the tempo line
   (local tempo error plus the acoustic-vs-musical onset gap) through
   the free scalar alone; structure adds nothing on top.
3. **H-B2 holds, and reproduces the chain study.** On clean audio all
   three arms land on the same boundaries (19.6 ms median, q90 47.8;
   the chain study's deployment arm measured 19.5): a prior over the
   offsets neither helps nor hurts deployment when the audio is
   present. The starred −0.00 contrasts are sub-0.01 ms differences,
   numerically real and scientifically nothing.
4. **H-B3, two calibration findings.** (a) The masked conditional
   bands are far too wide: coverage 1.00 with median sd 73–95 ms
   against errors of ~30 ms. The prior scale σ_b (the track's full τ
   variance, 62 ms on these tracks) is a population scale that
   overstates within-chain variation; a chain-scale σ_b would tighten
   the band without changing the point verdict. (b) The observed
   joints' bands on clean audio are far too narrow: coverage 0.05 with
   sd 0.7 ms. The likelihood pins the ACOUSTIC transition to under a
   millisecond while the annotation marks the musical onset ~20 ms
   away — the estimand gap of the intonation studies, restated at the
   boundary, and the reason any boundary band reported against
   annotations needs the measured discrepancy floor.
5. **Unison stratum** (n=7): graph 3.1 / indep 5.9 / temporal 11.1 ms
   medians; too few cells to read.
6. Scope: 47 chains on 4 development tracks (the chain study's set),
   sharp edges, boundaries initialized from the tempo line, 156 s per
   fit; development evidence, no claims.
