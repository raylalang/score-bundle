# The boundary-prior experiment: score structure in note-boundary inference from the waveform (chain-scale prior width; DEV, exploratory, no claims)

Three cross-joint priors over boundary offsets from the tempo-line boundaries, matched marginal
variance, free per-chain offset scalar in every arm, identical likelihood/data/optimizer, sharp
edges. Masked = the hidden (middle) joint removed from the design (hard cuts +/-100 ms, samples
masked); clean = deployment from the tempo line with full audio. Committed hypotheses and
falsification rule in the script docstring (`scripts/eval_boundary_prior_dev.py`). Scored against the
annotated onset. Unison joints are a separate stratum, excluded from the main rows.

47 cells (chains), 40 non-unison, tracks: (1, 1)(vn), (1, 2)(vc), (3, 2)(cl), (7, 2)(tbn)

| condition | arm | median abs err (ms) | q90 | cov@90 | median sd |
|---|---|---|---|---|---|
| masked | indep | 32.1 | 73.8 | 0.88 | 45.9 |
| masked | temporal | 33.6 | 74.9 | 0.75 | 33.2 |
| masked | graph | 38.6 | 75.4 | 0.75 | 35.1 |
| clean | indep | 19.5 | 47.8 | 0.05 | 0.7 |
| clean | temporal | 19.5 | 47.8 | 0.05 | 0.7 |
| clean | graph | 19.4 | 47.8 | 0.05 | 0.7 |
| masked | (deadpan = tempo line, analytic indep without the scalar) | 37.8 | 118.9 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs indep: +0.61 [-2.07, +3.29]  (better on 42%, n=40)
- temporal vs indep: -0.32 [-4.79, +3.96]  (better on 42%, n=40)
- graph vs temporal: +0.92 [-2.26, +3.96]  (better on 55%, n=40)
- indep vs deadpan (tempo line): -12.08 [-23.75, -0.75]* (better on 60%)
- temporal vs deadpan (tempo line): -12.40 [-26.60, +0.76]  (better on 52%)
- graph vs deadpan (tempo line): -11.48 [-25.66, +1.44]  (better on 52%)

## Paired contrasts, clean condition (H-B2: expect ties; deployment with a prior)

- graph vs indep: -0.02 [-0.03, -0.01]* (better on 70%, n=40)
- temporal vs indep: -0.02 [-0.03, -0.01]* (better on 68%, n=40)

## Unison stratum (n=7), masked median abs err: indep 3.5, temporal 11.7, graph 5.0

wall per fit incl. Laplace: median 501 s

## Reading (2026-10-22)

Calibration-only rerun of `results/boundary_prior_dev.md` with the
prior width set at chain scale (σ_b² = median over the track's chains
of the within-chain τ variance; 31 ms on the violin track versus 62 ms
for the track's full τ spread). Fits ran alongside the track-scale
study, hence the slower wall time.

1. **H-B4, partly.** Masked coverage of the hidden joint moved from
   1.00 to 0.88 for the independent arm (median sd 46 ms), at nominal.
   For the structured arms it moved to 0.75 (sd 33–35 ms): the
   chain-scale width overshoots for them, because their conditional
   band is tighter still once the neighbours are known. No single
   population-scale choice of σ_b calibrates all three arms at once;
   the principled fix is a σ_b learned per chain by evidence, which this
   rerun does not do. Reported as such.
2. **H-B5 holds.** The arms still tie on the hidden joint (graph vs
   independent +0.61 [−2.07, +3.29]; temporal vs independent −0.32;
   graph vs temporal +0.92), and clean deployment is unchanged at
   19.5 ms (the starred −0.02 ms differences are numerical noise). The
   gain against the tempo line stays for the independent arm (−12.1*)
   and keeps its size but loses significance for the structured arms
   (−12.4, −11.5; wider intervals, medians 32–39 ms versus 28–31 before):
   with a tighter structured prior the hidden joint is pulled more
   toward its neighbours and less toward the chain mean, and the
   neighbours are no better a guide than the mean. The verdict of the
   second span is unchanged.
3. Unison stratum (n=7): 3.5 / 11.7 / 5.0 ms; too few to read.
4. Scope: as the first run; development, no claims. Worth carrying into
   the thesis as one sentence on the band width, nothing more.
