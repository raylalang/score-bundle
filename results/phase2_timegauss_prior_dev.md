# Score graph vs time-only Gaussian graph vs time-only chain on the strings-and-winds bundle at track scale — DEV, exploratory, no claims (77 unique tracks, 2 seeds, 30% of notes hidden)

> Hypothesis H-T3 and the falsification rule in the docstring of scripts/eval_phase2_real.py (committed before the run). The time-only Gaussian graph is the score graph with the pitch term removed (same 2-beat time window). Primary read: gp_asgiven - gp_tg_asgiven on channel c.

## vs estimator targets (primary; predictive sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | loudness ell RMSE / NLL / cov@90 | tau (s) RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|---|---|
| score-graph GP (learned scale) | 14.055 / +3.89 / 0.89 (n=9629) | 1.527 / +4.79 / 0.91 (n=3275) | 0.406 / +0.22 / 0.89 (n=3275) | 0.415 / +0.50 / 0.89 (n=9629) | 0.105 / -1.63 / 0.91 (n=9091) | 0.129 / -0.87 / 0.90 (n=3833) |
| score-graph GP (as-given) | 14.132 / +3.92 / 0.89 (n=9629) | 1.513 / +1.89 / 0.90 (n=3275) | 0.376 / +0.24 / 0.88 (n=3275) | 0.420 / +0.52 / 0.90 (n=9629) | 0.109 / -1.63 / 0.91 (n=9091) | 0.130 / -0.86 / 0.89 (n=3833) |
| no-graph ablation | 14.383 / +3.90 / 0.90 (n=9629) | 1.554 / +5.53 / 0.92 (n=3275) | 0.563 / +0.40 / 0.89 (n=3275) | 0.429 / +0.51 / 0.90 (n=9629) | 0.110 / -1.44 / 0.91 (n=9091) | 0.132 / -0.88 / 0.90 (n=3833) |
| time-only chain GP (learned scale) | 14.597 / +3.93 / 0.88 (n=9629) | 1.636 / +6.05 / 0.90 (n=3275) | 0.493 / +0.34 / 0.87 (n=3275) | 0.421 / +0.51 / 0.89 (n=9629) | 0.106 / -1.55 / 0.92 (n=9091) | 0.135 / -0.80 / 0.89 (n=3833) |
| time-only chain GP (as-given) | 14.665 / +3.96 / 0.88 (n=9629) | 1.565 / +2.17 / 0.89 (n=3275) | 0.497 / +0.40 / 0.87 (n=3275) | 0.426 / +0.55 / 0.89 (n=9629) | 0.110 / -1.52 / 0.92 (n=9091) | 0.136 / -0.78 / 0.89 (n=3833) |
| time-only Gaussian graph GP (learned scale) | 14.122 / +3.90 / 0.89 (n=9629) | 1.504 / +5.68 / 0.91 (n=3275) | 0.371 / +0.16 / 0.89 (n=3275) | 0.423 / +0.52 / 0.89 (n=9629) | 0.105 / -1.61 / 0.91 (n=9091) | 0.129 / -0.85 / 0.89 (n=3833) |
| time-only Gaussian graph GP (as-given) | 14.179 / +3.92 / 0.89 (n=9629) | 1.501 / +2.14 / 0.89 (n=3275) | 0.350 / +0.21 / 0.88 (n=3275) | 0.426 / +0.54 / 0.90 (n=9629) | 0.109 / -1.62 / 0.91 (n=9091) | 0.130 / -0.87 / 0.88 (n=3833) |

## vs ground-truth-derived targets (quasi-truth; latent sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|
| score-graph GP (learned scale) | 13.985 / +4.39 / 0.79 (n=9627) | 1.020 / +2.66 / 0.73 (n=3605) | 0.411 / +2.18 / 0.72 (n=3605) | 0.138 / -0.80 / 0.86 (n=4112) |
| score-graph GP (as-given) | 14.101 / +4.37 / 0.82 (n=9627) | 0.999 / +1.64 / 0.83 (n=3605) | 0.368 / +0.42 / 0.85 (n=3605) | 0.139 / -0.76 / 0.86 (n=4112) |
| no-graph ablation | 14.280 / +4.19 / 0.83 (n=9627) | 1.177 / +2.56 / 0.74 (n=3605) | 0.577 / +2.27 / 0.72 (n=3605) | 0.139 / -0.85 / 0.87 (n=4112) |
| time-only chain GP (learned scale) | 14.456 / +4.29 / 0.80 (n=9627) | 1.239 / +2.55 / 0.72 (n=3605) | 0.517 / +2.09 / 0.70 (n=3605) | 0.142 / -0.72 / 0.84 (n=4112) |
| time-only chain GP (as-given) | 14.584 / +4.28 / 0.82 (n=9627) | 1.161 / +1.64 / 0.83 (n=3605) | 0.515 / +0.57 / 0.85 (n=3605) | 0.145 / -0.75 / 0.86 (n=4112) |
| time-only Gaussian graph GP (learned scale) | 14.052 / +4.37 / 0.80 (n=9627) | 1.003 / +3.98 / 0.73 (n=3605) | 0.375 / +1.97 / 0.73 (n=3605) | 0.138 / -0.80 / 0.86 (n=4112) |
| time-only Gaussian graph GP (as-given) | 14.171 / +4.33 / 0.82 (n=9627) | 0.982 / +1.72 / 0.82 (n=3605) | 0.342 / +0.41 / 0.85 (n=3605) | 0.140 / -0.81 / 0.85 (n=4112) |

## Median per-(track, seed) RMSE vs estimator targets

| system | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| score-graph GP (learned scale) | 11.272 | 0.712 | 0.344 | 0.399 | 0.073 | 0.086 |
| score-graph GP (as-given) | 11.054 | 0.621 | 0.300 | 0.402 | 0.072 | 0.084 |
| no-graph ablation | 12.558 | 0.920 | 0.407 | 0.413 | 0.074 | 0.086 |
| time-only chain GP (learned scale) | 11.762 | 0.848 | 0.394 | 0.400 | 0.066 | 0.091 |
| time-only chain GP (as-given) | 11.731 | 0.843 | 0.407 | 0.405 | 0.074 | 0.091 |
| time-only Gaussian graph GP (learned scale) | 11.286 | 0.642 | 0.323 | 0.403 | 0.071 | 0.085 |
| time-only Gaussian graph GP (as-given) | 11.301 | 0.600 | 0.287 | 0.413 | 0.072 | 0.081 |

## Paired graph value (learned scale; gp - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.925 [-1.518, -0.435]* | -0.033 [-0.079, +0.015]  (n=153) |
| log gamma | -0.186 [-0.275, -0.099]* | -0.987 [-2.384, -0.219]* (n=148) |
| log f | -0.243 [-0.317, -0.172]* | -0.356 [-0.456, -0.258]* (n=148) |
| loudness ell | -0.006 [-0.012, -0.001]* | +0.026 [-0.007, +0.071]  (n=153) |
| tau (s) | -0.008 [-0.014, -0.004]* | -0.319 [-0.829, -0.056]* (n=149) |
| delta_vib (s) | -0.005 [-0.009, -0.001]* | +0.014 [-0.040, +0.086]  (n=153) |

## Paired graph value (as-given, the default; gp_asgiven - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.896 [-1.517, -0.383]* | -0.020 [-0.068, +0.023]  (n=153) |
| log gamma | -0.258 [-0.364, -0.139]* | -4.293 [-12.064, -0.336]* (n=148) |
| log f | -0.301 [-0.383, -0.225]* | -0.442 [-0.580, -0.309]* (n=148) |
| loudness ell | -0.003 [-0.010, +0.003]  | +0.041 [+0.009, +0.082]* (n=153) |
| tau (s) | -0.002 [-0.004, +0.000]  | -0.297 [-0.793, -0.036]* (n=149) |
| delta_vib (s) | -0.001 [-0.007, +0.005]  | +0.057 [-0.034, +0.181]  (n=153) |

## Paired SCORE GRAPH vs TIME-ONLY GAUSSIAN GRAPH (PRIMARY read, the pitch term; as-given, gp_asgiven - gp_tg_asgiven), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.018 [-0.103, +0.064]  | +0.006 [-0.011, +0.025]  (n=153) |
| log gamma | +0.042 [+0.006, +0.086]* | -0.260 [-0.830, +0.061]  (n=148) |
| log f | +0.030 [+0.004, +0.063]* | +0.063 [+0.006, +0.132]* (n=148) |
| loudness ell | -0.007 [-0.010, -0.003]* | -0.014 [-0.024, -0.004]* (n=153) |
| tau (s) | +0.000 [-0.000, +0.001]  | -0.019 [-0.061, +0.004]  (n=149) |
| delta_vib (s) | -0.000 [-0.002, +0.001]  | +0.004 [-0.048, +0.049]  (n=153) |

## Paired score graph vs time-only Gaussian graph (learned scale; gp - gp_tg), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.046 [-0.115, +0.015]  | +0.001 [-0.014, +0.016]  (n=153) |
| log gamma | +0.060 [+0.013, +0.109]* | -1.001 [-2.993, +0.036]  (n=148) |
| log f | +0.042 [-0.000, +0.088]  | +0.098 [+0.018, +0.182]* (n=148) |
| loudness ell | -0.007 [-0.009, -0.004]* | -0.018 [-0.030, -0.007]* (n=153) |
| tau (s) | +0.000 [-0.001, +0.001]  | -0.033 [-0.119, +0.016]  (n=149) |
| delta_vib (s) | -0.001 [-0.003, +0.001]  | -0.045 [-0.121, +0.010]  (n=153) |

## Paired time-Gaussian vs time-only chain (the window; as-given), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.479 [-0.646, -0.322]* | -0.043 [-0.065, -0.021]* (n=153) |
| log gamma | -0.266 [-0.350, -0.189]* | -0.168 [-0.908, +0.952]  (n=148) |
| log f | -0.242 [-0.319, -0.172]* | -0.456 [-0.576, -0.343]* (n=148) |
| loudness ell | -0.000 [-0.006, +0.006]  | -0.022 [-0.055, +0.012]  (n=153) |
| tau (s) | -0.002 [-0.004, +0.000]  | -0.203 [-0.616, +0.010]  (n=149) |
| delta_vib (s) | -0.004 [-0.010, +0.001]  | -0.113 [-0.171, -0.058]* (n=153) |

## Paired time-Gaussian value (as-given; gp_tg_asgiven - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.878 [-1.468, -0.382]* | -0.026 [-0.075, +0.021]  (n=153) |
| log gamma | -0.300 [-0.418, -0.187]* | -4.033 [-11.342, -0.383]* (n=148) |
| log f | -0.331 [-0.413, -0.252]* | -0.505 [-0.642, -0.375]* (n=148) |
| loudness ell | +0.003 [-0.003, +0.010]  | +0.055 [+0.021, +0.095]* (n=153) |
| tau (s) | -0.002 [-0.004, +0.000]  | -0.278 [-0.733, -0.037]* (n=149) |
| delta_vib (s) | -0.001 [-0.007, +0.006]  | +0.054 [-0.022, +0.162]  (n=153) |

## Paired SCORE GRAPH vs TIME-ONLY CHAIN (PRIMARY read; as-given, gp_asgiven - gp_chain_asgiven), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.496 [-0.678, -0.321]* | -0.037 [-0.066, -0.009]* (n=153) |
| log gamma | -0.224 [-0.304, -0.146]* | -0.428 [-0.984, +0.114]  (n=148) |
| log f | -0.212 [-0.287, -0.148]* | -0.393 [-0.509, -0.282]* (n=148) |
| loudness ell | -0.007 [-0.013, -0.001]* | -0.036 [-0.071, -0.003]* (n=153) |
| tau (s) | -0.001 [-0.004, +0.001]  | -0.222 [-0.674, +0.012]  (n=149) |
| delta_vib (s) | -0.004 [-0.010, +0.001]  | -0.109 [-0.191, -0.037]* (n=153) |

## Paired score graph vs time-only chain (learned scale; gp - gp_chain), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.427 [-0.574, -0.282]* | -0.042 [-0.070, -0.013]* (n=153) |
| log gamma | -0.182 [-0.255, -0.109]* | -1.501 [-3.934, -0.220]* (n=148) |
| log f | -0.142 [-0.202, -0.089]* | -0.233 [-0.321, -0.152]* (n=148) |
| loudness ell | -0.002 [-0.008, +0.004]  | -0.007 [-0.040, +0.029]  (n=153) |
| tau (s) | -0.002 [-0.007, +0.002]  | -0.229 [-0.795, +0.066]  (n=149) |
| delta_vib (s) | -0.006 [-0.011, -0.002]* | -0.076 [-0.141, -0.005]* (n=153) |

## Paired chain value (as-given; gp_chain_asgiven - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.399 [-0.987, +0.119]  | +0.017 [-0.039, +0.067]  (n=153) |
| log gamma | -0.034 [-0.115, +0.060]  | -3.865 [-11.584, +0.079]  (n=148) |
| log f | -0.089 [-0.148, -0.032]* | -0.049 [-0.153, +0.041]  (n=148) |
| loudness ell | +0.004 [-0.005, +0.012]  | +0.077 [+0.032, +0.134]* (n=153) |
| tau (s) | -0.000 [-0.001, +0.001]  | -0.075 [-0.121, -0.037]* (n=149) |
| delta_vib (s) | +0.003 [-0.001, +0.009]  | +0.166 [+0.083, +0.275]* (n=153) |


## Paired graph value by instrument family (dRMSE, gp - nograph)

| family | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| strings | -0.415* (n=70) | -0.078* (n=70) | -0.100* (n=70) | -0.014* (n=70) | -0.004* (n=68) | -0.004* (n=70) |
| wood | -2.349* (n=47) | -0.290* (n=45) | -0.461* (n=45) | +0.007  (n=47) | -0.016* (n=47) | -0.012* (n=47) |
| brass | -0.058  (n=36) | -0.272* (n=33) | -0.246* (n=33) | -0.009* (n=36) | -0.007* (n=34) | +0.003  (n=36) |

## Reading (2026-10-23)

Same 153 cells as `results/phase2_chain_prior_dev.md`, seven arms. The
time-only Gaussian graph is the score graph with the pitch term removed
(`build_adjacency(score, ell_p=1e6)`), the same 2-beat time window.

1. **H-T3 is falsified: the pitch term carries nothing.** Score graph
   minus time-only Gaussian graph, as-given: intonation −0.018 cents
   [−0.103, +0.064] (tie), NLL +0.006 (tie); vibrato extent +0.042*
   and rate +0.030* (the pitch term slightly HURTS), rate NLL +0.063*;
   loudness −0.007* (a small gain); timing and vibrato onset ties.
2. **The whole advantage over the chain is the time window.**
   Time-Gaussian minus chain reproduces the score-graph-minus-chain
   contrasts almost exactly: intonation −0.48* (graph: −0.50*), extent
   −0.27* (−0.22*), rate −0.24* (−0.21*), onset NLL −0.11* (−0.11*).
   And time-Gaussian minus no-graph on intonation is −0.878*, the
   score graph's −0.896* to within noise.
3. **What this says.** At track scale on tracked targets, the
   confirmed strings-and-winds "graph value" is temporal smoothing
   over a Gaussian window of about two beats. Linking notes by pitch
   proximity on top of that adds nothing measurable on intonation and
   costs a little on vibrato. Together with the waveform spans (nothing
   beyond the common offset at 3–10 notes, no dose-response on the
   recurrence edge), the picture is consistent at every scale and
   observation type tested: the structure that helps is smoothness in
   score time with the right window; the score-specific pitch
   coupling does not show.
4. **Open on piano.** The piano confirmation's graph value was measured
   against a nearest-neighbour chain in the development kernel study,
   not against a time-only Gaussian graph; on polyphonic piano the time
   window links chord notes (Δb = 0) strongly, so the pitch term may
   matter differently there. Checked next if cheap.
5. Scope: development split, selection-reused, no registration; the
   registered results untouched.
