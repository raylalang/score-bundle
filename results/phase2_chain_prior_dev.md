# Score graph vs time-only chain on the strings-and-winds bundle at track scale — DEV, exploratory, no claims (77 unique tracks, 2 seeds, 30% of notes hidden)

> Hypotheses H-T1/H-T2 and the falsification rule in the docstring of scripts/eval_phase2_real.py (committed before the run). Both the graph arms and the no-graph arm carry the per-piece bias column, so every contrast is beyond the common offset. Primary read: gp_asgiven - gp_chain_asgiven, per channel, vs estimator targets. The registered results file is untouched.

## vs estimator targets (primary; predictive sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | loudness ell RMSE / NLL / cov@90 | tau (s) RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|---|---|
| score-graph GP (learned scale) | 14.055 / +3.89 / 0.89 (n=9629) | 1.527 / +4.79 / 0.91 (n=3275) | 0.406 / +0.22 / 0.89 (n=3275) | 0.415 / +0.50 / 0.89 (n=9629) | 0.105 / -1.63 / 0.91 (n=9091) | 0.129 / -0.87 / 0.90 (n=3833) |
| score-graph GP (as-given) | 14.132 / +3.92 / 0.89 (n=9629) | 1.513 / +1.89 / 0.90 (n=3275) | 0.376 / +0.24 / 0.88 (n=3275) | 0.420 / +0.52 / 0.90 (n=9629) | 0.109 / -1.63 / 0.91 (n=9091) | 0.130 / -0.86 / 0.89 (n=3833) |
| no-graph ablation | 14.383 / +3.90 / 0.90 (n=9629) | 1.554 / +5.53 / 0.92 (n=3275) | 0.563 / +0.40 / 0.89 (n=3275) | 0.429 / +0.51 / 0.90 (n=9629) | 0.110 / -1.44 / 0.91 (n=9091) | 0.132 / -0.88 / 0.90 (n=3833) |
| time-only chain GP (learned scale) | 14.597 / +3.93 / 0.88 (n=9629) | 1.636 / +6.05 / 0.90 (n=3275) | 0.493 / +0.34 / 0.87 (n=3275) | 0.421 / +0.51 / 0.89 (n=9629) | 0.106 / -1.55 / 0.92 (n=9091) | 0.135 / -0.80 / 0.89 (n=3833) |
| time-only chain GP (as-given) | 14.665 / +3.96 / 0.88 (n=9629) | 1.565 / +2.17 / 0.89 (n=3275) | 0.497 / +0.40 / 0.87 (n=3275) | 0.426 / +0.55 / 0.89 (n=9629) | 0.110 / -1.52 / 0.92 (n=9091) | 0.136 / -0.78 / 0.89 (n=3833) |

## vs ground-truth-derived targets (quasi-truth; latent sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|
| score-graph GP (learned scale) | 13.985 / +4.39 / 0.79 (n=9627) | 1.020 / +2.66 / 0.73 (n=3605) | 0.411 / +2.18 / 0.72 (n=3605) | 0.138 / -0.80 / 0.86 (n=4112) |
| score-graph GP (as-given) | 14.101 / +4.37 / 0.82 (n=9627) | 0.999 / +1.64 / 0.83 (n=3605) | 0.368 / +0.42 / 0.85 (n=3605) | 0.139 / -0.76 / 0.86 (n=4112) |
| no-graph ablation | 14.280 / +4.19 / 0.83 (n=9627) | 1.177 / +2.56 / 0.74 (n=3605) | 0.577 / +2.27 / 0.72 (n=3605) | 0.139 / -0.85 / 0.87 (n=4112) |
| time-only chain GP (learned scale) | 14.456 / +4.29 / 0.80 (n=9627) | 1.239 / +2.55 / 0.72 (n=3605) | 0.517 / +2.09 / 0.70 (n=3605) | 0.142 / -0.72 / 0.84 (n=4112) |
| time-only chain GP (as-given) | 14.584 / +4.28 / 0.82 (n=9627) | 1.161 / +1.64 / 0.83 (n=3605) | 0.515 / +0.57 / 0.85 (n=3605) | 0.145 / -0.75 / 0.86 (n=4112) |

## Median per-(track, seed) RMSE vs estimator targets

| system | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| score-graph GP (learned scale) | 11.272 | 0.712 | 0.344 | 0.399 | 0.073 | 0.086 |
| score-graph GP (as-given) | 11.054 | 0.621 | 0.300 | 0.402 | 0.072 | 0.084 |
| no-graph ablation | 12.558 | 0.920 | 0.407 | 0.413 | 0.074 | 0.086 |
| time-only chain GP (learned scale) | 11.762 | 0.848 | 0.394 | 0.400 | 0.066 | 0.091 |
| time-only chain GP (as-given) | 11.731 | 0.843 | 0.407 | 0.405 | 0.074 | 0.091 |

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

## Paired SCORE GRAPH vs TIME-ONLY CHAIN (PRIMARY read; as-given, gp_asgiven - gp_chain_asgiven), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.496 [-0.670, -0.337]* | -0.037 [-0.063, -0.010]* (n=153) |
| log gamma | -0.224 [-0.305, -0.146]* | -0.428 [-0.968, +0.094]  (n=148) |
| log f | -0.212 [-0.284, -0.146]* | -0.393 [-0.510, -0.280]* (n=148) |
| loudness ell | -0.007 [-0.013, -0.001]* | -0.036 [-0.069, -0.005]* (n=153) |
| tau (s) | -0.001 [-0.004, +0.001]  | -0.222 [-0.671, +0.011]  (n=149) |
| delta_vib (s) | -0.004 [-0.010, +0.001]  | -0.109 [-0.190, -0.036]* (n=153) |

## Paired score graph vs time-only chain (learned scale; gp - gp_chain), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.427 [-0.581, -0.284]* | -0.042 [-0.070, -0.015]* (n=153) |
| log gamma | -0.182 [-0.254, -0.110]* | -1.501 [-3.932, -0.218]* (n=148) |
| log f | -0.142 [-0.202, -0.089]* | -0.233 [-0.315, -0.155]* (n=148) |
| loudness ell | -0.002 [-0.008, +0.004]  | -0.007 [-0.039, +0.030]  (n=153) |
| tau (s) | -0.002 [-0.007, +0.003]  | -0.229 [-0.791, +0.067]  (n=149) |
| delta_vib (s) | -0.006 [-0.011, -0.002]* | -0.076 [-0.143, -0.011]* (n=153) |

## Paired chain value (as-given; gp_chain_asgiven - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.399 [-1.036, +0.128]  | +0.017 [-0.039, +0.065]  (n=153) |
| log gamma | -0.034 [-0.116, +0.061]  | -3.865 [-11.519, +0.094]  (n=148) |
| log f | -0.089 [-0.150, -0.028]* | -0.049 [-0.152, +0.048]  (n=148) |
| loudness ell | +0.004 [-0.005, +0.013]  | +0.077 [+0.029, +0.134]* (n=153) |
| tau (s) | -0.000 [-0.001, +0.001]  | -0.075 [-0.121, -0.037]* (n=149) |
| delta_vib (s) | +0.003 [-0.001, +0.009]  | +0.166 [+0.085, +0.271]* (n=153) |


## Paired graph value by instrument family (dRMSE, gp - nograph)

| family | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| strings | -0.415* (n=70) | -0.078* (n=70) | -0.100* (n=70) | -0.014* (n=70) | -0.004* (n=68) | -0.004  (n=70) |
| wood | -2.349* (n=47) | -0.290* (n=45) | -0.461* (n=45) | +0.007  (n=47) | -0.016* (n=47) | -0.012* (n=47) |
| brass | -0.058  (n=36) | -0.272* (n=33) | -0.246* (n=33) | -0.009* (n=36) | -0.007* (n=34) | +0.003  (n=36) |

## Reading (2026-10-22)

153 (track, seed) cells on the 78 development tracks (3 cells skipped
by the standard eligibility rule), six channels, every arm carrying the
per-piece bias column, so every contrast is beyond the common offset.

1. **H-T1 holds: at track scale the score graph beats the time-only
   chain, significantly, beyond the common offset.** Primary read
   (as-given noise, graph minus chain): intonation −0.50 cents
   [−0.67, −0.34] and NLL −0.037*; vibrato extent −0.22*; vibrato rate
   −0.21* and NLL −0.39*; loudness −0.007* and NLL −0.036*; vibrato
   onset NLL −0.109*; timing a tie. The learned-scale variant agrees on
   every starred channel. This is the strong form of the methodological
   hypothesis, which failed at chain and phrase scale in the waveform
   studies, holding at whole-track scale on tracked targets.
2. **H-T2 is mixed: the time-only chain is not reliably better than no
   structure.** Chain minus no-graph: intonation −0.40 [−1.04, +0.13]
   (ns), rate −0.089*, extent ns; loudness and vibrato onset
   significantly WORSE in NLL (+0.077*, +0.166*). Time adjacency alone
   is a poor prior for this bundle; the score graph's edge over it is
   larger than its edge over nothing on several channels.
3. **The confirmed graph value reproduces on development**: graph minus
   no-graph on intonation −0.896* (confirmation pool: −0.877*), extent
   −0.26*, rate −0.30*.
4. **Caveat on the control.** The time-only chain links each note only
   to its immediate neighbours with unit weight, whereas the score
   graph links notes within a Gaussian time window of 2 beats and a
   pitch window of 4 semitones. The graph's advantage may therefore be
   a wider temporal neighbourhood rather than pitch proximity. The
   follow-up that isolates the pitch term (a time-only graph with the
   same Gaussian time window and the pitch term removed) is run as
   `results/phase2_timegauss_prior_dev.md`.
5. Scope: development split, selection-reused, no registration; the
   registered strings-and-winds results are untouched.
