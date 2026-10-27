# Tonal-metric CONFIRMATION — one shot on the RE-USED 13-piece pool, disclosed (40 unique tracks, 2 seeds, 30% of notes hidden)

> Registered protocol: docs/phase2_tonal_prereg_design.md, tag phase2-tonal-registration-2026-10-27. The pool was spent 2026-08-27 for claims C1-C4 and is re-used here with full disclosure: same archived inputs, same masks, same cells. Primary T2 = channel c NLL, as-given, paired gp_tonal_asgiven - gp_asgiven. Run ONCE; every number reported whatever the outcome.

## vs estimator targets (primary; predictive sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | loudness ell RMSE / NLL / cov@90 | tau (s) RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|---|---|
| graph GP (learned scale) | 12.006 / +3.74 / 0.88 (n=4095) | 1.378 / +7.25 / 0.91 (n=1486) | 0.375 / +0.37 / 0.90 (n=1486) | 0.401 / +0.46 / 0.90 (n=4095) | 0.067 / -1.73 / 0.90 (n=4016) | 0.187 / -0.68 / 0.89 (n=1920) |
| graph GP (as-given) | 12.058 / +3.77 / 0.88 (n=4095) | 1.367 / +3.03 / 0.89 (n=1486) | 0.381 / +0.30 / 0.89 (n=1486) | 0.403 / +0.46 / 0.90 (n=4095) | 0.065 / -1.72 / 0.91 (n=4016) | 0.194 / -0.63 / 0.88 (n=1920) |
| no-graph ablation | 12.709 / +3.77 / 0.89 (n=4095) | 1.457 / +8.39 / 0.91 (n=1486) | 0.628 / +0.52 / 0.87 (n=1486) | 0.405 / +0.46 / 0.90 (n=4095) | 0.063 / -1.69 / 0.91 (n=4016) | 0.186 / -0.60 / 0.90 (n=1920) |
| tonal graph GP (learned scale) | 11.842 / +3.72 / 0.88 (n=4095) | 1.375 / +5.73 / 0.90 (n=1486) | 0.372 / +0.35 / 0.90 (n=1486) | 0.401 / +0.47 / 0.89 (n=4095) | 0.068 / -1.72 / 0.90 (n=4016) | 0.187 / -0.64 / 0.89 (n=1920) |
| tonal graph GP (as-given) | 11.854 / +3.76 / 0.88 (n=4095) | 1.378 / +2.45 / 0.87 (n=1486) | 0.364 / +0.32 / 0.88 (n=1486) | 0.402 / +0.48 / 0.89 (n=4095) | 0.066 / -1.73 / 0.90 (n=4016) | 0.189 / -0.56 / 0.88 (n=1920) |

## vs ground-truth-derived targets (reference targets; latent sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|
| graph GP (learned scale) | 12.243 / +4.15 / 0.81 (n=4095) | 1.669 / +23.77 / 0.72 (n=1643) | 0.391 / +2.32 / 0.73 (n=1643) | 0.177 / -0.65 / 0.86 (n=2079) |
| graph GP (as-given) | 12.342 / +4.14 / 0.82 (n=4095) | 1.660 / +11.72 / 0.80 (n=1643) | 0.389 / +0.50 / 0.85 (n=1643) | 0.180 / -0.61 / 0.86 (n=2079) |
| no-graph ablation | 12.742 / +4.05 / 0.82 (n=4095) | 1.590 / +2.63 / 0.70 (n=1643) | 0.643 / +2.30 / 0.72 (n=1643) | 0.174 / -0.62 / 0.86 (n=2079) |
| tonal graph GP (learned scale) | 12.085 / +4.17 / 0.80 (n=4095) | 1.652 / +21.61 / 0.71 (n=1643) | 0.386 / +2.30 / 0.72 (n=1643) | 0.181 / -0.67 / 0.85 (n=2079) |
| tonal graph GP (as-given) | 12.138 / +4.15 / 0.82 (n=4095) | 1.649 / +11.47 / 0.80 (n=1643) | 0.379 / +0.54 / 0.83 (n=1643) | 0.181 / -0.63 / 0.86 (n=2079) |

## Median per-(track, seed) RMSE vs estimator targets

| system | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| graph GP (learned scale) | 10.575 | 0.575 | 0.361 | 0.352 | 0.046 | 0.102 |
| graph GP (as-given) | 10.793 | 0.547 | 0.329 | 0.349 | 0.047 | 0.100 |
| no-graph ablation | 11.412 | 1.097 | 0.525 | 0.360 | 0.047 | 0.101 |
| tonal graph GP (learned scale) | 10.965 | 0.572 | 0.353 | 0.357 | 0.047 | 0.104 |
| tonal graph GP (as-given) | 10.829 | 0.584 | 0.309 | 0.351 | 0.047 | 0.107 |

## Paired graph value (learned scale; gp - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.931 [-1.275, -0.606]* | -0.063 [-0.098, -0.031]* (n=80) |
| log gamma | -0.406 [-0.549, -0.274]* | -0.776 [-2.261, +0.459]  (n=78) |
| log f | -0.408 [-0.527, -0.297]* | -0.462 [-0.612, -0.322]* (n=78) |
| loudness ell | -0.002 [-0.010, +0.007]  | +0.014 [-0.020, +0.051]  (n=80) |
| tau (s) | +0.003 [+0.000, +0.007]* | -0.042 [-0.065, -0.017]* (n=80) |
| delta_vib (s) | +0.002 [-0.004, +0.009]  | -0.077 [-0.289, +0.053]  (n=78) |

## Paired graph value (as-given, the default; gp_asgiven - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.877 [-1.236, -0.538]* | -0.029 [-0.066, +0.006]  (n=80) |
| log gamma | -0.425 [-0.565, -0.295]* | -2.990 [-6.575, -0.435]* (n=78) |
| log f | -0.429 [-0.556, -0.312]* | -0.564 [-0.739, -0.392]* (n=78) |
| loudness ell | -0.002 [-0.009, +0.005]  | +0.015 [-0.011, +0.044]  (n=80) |
| tau (s) | +0.003 [+0.001, +0.007]* | -0.030 [-0.061, +0.006]  (n=80) |
| delta_vib (s) | +0.006 [-0.001, +0.014]  | +0.065 [-0.192, +0.331]  (n=78) |

## Paired TONAL vs PLAIN metric (PRIMARY read; as-given, gp_tonal_asgiven - gp_asgiven), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.312 [-0.599, -0.081]* | -0.023 [-0.049, +0.001]  (n=80) |
| log gamma | +0.029 [-0.028, +0.087]  | -0.290 [-0.709, +0.006]  (n=78) |
| log f | -0.023 [-0.055, +0.004]  | +0.029 [-0.053, +0.123]  (n=78) |
| loudness ell | +0.000 [-0.005, +0.005]  | +0.020 [+0.003, +0.041]* (n=80) |
| tau (s) | +0.000 [-0.001, +0.002]  | -0.002 [-0.011, +0.007]  (n=80) |
| delta_vib (s) | -0.000 [-0.007, +0.006]  | +0.050 [-0.057, +0.185]  (n=78) |

## Paired tonal vs plain metric (learned scale; gp_tonal - gp), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -0.262 [-0.607, -0.008]* | -0.025 [-0.054, +0.002]  (n=80) |
| log gamma | -0.006 [-0.032, +0.019]  | -0.800 [-1.826, -0.040]* (n=78) |
| log f | -0.013 [-0.042, +0.015]  | -0.035 [-0.105, +0.032]  (n=78) |
| loudness ell | -0.000 [-0.005, +0.004]  | +0.007 [-0.011, +0.024]  (n=80) |
| tau (s) | +0.001 [-0.001, +0.004]  | +0.020 [+0.003, +0.038]* (n=80) |
| delta_vib (s) | +0.002 [-0.002, +0.007]  | -0.012 [-0.190, +0.137]  (n=78) |

## Paired tonal graph value (as-given; gp_tonal_asgiven - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] | dNLL [95% CI] |
|---|---|---|
| c (cents) | -1.189 [-1.654, -0.778]* | -0.053 [-0.099, -0.010]* (n=80) |
| log gamma | -0.396 [-0.533, -0.265]* | -3.280 [-7.372, -0.436]* (n=78) |
| log f | -0.452 [-0.582, -0.333]* | -0.535 [-0.730, -0.334]* (n=78) |
| loudness ell | -0.002 [-0.010, +0.006]  | +0.036 [+0.007, +0.069]* (n=80) |
| tau (s) | +0.004 [+0.001, +0.008]* | -0.032 [-0.061, +0.002]  (n=80) |
| delta_vib (s) | +0.005 [+0.001, +0.011]* | +0.115 [-0.059, +0.354]  (n=78) |


## Paired graph value by instrument family (dRMSE, gp - nograph)

| family | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| strings | -0.381* (n=18) | -0.053  (n=18) | -0.081* (n=18) | -0.013* (n=18) | -0.001  (n=18) | +0.002  (n=18) |
| wood | -0.304  (n=28) | -0.138* (n=26) | -0.184* (n=26) | -0.008  (n=28) | +0.004  (n=28) | +0.001  (n=27) |
| brass | -1.738* (n=34) | -0.797* (n=34) | -0.752* (n=34) | +0.009  (n=34) | +0.005  (n=34) | +0.003  (n=33) |

## VERDICT against the registered claims (scored 2026-10-27, run 18:25-20:28 +09:00, 2 h 3 m, 6 shards x OMP=4, code state cf1b48e with the frozen files identical to tag phase2-tonal-registration-2026-10-27; decision rule verbatim from docs/phase2_tonal_prereg_design.md @ that tag)

- **T2 (intonation calibration, PRIMARY) — FAIL.** as-given paired dNLL on c,
  tonal minus plain (gp_tonal_asgiven - gp_asgiven, B = 2000, rng 31):
  **-0.023 [-0.049, +0.001]** — negative mean but the 95% interval includes 0
  (development basis -0.050 [-0.072, -0.029]*). Reported verbatim per the
  rule: the circle-of-fifths metric stays exploratory, the appendix keeps its
  status, no adoption.
- **T1 (intonation recovery, SECONDARY, non-gating) — PASS.** as-given paired
  dRMSE on c: **-0.312 [-0.599, -0.081]*** (development -0.213*; registered as
  underpowered at 0.52-0.63 and starred anyway). Non-gating: it does not rescue
  T2.
- **T3 (timing penalty, report-only) — the predicted penalty did not appear**
  in the registered variant: as-given tau dNLL -0.002 [-0.011, +0.007] ns
  (development +0.027*). It appears in the learned-scale variant only, tau dNLL
  +0.020 [+0.003, +0.038]* (report-only, not a claim).
- **Identity check (pre-stated) — PASSED.** All 24 plain-arm rows (gp,
  gp_asgiven, nograph in the estimator, reference, median and family tables;
  both paired graph-value tables) reproduce results/phase2_confirmation_results.md
  digit for digit. The 2026-08-27 confirmation numbers are untouched by this run.
- Cells: 80 of 80 (track, seed) cells scored, five systems each.

**Decision rule: T2 fails ⇒ the thesis does NOT state that the tonal metric
improves intonation calibration. The one registered claim on this metric
is an honest miss on a pool where the development interval was
[-0.072, -0.029] and the confirmation interval is [-0.049, +0.001].**

Honest cells, reported next to the claims (one-shot; no reruns, no added
seeds, no post-hoc filters):
- ell dNLL +0.020 [+0.003, +0.041]* — a small starred ADVERSE calibration cell
  on loudness under the tonal metric (0.02 nats on a channel the metric does not
  touch directly; the coupling carries it).
- Tonal graph vs no-graph, as-given: c dNLL -0.053 [-0.099, -0.010]* — the
  tonal graph beats the no-graph ablation on intonation calibration where the
  plain graph did not on 2026-08-27 (-0.029 ns). Report-only: no such claim was
  registered, and the registered tonal-vs-plain contrast is the one that failed.
- Coverage@90 under the tonal metric, as-given, vs estimator targets: c 0.88,
  log gamma 0.87, log f 0.88, ell 0.89, tau 0.90, delta_vib 0.88 — all within
  [0.85, 0.95]; no coverage claim was registered for this metric.
- Pool disclosure: these 80 cells were spent on claims C1-C4 (2026-08-27) and
  their plain fits were known when T1-T3 were selected; the tonal arms had
  never been computed on them. Evidence: evidence/phase2_reuse_confirmation/tonal/.
