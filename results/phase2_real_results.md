# Phase 2 on real audio — first URMP dev results (77 unique tracks, 2 seeds, 30% of notes hidden)

## vs estimator targets (primary; predictive sd)

| system | c (cents) RMSE / cov@90 | log gamma RMSE / cov@90 | log f RMSE / cov@90 | loudness ell RMSE / cov@90 |
|---|---|---|---|---|
| graph GP (learned scale) | 14.042 / 0.90 (n=9629) | 1.512 / 0.92 (n=3275) | 0.461 / 0.89 (n=3275) | 0.414 / 0.90 (n=9629) |
| graph GP (as-given) | 14.049 / 0.89 (n=9629) | 1.501 / 0.91 (n=3275) | 0.444 / 0.88 (n=3275) | 0.419 / 0.91 (n=9629) |
| no-graph ablation | 14.380 / 0.90 (n=9629) | 1.556 / 0.92 (n=3275) | 0.582 / 0.89 (n=3275) | 0.430 / 0.90 (n=9629) |

## vs ground-truth-derived targets (quasi-truth; latent sd)

| system | c (cents) RMSE / cov@90 | log gamma RMSE / cov@90 | log f RMSE / cov@90 |
|---|---|---|---|
| graph GP (learned scale) | 13.969 / 0.81 (n=9627) | 1.068 / 0.80 (n=3605) | 0.480 / 0.75 (n=3605) |
| graph GP (as-given) | 13.995 / 0.83 (n=9627) | 1.059 / 0.86 (n=3605) | 0.451 / 0.86 (n=3605) |
| no-graph ablation | 14.286 / 0.82 (n=9627) | 1.169 / 0.80 (n=3605) | 0.586 / 0.76 (n=3605) |

## Median per-(track, seed) RMSE vs estimator targets

| system | c (cents) | log gamma | log f | loudness ell |
|---|---|---|---|---|
| graph GP (learned scale) | 11.185 | 0.756 | 0.360 | 0.399 |
| graph GP (as-given) | 11.251 | 0.721 | 0.351 | 0.402 |
| no-graph ablation | 12.323 | 0.887 | 0.390 | 0.411 |

## Paired graph value (gp - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] |
|---|---|
| c (cents) | -0.917 [-1.500, -0.430]* (n=153) |
| log gamma | -0.150 [-0.225, -0.075]* (n=148) |
| log f | -0.165 [-0.232, -0.110]* (n=148) |
| loudness ell | -0.010 [-0.015, -0.005]* (n=153) |

## Paired graph value by instrument family (dRMSE, gp - nograph)

| family | c (cents) | log gamma | log f | loudness ell |
|---|---|---|---|---|
| strings | -0.289* (n=70) | -0.079* (n=70) | -0.058* (n=70) | -0.017* (n=70) |
| wood | -2.382* (n=47) | -0.268* (n=45) | -0.351* (n=45) | +0.000  (n=47) |
| brass | -0.226* (n=36) | -0.139* (n=33) | -0.141* (n=33) | -0.010* (n=36) |
