# Phase 2 on real audio — first URMP dev results (77 unique tracks, 2 seeds, 30% of notes hidden)

## vs estimator targets (primary; predictive sd)

| system | c (cents) RMSE / cov@90 | log gamma RMSE / cov@90 | log f RMSE / cov@90 |
|---|---|---|---|
| graph GP (learned scale) | 14.081 / 0.90 (n=9629) | 1.509 / 0.92 (n=3275) | 0.439 / 0.89 (n=3275) |
| graph GP (as-given) | 14.123 / 0.89 (n=9629) | 1.491 / 0.91 (n=3275) | 0.466 / 0.88 (n=3275) |
| no-graph ablation | 14.374 / 0.90 (n=9629) | 1.558 / 0.92 (n=3275) | 0.580 / 0.90 (n=3275) |

## vs ground-truth-derived targets (quasi-truth; latent sd)

| system | c (cents) RMSE / cov@90 | log gamma RMSE / cov@90 | log f RMSE / cov@90 |
|---|---|---|---|
| graph GP (learned scale) | 14.022 / 0.80 (n=9627) | 1.038 / 0.78 (n=3605) | 0.461 / 0.74 (n=3605) |
| graph GP (as-given) | 14.075 / 0.83 (n=9627) | 1.031 / 0.86 (n=3605) | 0.469 / 0.86 (n=3605) |
| no-graph ablation | 14.291 / 0.82 (n=9627) | 1.171 / 0.81 (n=3605) | 0.585 / 0.76 (n=3605) |

## Median per-(track, seed) RMSE vs estimator targets

| system | c (cents) | log gamma | log f |
|---|---|---|---|
| graph GP (learned scale) | 11.360 | 0.704 | 0.346 |
| graph GP (as-given) | 11.307 | 0.718 | 0.354 |
| no-graph ablation | 12.408 | 0.921 | 0.378 |

## Paired graph value (gp - nograph), per (track, seed), vs estimator targets

| channel | dRMSE [95% CI] |
|---|---|
| c (cents) | -0.835 [-1.398, -0.378]* (n=153) |
| log gamma | -0.211 [-0.294, -0.131]* (n=148) |
| log f | -0.225 [-0.301, -0.155]* (n=148) |
