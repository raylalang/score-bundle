# Phase 2 CONFIRMATION — one-shot on the frozen 13-piece pool (40 unique tracks, 2 seeds, 30% of notes hidden)

> Registered protocol: docs/phase2_prereg_design.md, tag phase2-registration-2026-08-17. Run ONCE; every number reported whatever the outcome.

## vs estimator targets (primary; predictive sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | loudness ell RMSE / NLL / cov@90 | tau (s) RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|---|---|
| graph GP (learned scale) | 12.006 / +3.74 / 0.88 (n=4095) | 1.378 / +7.25 / 0.91 (n=1486) | 0.375 / +0.37 / 0.90 (n=1486) | 0.401 / +0.46 / 0.90 (n=4095) | 0.067 / -1.73 / 0.90 (n=4016) | 0.187 / -0.68 / 0.89 (n=1920) |
| graph GP (as-given) | 12.058 / +3.77 / 0.88 (n=4095) | 1.367 / +3.03 / 0.89 (n=1486) | 0.381 / +0.30 / 0.89 (n=1486) | 0.403 / +0.46 / 0.90 (n=4095) | 0.065 / -1.72 / 0.91 (n=4016) | 0.194 / -0.63 / 0.88 (n=1920) |
| no-graph ablation | 12.709 / +3.77 / 0.89 (n=4095) | 1.457 / +8.39 / 0.91 (n=1486) | 0.628 / +0.52 / 0.87 (n=1486) | 0.405 / +0.46 / 0.90 (n=4095) | 0.063 / -1.69 / 0.91 (n=4016) | 0.186 / -0.60 / 0.90 (n=1920) |

## vs ground-truth-derived targets (quasi-truth; latent sd)

| system | c (cents) RMSE / NLL / cov@90 | log gamma RMSE / NLL / cov@90 | log f RMSE / NLL / cov@90 | delta_vib (s) RMSE / NLL / cov@90 |
|---|---|---|---|---|
| graph GP (learned scale) | 12.243 / +4.15 / 0.81 (n=4095) | 1.669 / +23.77 / 0.72 (n=1643) | 0.391 / +2.32 / 0.73 (n=1643) | 0.177 / -0.65 / 0.86 (n=2079) |
| graph GP (as-given) | 12.342 / +4.14 / 0.82 (n=4095) | 1.660 / +11.72 / 0.80 (n=1643) | 0.389 / +0.50 / 0.85 (n=1643) | 0.180 / -0.61 / 0.86 (n=2079) |
| no-graph ablation | 12.742 / +4.05 / 0.82 (n=4095) | 1.590 / +2.63 / 0.70 (n=1643) | 0.643 / +2.30 / 0.72 (n=1643) | 0.174 / -0.62 / 0.86 (n=2079) |

## Median per-(track, seed) RMSE vs estimator targets

| system | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| graph GP (learned scale) | 10.575 | 0.575 | 0.361 | 0.352 | 0.046 | 0.102 |
| graph GP (as-given) | 10.793 | 0.547 | 0.329 | 0.349 | 0.047 | 0.100 |
| no-graph ablation | 11.412 | 1.097 | 0.525 | 0.360 | 0.047 | 0.101 |

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


## Paired graph value by instrument family (dRMSE, gp - nograph)

| family | c (cents) | log gamma | log f | loudness ell | tau (s) | delta_vib (s) |
|---|---|---|---|---|---|---|
| strings | -0.381* (n=18) | -0.053  (n=18) | -0.081* (n=18) | -0.013* (n=18) | -0.001  (n=18) | +0.002  (n=18) |
| wood | -0.304  (n=28) | -0.138* (n=26) | -0.184* (n=26) | -0.008  (n=28) | +0.004  (n=28) | +0.001  (n=27) |
| brass | -1.738* (n=34) | -0.797* (n=34) | -0.752* (n=34) | +0.009  (n=34) | +0.005  (n=34) | +0.003  (n=33) |

## VERDICT against the registered claims (scored 2026-08-27, run 04:25-05:46 +09:00, 1 h 21 m, 8 shards x OMP=4, code state 5144135; decision rule verbatim from docs/phase2_prereg_design.md @ tag phase2-registration-2026-08-17)

- **C1 (recovery, intonation) — PASS.** as-given paired dRMSE on c:
  **-0.877 [-1.236, -0.538]*** (negative, starred; development basis -0.891*).
- **C2 (calibration, vibrato) — PASS.** as-given paired dNLL:
  **log gamma -2.990 [-6.575, -0.435]***; **log f -0.564 [-0.739, -0.392]***
  (both channels starred; the extent half was the pre-flagged seed-sensitive
  risk and held).
- **C3 (coverage) — PASS.** as-given coverage@90 vs estimator targets:
  c 0.88, log gamma 0.89, log f 0.89, ell 0.90, tau 0.91, delta_vib 0.88 —
  all six within [0.85, 0.95].
- **C4 (timing calibration, secondary) — FAIL.** as-given paired dNLL on tau:
  -0.030 [-0.061, +0.006] — negative but the CI includes 0 (development basis
  -0.292*). Reported verbatim per the rule; C4 does not gate C1-C3.

**Decision rule: C1, C2, C3 all pass ⇒ the Phase-2 headline — "the unchanged
graph prior extends to real audio with calibrated uncertainty, at
confirmation level" — is CONFIRMED.**

Honest cells, reported next to the claims (one-shot; no reruns, no added
seeds, no post-hoc filters):
- tau dRMSE **+0.003 [+0.000, +0.007]*** — a starred ADVERSE recovery cell
  (~3 ms on an RMSE of 65 ms). No recovery claim was registered for tau; the
  registered calibration claim on it (C4) failed as above, so the graph's
  value on the adopted timing channel did not confirm on this pool.
- ell dNLL +0.015 [-0.011, +0.044] ns — the development set's starred adverse
  ell cell (+0.042*) did NOT replicate as significant.
- delta_vib graph-neutral as registered (dRMSE +0.006 ns, dNLL +0.065 ns);
  no claim was made and none is claimed now.
- Quasi-truth ordering unchanged: as-given remains the better-calibrated
  variant (vibrato coverage 0.80/0.85 vs the learned scale's 0.72/0.73);
  no-graph shows lower quasi-truth gamma RMSE (1.590 vs 1.660) at collapsed
  coverage (0.70) — the same accuracy-without-calibration trade recorded in
  development.
- All 40 unique tracks warped (15 exact, 25 DTW, 0 failed) — no missing-tau
  tracks on this pool (development had 2/78).
