# Phase 3 integration: the CURVE posterior as the waveform channel (DEV, exploratory, no claims)

Re-run of the August integration design (results/phase3_integration_dev.md) with the
realized-average read-out of the curve posterior (CurveFit.average, fitted-prior arm,
exact Laplace sd) replacing the scalar dev8 channel; same hold-out recipe, so cells
pair exactly across runs (base6 cross-run max |RMSE diff| = 0.00e+00, must be ~0).

n = 14 (track, seed) pairs; curve-source floor median 1.80 cents (scalar-source 3.09).

| system | vs estimator: RMSE / NLL / cov@90 | vs quasi-truth: RMSE / NLL / cov@90 |
|---|---|---|
| base6 | 10.874 / +3.752 / 0.87 | 11.740 / +3.983 / 0.81 |
| scalar wave_floor | 9.667 / +3.516 / 0.87 | 10.703 / +3.975 / 0.80 |
| curve_floor | 9.513 / +3.491 / 0.90 | 10.569 / +4.292 / 0.81 |
| curve_nofloor | 9.707 / +3.476 / 0.90 | 10.598 / +4.517 / 0.80 |

## Paired contrasts (negative favours the first)

- curve_floor vs base6 (the curve channel's value), est RMSE: -1.361 [-2.790, -0.328]* (better on 100%)
- curve_floor vs base6 (the curve channel's value), est NLL: -0.261 [-0.545, -0.061]* (better on 100%)
- curve_floor vs base6 (the curve channel's value), gt RMSE: -1.171 [-2.444, -0.244]* (better on 86%)
- curve_floor vs base6 (the curve channel's value), gt NLL: +0.308 [-0.258, +1.273]  (better on 79%)
- curve_floor vs scalar wave_floor (curve vs scalar read-out), est RMSE: -0.154 [-0.624, +0.353]  (better on 64%)
- curve_floor vs scalar wave_floor (curve vs scalar read-out), est NLL: -0.025 [-0.173, +0.133]  (better on 71%)
- curve_floor vs scalar wave_floor (curve vs scalar read-out), gt RMSE: -0.134 [-0.494, +0.304]  (better on 71%)
- curve_floor vs scalar wave_floor (curve vs scalar read-out), gt NLL: +0.317 [-0.116, +1.036]  (better on 71%)
- curve_floor vs curve_nofloor (the floor's value), est RMSE: -0.194 [-0.803, +0.190]  (better on 71%)
- curve_floor vs curve_nofloor (the floor's value), est NLL: +0.015 [-0.172, +0.178]  (better on 64%)
- curve_floor vs curve_nofloor (the floor's value), gt RMSE: -0.029 [-0.599, +0.437]  (better on 57%)
- curve_floor vs curve_nofloor (the floor's value), gt NLL: -0.226 [-0.599, +0.056]  (better on 71%)

## Reading (2026-10-09)

1. **The pairing is exact** (base6 cross-run max diff 0.00e+00): the GP
   fits are deterministic and the hold-out recipe reproduced, so the
   cross-run contrasts are genuinely paired.
2. **The curve channel reproduces the August integration value and
   slightly exceeds it**: est RMSE −1.361* (all 14 pairs; the scalar
   study measured −1.206*), est NLL −0.261*.
3. **Curve vs scalar read-out: a tie at n = 14, trend to the curve**
   (RMSE better on 64–71% of pairs, nothing significant). The committed
   question is answered "no significant difference, directionally
   better" — reported as measured.
4. **The mechanistic finding is the floor**: the per-(track, seed)
   discrepancy floor HALVES (3.09 → 1.80 cents median). The realized
   average of the inferred curve is the estimator's own functional, so
   it closes about half the estimand gap by construction — which also
   explains why the floor matters less here than in the scalar study
   (floor-vs-nofloor contrasts ns). Caution kept: vs quasi-truth NLL the
   tighter channel is non-significantly worse (+0.32), i.e. the smaller
   floor buys RMSE but gives back a little latent-band honesty.
5. **Dev default**: the curve read-out stands wherever the scalar stood
   (never significantly worse, floor halved, and it comes with the full
   curve for free); nothing registered is touched.
