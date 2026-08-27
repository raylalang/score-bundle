# Phase 3 integration: the waveform as a 7th bundle channel (DEV)

**DEV ONLY — EXPLORATORY.** Design + rationale in the script docstring (`scripts/eval_phase3_integration.py`); thesis context sec:phase3-study. n = 14 (track, seed) pairs over 7 tracks; discrepancy floor median 3.09 cents.

## Intonation channel c at held-out notes

| system | vs estimator: RMSE / NLL / cov@90 | vs quasi-truth: RMSE / NLL / cov@90 |
|---|---|---|
| base6 | 10.874 / +3.752 / 0.87 | 11.740 / +3.983 / 0.81 |
| wave_floor | 9.667 / +3.516 / 0.87 | 10.703 / +3.975 / 0.80 |
| wave_nofloor | 9.739 / +3.501 / 0.86 | 10.629 / +4.199 / 0.77 |

## Paired contrasts (negative favours the first)

- wave_floor vs base6, est RMSE: -1.206 [-2.290, -0.327]* (better on 100%)  <!-- the integration's value -->
- wave_floor vs base6, est NLL: -0.236 [-0.428, -0.060]* (better on 93%)  <!-- the integration's value -->
- wave_floor vs base6, gt RMSE: -1.037 [-2.030, -0.229]* (better on 86%)  <!-- the integration's value -->
- wave_floor vs base6, gt NLL: -0.009 [-0.214, +0.272]  (better on 71%)  <!-- the integration's value -->
- wave_floor vs wave_nofloor, est RMSE: -0.072 [-0.505, +0.324]  (better on 71%)  <!-- the floor's value (calibration) -->
- wave_floor vs wave_nofloor, est NLL: +0.016 [-0.143, +0.173]  (better on 50%)  <!-- the floor's value (calibration) -->
- wave_floor vs wave_nofloor, gt RMSE: +0.075 [-0.381, +0.590]  (better on 50%)  <!-- the floor's value (calibration) -->
- wave_floor vs wave_nofloor, gt NLL: -0.224 [-0.496, -0.016]* (better on 57%)  <!-- the floor's value (calibration) -->

## Reading

1. **The integration works.** The waveform channel improves held-out
   intonation recovery on 14 of 14 (track, seed) pairs: dRMSE −1.21
   cents* (10.87 → 9.67, ~11%), dNLL −0.24* vs estimator targets,
   coverage held at 0.87; the gain persists against quasi-truth
   (dRMSE −1.04*). The evidence learns the wave↔c coupling per piece
   (B is 7×7); nothing else changed.
2. **The floor is what keeps it honest.** Against quasi-truth the
   no-floor control is significantly worse-calibrated (dNLL −0.22* in
   the floor's favour; coverage 0.80 vs 0.77): with only its tiny
   posterior variance, the GP over-trusts the waveform measurement —
   precisely the failure mode the Phase-3 study predicted. On
   estimator-target metrics the two are indistinguishable (the
   predictive sd absorbs the difference), so the floor costs nothing
   where it isn't needed.
3. **Scope honesty:** n = 14 pairs, 7 tracks, one channel, dev only;
   the floor is calibrated per (track, seed) from visible notes only
   (median 3.09 cents — the estimand-gap scale, found automatically).
   The fusion design observes the wave channel at held-out notes,
   which is the deployment situation (audio is always available) and is
   the disclosed point of the study. Any claim requires its own
   registration; this is the feasibility evidence for it.
