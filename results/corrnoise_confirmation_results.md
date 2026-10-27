# Correlated-timing-noise CONFIRMATION — one shot on the RE-USED 13-piece pool, disclosed

> Registered protocol: docs/corrnoise_prereg_design.md, tag corrnoise-registration-2026-10-27. The pool was spent 2026-08-27 for claims C1-C4 and is re-used here with full disclosure: same archived inputs, same masks, same (track, seed) cells. Baseline rho = 0 is the published diagonal path. Primary N1 = paired tau NLL, correlated minus diagonal, cluster bootstrap over unique tracks. Run ONCE; every number reported whatever the outcome.

```
80 (track, seed) cells

chosen rho: median 0.45, rho>0 on 99% of cells, histogram: 0.00:1 0.15:10 0.30:21 0.45:26 0.60:11 0.75:8 0.90:3
tau nll : corr -1.8079 vs diag -1.6872  paired dmean -0.1208 [-0.1557,-0.0894]* (median d -0.0899)
tau cov : corr  0.9080 vs diag  0.9105  paired dmean -0.0025 [-0.0095,+0.0049]  (median d +0.0000)
tau rmse: corr  0.0473 vs diag  0.0583  paired dmean -0.0110 [-0.0156,-0.0073]* (median d -0.0064)
```

## VERDICT against the registered claims (scored 2026-10-27, run 17:59-18:24 +09:00, 25 min, 6 shards x OMP=4, code state 1603780 = tag corrnoise-registration-2026-10-27; decision rule verbatim from docs/corrnoise_prereg_design.md @ that tag)

- **N1 (timing calibration, PRIMARY) — PASS.** Paired τ ΔNLL, correlated minus diagonal: −0.1208 [−0.1557, −0.0894]\*, cluster bootstrap over the 40 unique tracks, B = 2000, rng 0. Negative mean, interval excludes zero. Development basis −0.111 [−0.142, −0.080]\* reproduced on the re-used pool.
- **N2 (timing recovery, SECONDARY, non-gating) — PASS.** Paired τ ΔRMSE −0.0110 [−0.0156, −0.0073]\* (0.0583 → 0.0473 s, an 19% cut; development −0.0192 on a noisier channel). The pre-stated risk (cleaner τ on this pool) materialised as a smaller absolute gain, not a lost star.
- **N3 (coverage boundary, report-only) — within [0.85, 0.95]:** correlated 0.908 (diagonal 0.911); paired −0.0025 [−0.0095, +0.0049], not significant.
- **N4 (detection, report-only):** ρ > 0 on 99% of cells (79 of 80), median ρ 0.45, histogram 0.00:1 0.15:10 0.30:21 0.45:26 0.60:11 0.75:8 0.90:3. Development 97–100%.
- Cells: 80 of 80 (track, seed) cells scored; none skipped by the fewer-than-three-held-out-τ rule.
- **Headline, by the registered rule:** the correlated timing-noise row repairs the failed timing-calibration claim of 2026-08-27, confirmed once on the re-used pool (disclosed). The pool's disclosure stands: these 80 cells were spent on claims C1–C4 and their plain fits were known when this claim was selected; the correlated-versus-diagonal contrast had never been computed on them.
- Evidence: evidence/phase2_reuse_confirmation/corrnoise/ (run log, shard logs, cell fragments, merged cells).
