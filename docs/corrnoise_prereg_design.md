# Correlated-timing-noise confirmation design — REGISTRATION (2026-10-27)

> **Status: REGISTERED as of the commit that carries this banner (tag
> `corrnoise-registration-2026-10-27`).** This committed revision freezes
> the claim, the decision rule, the systems, the inputs and the exact
> commands BEFORE the run. The run is a separate, deliberate act,
> performed ONCE, with every number reported whatever the outcome.
> Pool: the 13-piece URMP confirmation pool, spent 2026-08-27 for claims
> C1–C4 and RE-USED HERE WITH FULL DISCLOSURE (section "The pool").
> (Document history: drafted 2026-09-16 as a proposal; power check run
> 2026-09-16; pool decided 2026-10-23 by Ray, the professor hands-off;
> frozen 2026-10-27. After the run this file is never edited except by a
> dated erratum block.)

## What is claimed

The development result (`results/corrnoise_tau_dev.md`; three independent
runs; all six contrasts survive the week-wide BH pass,
`logs/week_bh_recheck.log`): an AR(1)-correlated noise block on the
timing channel, its one parameter chosen per (track, seed) by the
observed-block evidence, improves timing calibration and recovery over
the published diagonal noise row. It is the named remedy for the one
registered claim that failed on 2026-08-27 (C4, timing calibration).

- **N1 (timing calibration, PRIMARY).** Paired ΔNLL on held-out τ cells,
  correlated minus diagonal, negative with the 95% cluster-bootstrap
  interval excluding zero. Development basis: −0.111 [−0.142, −0.080]\*
  (seeds 0/1); −0.108\* (fresh seeds 2/3); −0.101\* (joint refit).
- **N2 (timing recovery, SECONDARY, non-gating).** Paired ΔRMSE on
  held-out τ cells, negative with the interval excluding zero.
  Development basis: −0.0192\* / −0.0205\* / −0.0199\*.
- **N3 (honest boundary, report-only).** Coverage at nominal 90% under
  the correlated predictive stays in [0.85, 0.95] (development: 0.90 in
  all three runs). Gates nothing.
- **N4 (detection, report-only).** The evidence-chosen ρ is positive on a
  majority of cells (development: 97–100%). Gates nothing.

## Decision rule (verbatim, frozen)

N1 PASSES iff the paired τ ΔNLL (correlated minus diagonal, held-out τ
cells, profile mode) has a negative mean AND its 95% cluster-bootstrap
interval (clusters = unique tracks, B = 2000, `numpy.random.default_rng(0)`,
percentile) excludes zero. N1 pass ⇒ the thesis states: "the correlated
timing-noise row repairs the failed timing-calibration claim, confirmed
once on the re-used pool (disclosed)". N1 fail ⇒ reported verbatim;
`gp.noise_corr` stays opt-in; the thesis keeps the result as a
development finding. N2: the same test on ΔRMSE, reported, non-gating.
N3, N4: reported as numbers. No post-hoc filters, no added seeds, no
reruns. Cells with fewer than three held-out τ observations are skipped
by the script's existing rule (`h_idx.size < 3`), and the count of
skipped cells is reported.

## The pool: disclosed re-use of the spent 13-piece URMP pool

- Composition: `phase2/splits.py` `CONF_PIECES = (9, 14, 15, 16, 24, 25,
  28, 29, 30, 32, 33, 34, 43)`, 7 compositions, 13 pieces, 40 unique
  tracks (17 brass, 14 woodwind, 9 strings), 80 (track, seed) cells.
- Inputs: `evidence/phase2_confirmation/urmp_targets_conf_inputs.pkl`
  (md5 `6a44d902589a1a0369f551cea1b991a9`), the archived post-`delta`
  targets of 2026-08-27, copied verbatim to `.cache/urmp_targets_conf.pkl`
  by the runner and md5-checked; no extraction stage is re-run. Masks are
  reproduced exactly (`default_rng(1000 + 7·piece + track + seed)`), so
  the cells and the hidden notes are those of 2026-08-27.
- **Disclosure.** Replication-grade, not pristine: this pool was spent on
  2026-08-27 for claims C1–C4. Its targets, the plain-graph fits, the
  per-family contrasts, the failed timing-calibration claim (τ ΔNLL
  −0.030, not significant) and the adverse τ recovery (+0.003\*) are in
  `results/phase2_confirmation_results.md` and in the thesis, and were
  known when this claim was selected. The claim-specific quantities (the
  evidence-chosen ρ and the correlated-versus-diagonal contrast) have
  never been computed on these tracks: `scripts/eval_corrnoise_tau.py`
  had no confirmation-mode path before the commit carrying this banner,
  and the confirmation caches were absent until the runner copies the
  archive. The claim and every protocol constant were developed on the
  31-piece development split only. The real Bach10 corpus was not
  pursued this cycle (the archive on disk is the Sibelius-resynthesised
  version, synthetic audio, unusable for timing or intonation claims).
  The thesis labels the result "confirmation on the re-used pool
  (disclosed)".
- **Where this pool differs from development, pre-stated.** Its timing
  channel is cleaner than development (median per-track τ standard
  deviation 50 ms against 79 ms; as-given τ RMSE 0.065 s against
  0.109 s; diagonal τ NLL already −1.72 against −1.63), so the
  correlated gain may be smaller here; N2 is the claim most at risk. The
  power table below assumed development-like noise.
- **Multiplicity.** A second claim (the tonal metric,
  `docs/phase2_tonal_prereg_design.md`) is registered on the same pool
  in the same week, on a different channel by a different mechanism. No
  cross-registration correction is applied; both are reported.

**Power check (RUN 2026-09-16, `scripts/power_corrnoise.py`,
`logs/power_corrnoise.log`).** Piece-level subsampling of the 149
development cells (31 pieces, median 4 cells/piece), P(95% cluster CI
stars) per pool size:

| pieces | P(N1 τ NLL stars) | P(N2 τ RMSE stars) |
|---|---|---|
| 4 | 0.92 | 0.76 |
| 6 | 0.99 | 0.76 |
| 8 | 1.00 | 0.81 |
| 10 | 1.00 | 0.83 |
| **13 (this pool)** | **1.00** | **0.94** |
| 15 | 1.00 | 0.99 |
| 20 | 1.00 | 1.00 |

## Protocol constants (frozen)

- System: the registered as-given six-channel pipeline verbatim, as
  executed by `scripts/eval_corrnoise_tau.py` (the script that produced
  the development basis; it imports its constants from
  `scripts/eval_phase2_real.py`): HOLD_FRAC 0.30, SEEDS (0, 1), MIN_NOTES
  30, C_MAX 150, identical masks for both systems.
- The correlated block: Σ_τ(ρ)_ij = √(d_i d_j) ρ^|i−j| in note order, with
  d_i the as-given τ cell variances; ρ per (track, seed) by the
  observed-block marginal likelihood at the fitted hyperparameters on
  the fixed grid (0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9): **profile mode**.
  The joint refit (`--joint`) and fresh seeds (`--seeds`) are development
  robustness checks and are refused in confirmation mode.
- Predictions for held-out τ through the full covariance T = C + Σ_n
  (noise cross-terms included). The adopted core equivalents
  `gp.noise_corr` / `gp.posterior_observations` are pinned bit-equal to
  this machinery by `tests/test_adoption_optin.py`.
- Baseline: ρ = 0, same fit, the published diagonal path.
- Scoring: paired per (track, seed); cluster bootstrap over unique tracks
  (what `report()` does; the proposal's "over pieces" was corrected at
  registration to match the code that produced the development basis),
  B = 2000, 95% percentile interval; every number reported; one shot.
- Determinism: OMP_NUM_THREADS=4, 6 shards (fits drift in the 4th
  decimal across BLAS thread counts; the development basis was at
  OMP_NUM_THREADS=2).
- Code state: the commit carrying this banner (tag
  `corrnoise-registration-2026-10-27`); the runner refuses if any frozen
  file differs from the tag.

## Exact commands

```
export PHASE2_SPLIT=confirmation PHASE2_CONFIRMATION_I_AM_SURE=yes PHASE2_REUSE_I_AM_SURE=yes
bash scripts/run_phase2_reuse_confirmation.sh corrnoise 6
# = md5 check of the inputs; 6 x `python scripts/eval_corrnoise_tau.py run --shard K/6`
#   at OMP_NUM_THREADS=4; `python scripts/eval_corrnoise_tau.py report --md results/corrnoise_confirmation_results.md`
```

What is reported, pass or fail: the full `report()` output (τ NLL,
coverage, RMSE for both rows; paired Δ with interval and star; median
and histogram of ρ; cell count), the verdict block appended to
`results/corrnoise_confirmation_results.md` in the 2026-08-27 format,
and the evidence (run log, shard logs, cell fragments) archived under
`evidence/phase2_reuse_confirmation/corrnoise/` in the same commit.

## Pre-registration checklist

- [x] Development evidence recorded and hardened (three runs; BH pass)
- [x] Power check run and claims ordered by it
- [x] Core code path pinned by tests (ρ=0 bit-equality; brute-force
      conditioning)
- [x] Pool decided: disclosed re-use of the spent URMP pool (Ray,
      2026-10-23)
- [x] Bach10 / new-corpus items: N/A for re-use (same corpus, same
      estimator chain, same archived targets; tracker calibration =
      `results/tracker_calibration_dev.md`, already in the registered
      design)
- [x] Guarded one-shot runner (`scripts/run_phase2_reuse_confirmation.sh`,
      triple env gate, one-shot lock, frozen-state check, md5 check) with
      the refusal test `tests/test_phase2_reuse_guard.py`
- [x] Claims file frozen in a dedicated commit + tag
      (`corrnoise-registration-2026-10-27`)
