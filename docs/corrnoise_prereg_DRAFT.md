# DRAFT — correlated-timing-noise confirmation design (NOT REGISTERED)

> **Status: PROPOSAL ONLY (2026-09-16).** Nothing here is registered.
> Registration is a separate act: a dedicated commit + tag, made before
> any confirmation data is touched, after the professor's corpus/pool
> decision. This draft exists so that the meeting can decide a pool and
> the registration can follow the same day.

## What would be claimed

The development result (`results/corrnoise_tau_dev.md`; three independent
runs; all six contrasts survive the week-wide BH pass,
`logs/week_bh_recheck.log`): an AR(1)-correlated noise block on the
timing channel, its one parameter chosen per piece by the observed-block
evidence, improves timing calibration and recovery over the published
diagonal noise row.

- **N1 (timing calibration, PRIMARY).** Paired ΔNLL on held-out τ cells,
  correlated minus diagonal, negative and starred (95% cluster-bootstrap
  CI over pieces excludes zero). Development basis: −0.111
  [−0.142, −0.080]\* (seeds 0/1); −0.108\* (fresh seeds 2/3); −0.101\*
  (joint refit). Directly answers the failed registered claim C4.
- **N2 (timing recovery, SECONDARY).** Paired ΔRMSE on held-out τ cells,
  negative and starred. Development basis: −0.0192\* / −0.0205\* /
  −0.0199\* across the three runs. Does not gate N1.
- **N3 (honest boundary, pre-stated).** Coverage at nominal 90% stays in
  [0.85, 0.95] under the correlated predictive (development: 0.90 across
  all three runs, moving toward nominal from the diagonal's 0.91–0.92
  mild over-coverage). Reported; gates nothing.
- **N4 (detection, descriptive).** The evidence-chosen ρ is positive on a
  majority of cells (development: 97–100%). Reported; gates nothing.

Ordering is set by the power check below: N1 is powered even on a small
pool, N2 is not — the tonal draft's calibration-primary move, repeated.

## The corpus decision (the open question — professor's call)

URMP is fully partitioned (13-piece confirmation pool SPENT 2026-08-27;
the other 31 pieces are development — `phase2/splits.py`). The same three
options as the tonal draft, and the same pool could serve both
registrations in one shot:

1. **A new corpus (preferred if available).** Requirements for the τ
   claim are modest: per-track monophonic audio + note-level onset
   annotations (for the leave-one-out warp that defines τ and its noise
   row); the other bundle channels additionally want ground-truth or
   trackable pitch. **Bach10 readiness (checked 2026-09-16):** the AIR
   lab page (labsites.rochester.edu/air/resource.html) confirms the
   dataset carries everything needed and more — audio of each part AND
   the ensemble for 10 four-part Bach chorales (violin, clarinet,
   saxophone, bassoon), MIDI scores, ground-truth audio–score alignment,
   ground-truth pitch per part, and ground-truth notes. At 4 stems ×
   2 seeds ≈ 8 cells/piece it sits above the 10-piece power row. The
   download is gated behind the lab's request form (the "Dataset
   Download" Google form linked from the page — a manual step, like
   URMP's Dryad gate): Ray fills the form, the corpus lands in
   `../data/bach10/`, then the URMP preparatory sequence applies
   (loader, composition-level split frozen data-blind, tracker
   calibration) before the tag.
2. **Disclosed re-use of the spent URMP pool.** Replication-grade, not
   pristine: the pool was spent on the Phase-2 claim set, which did not
   involve ρ, but the tracks are no longer untouched-by-any-decision.
   Would be labeled verbatim as re-use.
3. **Both:** register on the new corpus, report the spent pool as a
   labeled replication appendix.

**Power check (RUN 2026-09-16, `scripts/power_corrnoise.py`,
`logs/power_corrnoise.log`).** Piece-level subsampling of the 149
development cells (31 pieces, median 4 cells/piece), P(95% cluster CI
stars) per pool size:

| pieces | P(N1 τ NLL stars) | P(N2 τ RMSE stars) |
|---|---|---|
| 4 | 0.92 | 0.76 |
| 6 | 0.99 | 0.76 |
| 8 | 1.00 | 0.81 |
| 10 (≈ Bach10) | 1.00 | 0.83 |
| 13 (≈ spent pool) | 1.00 | 0.94 |
| 15 | 1.00 | 0.99 |
| 20 | 1.00 | 1.00 |

Calibration caveat: a corpus with more cells per piece sits above its row
(Bach10: 4 stems × 2 seeds ≈ 8 cells/piece), fewer sits below.

**Recommendation from the power check:** N1 primary (powered at ≥0.99
from 6 pieces on every option); N2 secondary and reported whatever it
does; on a 10-piece pool N2 is underpowered (~0.83) and this is stated at
registration rather than discovered after.

## Protocol constants (would be frozen at registration)

- System: the registered as-given six-channel pipeline verbatim
  (`scripts/eval_phase2_real.py` machinery), HOLD_FRAC 0.30, two seeds,
  identical masks for both systems.
- The correlated block: `gp.noise_corr` (core path, bit-equal to the
  study machinery, pinned by `tests/test_adoption_optin.py`);
  ρ per (track, seed) by the observed-block profile on the fixed grid
  (0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9) at the fitted hyperparameters —
  the joint refit is a robustness appendix, not the claim path.
- Predictions for held-out τ via `gp.posterior_observations` (the full
  covariance incl. noise cross-terms).
- Baseline: ρ = 0, same fit — the published diagonal path.
- Scoring: paired per (track, seed), cluster bootstrap over pieces,
  B = 2000, 95% percentile CI; every number reported; one shot.
- OMP_NUM_THREADS pinned at run registration (BLAS determinism, as in
  the Phase-2 confirmation).

## Pre-registration checklist (before the tag)

- [x] Development evidence recorded and hardened (three runs; BH pass)
- [x] Power check run and claims ordered by it
- [x] Core code path pinned by tests (ρ=0 bit-equality; brute-force
      conditioning)
- [ ] Corpus/pool decided with the professor
- [ ] If Bach10: request via the AIR lab form (manual; contents verified
      against the lab page 2026-09-16 — per-part audio, GT pitch, GT
      notes, MIDI scores, alignment)
- [ ] If a new corpus: loader + annotations verified, split frozen at
      composition level, tracker calibrated (the URMP preparatory steps)
- [ ] Claims file frozen in a dedicated commit + tag
      (`corrnoise-registration-YYYY-MM-DD`)
- [ ] Guarded one-shot runner (the `run_phase2_confirmation.sh`
      precedent: double env-var guard, refusal pinned by a test)
