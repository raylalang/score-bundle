# Tonal-metric confirmation design — REGISTRATION (2026-10-27)

> **Status: REGISTERED as of the commit that carries this banner (tag
> `phase2-tonal-registration-2026-10-27`).** This committed revision
> freezes the claims, the decision rule, the systems, the inputs and the
> exact commands BEFORE the run. The run is a separate, deliberate act,
> performed ONCE, with every number reported whatever the outcome.
> Pool: the 13-piece URMP confirmation pool, spent 2026-08-27 for claims
> C1–C4 and RE-USED HERE WITH FULL DISCLOSURE (section "The pool").
> (Document history: drafted 2026-08-27 as a proposal with the power
> check; pool decided 2026-10-23 by Ray, the professor hands-off; frozen
> 2026-10-27. After the run this file is never edited except by a dated
> erratum block.)

## What is claimed

The development result (`results/phase2_tonal_dev.md`, exploratory,
hypothesis pre-committed 2026-08-17 in the constants block of
`scripts/eval_phase2_real.py`, result seen 2026-08-19): replacing the
pitch metric of the score graph with circle-of-fifths (tonal) distance
helps the intonation channel and re-imposes the known replacement
penalty on timing. Claims are tonal minus plain graph, as-given variant,
paired per (track, seed), against estimator targets:

- **T2 (intonation calibration, PRIMARY).** Paired ΔNLL on channel `c`
  negative with the 95% bootstrap interval excluding zero. Development
  basis: −0.050 [−0.072, −0.029]\*.
- **T1 (intonation recovery, SECONDARY, non-gating).** Paired ΔRMSE on
  `c` negative with the interval excluding zero. Development basis:
  −0.213 [−0.339, −0.085]\*. Underpowered on this pool (0.52–0.63, table
  below) and stated so at registration.
- **T3 (honest boundary, report-only).** A timing penalty is predicted
  (development τ ΔNLL +0.027 [+0.003, +0.068]\*); reported, gates nothing.

T1 and T2 were ordered by the power check (below); T2 leads because it
is powered and because calibration-first is the thesis's posture.

## Decision rule (verbatim, frozen)

T2 PASSES iff the paired ΔNLL on channel `c` (gp_tonal_asgiven minus
gp_asgiven, per (track, seed), against estimator targets, as
`_render` computes it with `eval_graphgp.bootstrap_ci`, B = 2000,
percentile, `numpy.random.default_rng(31)` consumed in the contrast
order of the tagged code) has a negative mean AND its 95% interval
excludes zero. T2 pass ⇒ the thesis states: "the circle-of-fifths metric
improves intonation calibration, confirmed once on the re-used pool
(disclosed)". T2 fail ⇒ reported verbatim; the metric stays exploratory
and the appendix keeps its status. T1 and T3: reported with their
intervals, non-gating. Adopting the tonal metric as the intonation
channel's default is a separate, later decision (it would re-run
published tables) and is not implied by a pass. No post-hoc filters, no
added seeds, no reruns.

## Systems and the identity check

All five arms of the development configuration run, because
`_fit_systems` builds them together: `gp`, `gp_asgiven`, `nograph`,
`gp_tonal`, `gp_tonal_asgiven`. Only the as-given tonal-versus-plain
contrast is the claim; the learned-scale contrasts, the tonal-versus-
no-graph contrasts, the per-family table and the reference-target
scoring are report-only. **Identity check, pre-stated:** the three plain
arms run on the same inputs, masks and system order as on 2026-08-27 at
the same OMP_NUM_THREADS=4, so their rows must reproduce
`results/phase2_confirmation_results.md` digit for digit; a deviation
is reported as a deviation, not scored.

## The pool: disclosed re-use of the spent 13-piece URMP pool

- Composition, inputs, masks: as in `docs/corrnoise_prereg_design.md`
  (same archive, md5 `6a44d902589a1a0369f551cea1b991a9`, 40 unique
  tracks, 80 cells, masks reproduced exactly).
- **Disclosure.** Replication-grade, not pristine: the pool was spent on
  2026-08-27 for claims C1–C4, and its targets, plain-graph fits and
  per-family contrasts are in the thesis and were known when this claim
  was selected. The tonal-versus-plain contrast has never been computed
  on these tracks: `stage_run` has refused the tonal variant in
  confirmation mode since 2026-08-19 (commit e59c7a1), and the
  confirmation caches were absent until the runner copies the archive.
  The hypothesis was pre-committed on 2026-08-17, its development result
  seen on 2026-08-19 before claim selection: the ordinary
  development-to-confirmation path. The real Bach10 corpus was not
  pursued this cycle (the archive on disk is the Sibelius-resynthesised
  version, synthetic audio). The thesis labels the result "confirmation
  on the re-used pool (disclosed)".
- **Where this pool differs from development, pre-stated.** It is
  brass-heavy (17 of 40 tracks; development was strings-heavy, 70 of 153
  pairs), and the development tonal effect was carried by strings and
  woodwind, so the effect may be smaller here. T1 is underpowered.
- **Multiplicity.** The correlated-timing-noise claim is registered on
  the same pool in the same week, on a different channel by a different
  mechanism. No cross-registration correction; both reported.

**Power check (RUN 2026-08-27, piece-level subsamples of the 31
development pieces / 153 pairs, 95% interval star criterion):**

| pieces | P(T1 ΔRMSE stars) | P(T2 ΔNLL stars) |
|---|---|---|
| 6 | 0.47 | 0.61 |
| 10 | 0.52 | 0.72 |
| **13 (this pool, ≈3 tracks/piece: between the 10- and 15-piece rows)** | **0.52–0.63** | **0.72–0.90** |
| 15 | 0.63 | 0.90 |
| 20 | 0.80 | 0.99 |

## Protocol constants (frozen)

Identical to the 2026-08-17 registration except for the arms: 30% of
notes hidden, SEEDS (0, 1), MIN_NOTES 30, C_MAX 150, as-given noise;
tonal graph = `build_adjacency_tonal` with the constants of the
development study (unchanged since 2026-08-17); bootstrap B = 2000, rng
31, 95% percentile intervals; OMP_NUM_THREADS=4, 6 shards (shard
equivalence pinned by `tests/test_phase2_eval_shard.py`). Code state:
the commit carrying this banner (tag `phase2-tonal-registration-2026-10-27`);
the runner refuses if any frozen file differs from the tag.

## Exact commands

```
export PHASE2_SPLIT=confirmation PHASE2_CONFIRMATION_I_AM_SURE=yes PHASE2_REUSE_I_AM_SURE=yes
bash scripts/run_phase2_reuse_confirmation.sh tonal 6
# = md5 check of the inputs; 6 x `python scripts/eval_phase2_real.py run-tonal K/6`
#   at OMP_NUM_THREADS=4; `python scripts/eval_phase2_real.py report-tonal`
#   -> results/phase2_tonal_confirmation_results.md
```

What is reported, pass or fail: the full rendered tables (every arm,
every channel, both target sets, the family table), the verdict block
appended to `results/phase2_tonal_confirmation_results.md` in the
2026-08-27 format including the identity-check outcome, and the
evidence (run log, shard logs, cell fragments) archived under
`evidence/phase2_reuse_confirmation/tonal/` in the same commit.

## Pre-registration checklist

- [x] Pool decided: disclosed re-use of the spent URMP pool (Ray,
      2026-10-23)
- [x] Bach10 items: N/A for re-use (same corpus, same calibrated
      estimator chain, same archived targets)
- [x] Power check: RUN (table above); T2 primary, T1 secondary and
      stated underpowered
- [x] Guarded one-shot runner (`scripts/run_phase2_reuse_confirmation.sh`)
      with the refusal test `tests/test_phase2_reuse_guard.py`; the
      tonal variant stays refused in confirmation mode without the
      re-use consent
- [x] Claims T1–T3, the decision rule and this protocol frozen in the
      registration commit; tag `phase2-tonal-registration-2026-10-27`
