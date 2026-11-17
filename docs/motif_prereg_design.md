# Motif-study one-shot on held-out performances — REGISTRATION DRAFT (2026-11-17)

> **Status: DRAFT, written 2026-11-17 after Ray's decision (AskUserQuestion,
> 2026-11-17: the motif study becomes a thesis chapter with one
> pre-specified test on untouched performances). The numbers marked
> PROVISIONAL come from pilot one and are replaced by pilot two's before
> the freeze. It becomes REGISTERED at the commit that carries the tag
> `motif-registration-2026-11-20`, which freezes the claims, the decision
> rule, the pool, the arms and the exact commands BEFORE the run. The run
> is one deliberate act, performed ONCE by `scripts/run_motif_heldout.sh`,
> every number reported whatever the outcome. After the run this file is
> never edited except by a dated erratum block.**

## What is claimed

The development results (`results/motif_pilot_dev.md`, 2026-11-04;
`results/motif_pilot2_dev.md`, 2026-11-17; hypotheses committed before
each run in the docstring of `scripts/eval_motif_pilot_dev.py`): when
whole occurrences of an annotated motif are hidden, joining their notes
to the observed occurrences of the same motif (one more edge family in
the score graph, Gaussian in relative position inside the occurrence,
ℓ_u = 0.1, weight learned by the per-piece evidence) lowers the timing
log score and the articulation error on the hidden notes against the
time-only graph. Pilot one, 21 pieces, two seeds, one shared weight:
timing ΔNLL −0.117 [−0.245, −0.006]; articulation ΔRMSE −0.033 [−0.066,
−0.007]; velocity calibration worse (+0.069, one shared weight). Pilot
two (per-channel weight, four seeds): PROVISIONAL, filled in at the
freeze.

Claims, hidden motif notes under the occurrence-hidden scheme, paired per
piece (seeds and performances averaged within the piece), the chosen
motif arm minus the time-only arm:

- **M1 (PRIMARY).** The better-powered of the two pre-named contrasts by
  pilot two's power table (section "Power check"): either the timing
  log score (ΔNLL < 0) or the articulation RMSE (ΔRMSE < 0), with the
  95 percent bootstrap interval excluding zero. PROVISIONAL from pilot
  one: articulation RMSE (0.86 at 18 pieces) over the timing log score
  (0.60).
- **M2 (SECONDARY, non-gating).** The other of the two, same rule.
- **M3 (report-only).** Velocity log score and coverage on the same
  notes (the pilot-one penalty is expected to be gone with the
  per-channel weight; a remaining penalty is reported); the timing RMSE;
  the pitch term is not run (the arms are time-only against time plus
  motif).

The motif arm is ONE of `time_motif` (shared weight inside the
adjacency) and `time_motif_pc` (the family as its own per-channel block),
chosen at the freeze from pilot two by the same power table and named
here; the other runs as report-only.

## Decision rule (verbatim, to be frozen)

M1 PASSES iff the paired per-piece difference (the chosen motif arm
minus `time`, hidden motif notes, occurrence-hidden scheme, four seeds
averaged, then the piece's held-out performances averaged) of the chosen
metric, as `stage_report` computes it (B = 2000 percentile bootstrap,
`numpy.random.default_rng(0)` over the pieces present in both arms),
has a negative mean AND its 95 percent interval excludes zero. M1 pass ⇒
the thesis states: "annotated motif structure carries expressive
information beyond temporal smoothing, confirmed once on held-out
performances of the development pieces (disclosed)". M1 fail ⇒ reported
verbatim; the motif chapter keeps its development results as such and
states the held-out miss. M2 and M3: reported with their intervals,
non-gating. No post-hoc filters, no added seeds or performances, no
reruns, no change of ℓ_u, the masking rule or the scoring set.

## Systems

Exactly the development machinery of pilot two: score features with the
per-piece bias column, no music-model embeddings (so no contamination
question and no GPU), ICM coupling, floored noise, exact per-piece
evidence, the published predictive-variance floor; the time-only
Gaussian graph as the base; the motif family from the BPS-motif
annotations mapped by (crotchet onset, pitch) after the piece's offset
search (the 0.85 match rule applies per performance); 400-note excerpts
(the thesis's convention); the occurrence-hidden masking rule with seeds
0 to 3 (`default_rng(1000·seed + 17)`, masks drawn per performance since
the excerpt's matched notes differ by performance); the scoring set =
hidden motif notes whose type has another occurrence with an observed
note.

## The pool: held-out performances of the development pieces (disclosed)

- For each of the 21 development pieces, the aligned ASAP performances
  other than the first in sorted order (the development performance),
  capped at four in sorted order (`--perf-source heldout --max-perf 4`).
  Counted 2026-11-17 from ASAP's metadata: 18 pieces have at least one
  such performance (1-1, 9-1 and 10-1 have none), 58 performances in all
  (2-1: 2, 4-1: 4, 5-1: 2, 7-1: 4, 8-1: 1, 11-1: 4, 12-1: 2, 15-1: 2,
  16-1: 4, 17-1: 4, 18-1: 4, 21-1: 4, 22-1: 2, 26-1: 3, 27-1: 4, 28-1: 4,
  30-1: 4, 31-1: 4). The piano confirmation pieces 3-1 and 24-1 stay
  excluded. The exact list is written by the runner into the run log
  before any fit.
- **Disclosure.** The pieces, their scores, the annotations, the
  occurrence structure and hence the edges and the masking rule are
  those of the development study, where the claim was selected. What is
  new is the performances: the targets (timing, articulation, velocity
  of a different pianist) have never been read by any run of this study.
  The test therefore asks whether the development finding generalizes
  across performers of the same pieces, not across pieces. The thesis
  labels the result "confirmation on held-out performances of the
  development pieces (disclosed)". A test across pieces would need
  annotated movements outside ASAP (seven exist, without performances)
  and is stated as future work.
- **Unit of analysis.** The piece: performances are averaged within a
  piece before pairing, so n = 18 and performance-level noise is reduced
  without inflating the count.
- **Multiplicity.** One claim; the phrase-scale waveform one-shot of the
  same week is on a different corpus and setting.

## Power check

PROVISIONAL (pilot one, 21 pieces, two seeds, shared weight;
`scripts/power_motif.py`, pieces resampled, the report's bootstrap rule):

| n pieces | timing ΔNLL at the measured effect | articulation ΔRMSE at the measured effect |
|---|---|---|
| 10 | 0.52 | 0.62 |
| 14 | 0.55 | 0.73 |
| **18 (this pool)** | **0.60** | **0.86** |
| 21 | 0.69 | 0.91 |

At two thirds of the measured effect both fall below 0.35. The table is
recomputed from pilot two (per-channel arm, four seeds) at the freeze
and decides M1 and the motif arm.

## Protocol constants (frozen)

ℓ_u = 0.1; s_m = 1 (the per-channel block's shape); MAX_NOTES = 400;
match threshold 0.85; occurrence-hidden probability 0.4 with at least
one occurrence of each type observed and one hidden; seeds 0 to 3;
`maxiter` 200; bootstrap B = 2000, rng 0, 95 percent percentile;
`OMP_NUM_THREADS=2`, 14 shards (jobs to shards by index modulo 14).
Code state: the commit carrying the tag; the runner refuses if any
frozen file differs from the tag.

## Exact commands

```
export MOTIF_HELDOUT_I_AM_SURE=yes
bash scripts/run_motif_heldout.sh 14
# = align under the guard (the held-out list into the log); 14 x
#   `python scripts/eval_motif_pilot_dev.py --stage run --out results/motif_heldout
#    --perf-source heldout --max-perf 4 --arms time,time_motif,time_motif_pc
#    --schemes occ --seeds 4 --shard K/14`; then `--stage report`
#   -> results/motif_heldout/report.md, effects.json; verdict -> results/motif_heldout_results.md
```

Projected: 58 performances × 4 seeds × 3 arms = 696 fits, about 75 s
each, roughly 1 h on 14 shards.

What is reported, pass or fail: the full report (every arm, channel and
subset), the learned weights and per-channel scales per performance, the
verdict block in the 2026-10-27 format, and the evidence (run log with
the performance list, shard logs, cells, report, effects) archived under
`evidence/motif_heldout/` in the same commit.

## Pre-registration checklist

- [x] Pool decided: held-out performances of the development pieces,
      disclosed (Ray, 2026-11-17)
- [x] Guarded one-shot runner (`scripts/run_motif_heldout.sh`) with the
      refusal tests in `tests/test_motif_pilot.py`; the held-out source
      refuses without consent and never returns the first performance
- [ ] Pilot two read; M1, the motif arm and the power table fixed
- [ ] Claims M1–M3, the decision rule and this protocol frozen in the
      registration commit; tag `motif-registration-2026-11-20`
- [ ] Run once; evidence archived; verdict scored by the frozen rule
