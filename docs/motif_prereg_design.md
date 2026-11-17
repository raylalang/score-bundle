# Motif-study one-shot on held-out performances — REGISTRATION DRAFT (2026-11-17)

> **Status: DRAFT, written 2026-11-17 after Ray's decision (AskUserQuestion,
> 2026-11-17: the motif study becomes a thesis chapter with one
> pre-specified test on untouched performances). The numbers marked
> PROVISIONAL were replaced by pilot two's on 2026-11-17 (the draft is
> complete; the freeze is the tag). It becomes REGISTERED at the commit that carries the tag
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
two (four seeds; the per-channel block added): the shared weight's timing
ΔNLL is −0.071 [−0.230, +0.097] (the pilot-one star did not survive two
more seeds), its articulation ΔRMSE −0.0345 [−0.0600, −0.0125] on every
seed pair; the per-channel block gives articulation ΔRMSE −0.0189
[−0.0381, −0.0053], velocity ΔNLL −0.086 [−0.200, −0.018] with coverage
0.944, and nothing on timing (−0.005). The robust development finding is
articulation accuracy.

Claims, hidden motif notes under the occurrence-hidden scheme, paired per
piece (seeds and performances averaged within the piece), the chosen
motif arm minus the time-only arm:

- **M1 (PRIMARY).** The articulation RMSE on the hidden motif notes,
  ΔRMSE < 0 with the 95 percent bootstrap interval excluding zero. Fixed
  2026-11-17 from pilot two's power table (section "Power check"):
  articulation RMSE 0.95 at 18 pieces against 0.33 for the timing log
  score; the two contrasts were pre-named in the draft of the same day
  and the better-powered one is the primary, as that draft said.
- **M2 (SECONDARY, non-gating).** The timing log score on the same
  notes, ΔNLL < 0 with the interval excluding zero (power 0.33 at 18
  pieces at the four-seed effect; a miss is expected and reported).
- **M3 (report-only).** Velocity log score and coverage on the same
  notes (the pilot-one penalty is expected to be gone with the
  per-channel weight; a remaining penalty is reported); the timing RMSE;
  the pitch term is not run (the arms are time-only against time plus
  motif).

The motif arm for M1 and M2 is `time_motif` (the shared weight inside
the adjacency), chosen 2026-11-17 by the power table (0.95 against 0.93
for the per-channel block at 18 pieces; the larger and more even
articulation effect, 15 of 21 pieces below zero against 13).
`time_motif_pc` (the family as its own per-channel block) runs beside it
and is reported in full (its velocity gain and coverage, its articulation
RMSE); nothing about it gates.

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

Pilot two (21 pieces, four seeds; `scripts/power_motif.py
results/motif_pilot2 --arm ARM --scheme occ --channel CH --metric M
--reps 1000`, pieces resampled, the report's bootstrap rule), P(star):

| n pieces | articulation ΔRMSE, `time_motif` | `time_motif_pc` | timing ΔNLL, `time_motif` | `time_motif_pc` |
|---|---|---|---|---|
| 10 | 0.74 | 0.66 | 0.31 | 0.03 |
| 14 | 0.86 | 0.84 | 0.31 | 0.02 |
| **18 (this pool)** | **0.95** | **0.93** | **0.33** | **0.02** |
| 21 | 0.97 | 0.96 | 0.30 | 0.03 |

At two thirds of the measured effect, 18 pieces: articulation 0.51
(`time_motif`) and 0.39 (`time_motif_pc`); timing 0.14; at half, 0.26 and
0.17. The performances of the pool are different from the development
performance, so the effect on them is unknown; a held-out effect of two
thirds of the development one gives about one chance in two of a star.
The pilot-one table (timing 0.60, articulation 0.86 at 18 pieces, two
seeds) is superseded.

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
- [x] Pilot two read; M1, the motif arm and the power table fixed (2026-11-17)
- [ ] Claims M1–M3, the decision rule and this protocol frozen in the
      registration commit; tag `motif-registration-2026-11-20`
- [ ] Run once; evidence archived; verdict scored by the frozen rule
