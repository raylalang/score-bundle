# Phrase-scale prior-value one-shot on the URMP confirmation tracks — REGISTRATION DRAFT (2026-11-17)

> **Status: DRAFT, written 2026-11-17 after Ray's decision to register and
> run (AskUserQuestion, 2026-11-17: "Register and run anyway", with the
> power check below in view). It becomes REGISTERED at the commit that
> carries the tag `phrase-prior-registration-2026-11-18`, which freezes
> the claim, the decision rule, the cells, the inputs and the exact
> commands BEFORE the run. The run is one deliberate act, performed ONCE
> by `scripts/run_phrase_prior_confirmation.sh`, every number reported
> whatever the outcome. After the run this file is never edited except by
> a dated erratum block.**
> Pool: the 13-piece URMP confirmation pool, spent 2026-08-27 (C1–C4),
> re-used with disclosure 2026-10-27 (correlated timing noise; tonal
> metric), and used here a THIRD time with disclosure, the FIRST time at
> the waveform level (section "The pool").

## What is claimed

The development result (`results/prior_value_phrase_dev.md`, 2026-10-20,
hypotheses committed before the run in the docstring of
`scripts/eval_prior_value_phrase_dev.py`): on 6-to-10-note legato phrases
whose hidden middle note's pitch recurs at least two notes away, with the
hidden note acoustically absent from the design and a free common-mode
tuning scalar in every arm, the score-graph prior over the per-note
intonation constants gives a smaller absolute error on the hidden note
than a purely temporal (path-graph) prior. Development basis on 48 cells
from 78 tracks: graph minus temporal −1.63 cents [−2.93, −0.46], better
on 65 percent of cells; graph minus independent −0.07 [−0.42, +0.27];
temporal minus independent +1.56 [+0.26, +2.92]. The reading of the
three spans (thesis chapter 8): the weak form survives (score structure
at least as good as a naive temporal prior), the strong form does not
(nothing measurable beyond the common offset).

Claims, paired per cell on the masked condition, scored against the
reference centre:

- **W1 (PRIMARY, the weak form).** Mean paired difference in absolute
  error on the hidden note, graph minus temporal, negative with the
  95 percent bootstrap interval excluding zero.
- **W2 (report-only, the strong form).** Graph minus independent, with
  its interval. Development basis: a tie. A star here would be new
  evidence for the strong form and is reported as a development-level
  surprise, not as a confirmed claim (not pre-stated as expected).
- **W3 (control).** Clean-condition contrasts tie (both intervals cover
  zero, |mean| below 0.1 cents).

## Decision rule (verbatim, to be frozen)

W1 PASSES iff the mean over the drawn cells of |c_graph − c_ref| −
|c_temporal − c_ref| on the masked condition, as `stage_report` computes
it (B = 2000 percentile bootstrap, `numpy.random.default_rng(31)`
consumed in the contrast order of the tagged code: graph vs temporal
first), is negative AND its 95 percent interval excludes zero. W1 pass ⇒
the thesis states: "at phrase scale the score graph beats a purely
temporal prior on held-out tracks, confirmed once on the re-used pool
(disclosed)". W1 fail ⇒ reported verbatim; chapter 8 keeps the three-span
development verdict as its answer and states the held-out miss. W2 and
W3: reported with their intervals, non-gating. No post-hoc filters, no
added cells, no reruns, no change of the common-mode scalar, the curve
prior or the cell rule.

## Systems

Exactly the development machinery: `scripts/eval_prior_value_phrase_dev.py`
in confirmation mode (the only code change is the switch of the track
side and of the artifact paths; the development cell list is reproduced
bit-exactly by the same selection code, checked 2026-11-17 with the
`count` verb against the 48 development cells). Arms independent /
temporal / graph at matched marginal variance, each with the tuning
scalar (sd 50 cents); hidden note dropped from the design
(`PRIORVAL_DROP_HIDDEN=1`); the evidence-fitted curve prior of the
development side (`.cache/phase3_prior_fit.json`, frozen); masked and
clean conditions, six fits per cell.

## The cells (fixed before any fit)

The development rule unchanged: legato runs, phrases of 6 to 10 notes,
hidden middle note with a finite reference centre within ±150 cents
whose MIDI pitch recurs in the phrase at index distance ≥ 2 and never at
distance 1, at most 4 phrases per track, then `default_rng(0)` draws up
to 48. Counted on 2026-11-17 with `count` (no fit, no audio read): the 40
unique confirmation tracks hold 20 tracks with admissible phrases and 56
candidates after the cap; **48 cells are drawn**, strings 14 (vn 9,
va 4, vc 1), winds 25 (fl 7, ob 6, cl 5, bn 4, sax 3), brass 9 (tpt 6,
hn 2, tbn 1). The drawn list is printed by the runner into the run log.

## The pool: third disclosed re-use

- The 13 pieces, 40 unique tracks (md5-deduplicated across arrangements),
  the archived 2026-08-27 inputs (`evidence/phase2_confirmation/
  urmp_targets_conf_inputs.pkl`, md5 `6a44d902589a1a0369f551cea1b991a9`)
  for the reference centre and MIDI pitches; audio from the URMP files.
- **Disclosure.** The pool was spent on 2026-08-27 for claims C1–C4 and
  re-used on 2026-10-27 for two claims at the tracked-target level. Its
  reference targets and the strings-and-winds fits are in the thesis.
  **No waveform script has ever read a confirmation track** (every
  phase-3, prior-value and boundary script loads the development cache
  and `dev_unique_tracks()` only; no log or results header mentions the
  pool), so the phrase-scale contrast has never been computed on these
  tracks. The thesis labels the result "confirmation on the re-used pool
  (disclosed, third use)".
- **Where this pool differs from development, pre-stated.** The
  development effect is carried by strings: by family, graph minus
  temporal is −2.16 [−3.83, −0.59] on 33 string cells, −0.29 [−0.89,
  +0.45] on 7 wind cells, −0.59 [−2.21, +1.09] on 8 brass cells. The
  drawn confirmation cells are winds-heavy (25 of 48; strings 14). At
  the development family rates the expected mean here is about −0.9
  cents, not −1.6.
- **Multiplicity.** One claim, one pool use; the motif-study
  registration of the same week is on a different corpus.

## Power check (RUN 2026-11-17, before registration)

Resampling the 48 development paired differences (mean −1.63, SD 4.28)
to n cells, with the effect scaled, and counting how often the 95
percent percentile interval excludes zero (1500 draws):

| n cells | at the development effect (−1.63) | at two thirds (−1.09) | at half (−0.81) |
|---|---|---|---|
| 20 | 0.46 | 0.23 | 0.14 |
| 30 | 0.60 | 0.32 | 0.20 |
| 40 | 0.73 | 0.38 | 0.24 |
| **48 (this pool)** | **0.81** | **0.44** | **0.28** |

With the family mixture above the expected effect is near the "half"
column: the chance of a star is roughly 0.3 to 0.5. Ray chose to run
with this in view; a miss is a reported miss, not evidence against the
weak form at development level.

## Protocol constants (frozen)

`PHRASE_MIN, PHRASE_MAX = 6, 10`; `CAP_PER_TRACK = 4`; `N_CELLS = 48`;
`MASK_GUARD_S = 0.05`; tuning sd 50; `S_SCALE = 1.0`; the hypers of
`.cache/phase3_prior_fit.json`; bootstrap B = 2000, rng 31, 95 percent
percentile; `OMP_NUM_THREADS=2`, 12 shards (cells to shards by index
modulo 12, as on the development side). Code state: the commit carrying
the tag; the runner refuses if any frozen file differs from the tag.

## Exact commands

```
export PHASE2_SPLIT=confirmation PHASE2_CONFIRMATION_I_AM_SURE=yes PHASE2_REUSE_I_AM_SURE=yes
bash scripts/run_phrase_prior_confirmation.sh 12
# = md5 check of the inputs; `count` into the log; 12 x
#   `python scripts/eval_prior_value_phrase_dev.py run K/12` at OMP_NUM_THREADS=2;
#   `python scripts/eval_prior_value_phrase_dev.py report`
#   -> results/phrase_prior_confirmation_results.md
```

Projected wall time from the development run: median 4345 s per cell
(six fits), about 8 h on 12 shards, bounded by the slowest shard.

What is reported, pass or fail: the full rendered table (both
conditions, three arms, coverage, median sd), the three paired contrasts,
the clean contrasts, the dose-response correlation, the verdict block
appended to the results file in the 2026-10-27 format, and the evidence
(run log with the drawn cell list, shard logs, cell pickles) archived
under `evidence/phrase_prior_confirmation/` in the same commit.

## Pre-registration checklist

- [x] Pool decided: third disclosed re-use of the URMP confirmation pool
      (Ray, 2026-11-17)
- [x] Power check: RUN (table above); W1 primary, chance of a star
      stated as roughly 0.3 to 0.5
- [x] Cells counted without fitting: 48 drawn from 56 candidates
- [x] Guarded one-shot runner (`scripts/run_phrase_prior_confirmation.sh`)
      with the refusal test `tests/test_phrase_prior_guard.py`; the
      development path reproduces its 48 cells bit-exactly
- [ ] Claims W1–W3, the decision rule and this protocol frozen in the
      registration commit; tag `phrase-prior-registration-2026-11-18`
- [ ] Run once; evidence archived; verdict scored by the frozen rule
