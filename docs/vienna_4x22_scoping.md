# Vienna 4x22 — performer identification (scoping + wiring)

The one downstream task ASAP/MAESTRO cannot support — **performer identification** — needs a
corpus where the *same score* is played by *identified* performers. The Vienna 4x22 Piano
Corpus is exactly that, and this note records why it fits, how it is wired in, the eval
design, and how to obtain/run it.

## What it is

- **22 skilled pianists**, each performing the **same four excerpts** on one Bösendorfer SE
  computer-monitored grand (Goebl 1999):
  `Chopin_op10_no3`, `Chopin_op38`, `Mozart_K331_1st-mov`, `Schubert_D783_no15`.
- **88 note-level score↔performance alignments** in the `.match` format
  (`{piece}_p{NN}.match`, `p01`–`p22`, consistent performer ids across pieces).
- Repo: <https://github.com/CPJKU/vienna4x22> (v1.0.0 match files; the corrected
  `OFAI/vienna4x22_rematched` is a drop-in alternative). Tiny (MIDI 855 KB).

## Why it is the right corpus (and why the ASAP style probe was a negative)

Task 5 (composer-era classification, `downstream_tasks_results.md`) was an honest negative,
and it *had* to be confounded: on ASAP, expression is entangled with the score itself, so
"style from expression" cannot be cleanly separated from "style from the notes." Vienna 4x22
removes that confound by construction:

- **Real labels** — the performer id is known and consistent across the four pieces.
- **No score confound** — every performer plays the identical notes, so the *only* signal is
  how they play: the expression `y = [tau, log r, v]`. This is the cleanest possible test of
  whether our inferred expressive variables (and the graph prior applied as a feature
  cleaner) carry performer-discriminative information.

## What was wired in (this phase)

- `src/score_bundle/vienna.py` — loader:
  - `load_vienna_meta(root)` parses `match/{piece}_p{NN}.match` → `ViennaRecord`s.
  - `load_vienna_performance(match_path)` uses **partitura** (`load_match`, import-guarded)
    to read the matched score/performance note arrays → `(Score, y[N,3])`.
  - `performance_variables(...)` — pure-numpy `y = [tau, log r, v]` from matched arrays, with
    a **local linear tempo** estimate (the `.match` files carry no separate beat grid, unlike
    ASAP, so tempo is fit from the notes; a global ritardando is absorbed as tempo while a
    note early/late relative to its local pulse shows a residual). Tested directly on
    synthetic arrays.
- `src/score_bundle/downstream.py::grouped_nearest_centroid` — leave-one-**piece**-out
  nearest-centroid with unsupervised per-piece centering (removes piece-level offsets using
  group membership only, never performer labels).
- `scripts/eval_vienna_performer.py` — the eval (below). Guards on missing corpus/partitura
  (prints instructions, exits 0, so CI stays green).
- `tests/test_vienna.py` — meta parsing, variable maths (tempo recovery, rubato/articulation
  detection, chord-safety), and the grouped classifier on synthetic fixtures. No corpus or
  partitura needed for the test suite.

## Eval design (honest about the small N)

- **Task:** 1-of-22 performer classification, chance = 0.045.
- **Split:** leave-one-**piece**-out (train on 3 pieces, test on the 4th), so piece identity
  cannot be exploited — the classifier must generalize a performer's *style* across
  repertoire. Per-piece centering removes the piece-level expression offset.
- **Feature sources compared:** `raw` (aggregates of observed expression) vs `graph`
  (aggregates of graph-*denoised* expression) — the on-thesis question — plus a `per-segment`
  higher-N variant (each performance split into contiguous segments).
- **Classifier:** dependency-light nearest-centroid on 8 style descriptors (rubato
  magnitude/smoothness, articulation level/spread, dynamic range/jaggedness). Deliberately
  simple; the comparison of interest is raw-vs-graph *features*, not classifier tuning.
- **Honest caveats:** 22 classes × 4 examples is small; nearest-centroid is weak. Absolute
  accuracy is not the headline — the raw-vs-graph delta and "beats chance" are. A stronger
  classifier (e.g. per-segment features + a linear model) and the `rematched` files are the
  obvious next steps if the signal warrants it.

## Obtain & run

```bash
git clone https://github.com/CPJKU/vienna4x22 ../data/vienna4x22
pip install "partitura>=1.2.0"
python scripts/eval_vienna_performer.py --root ../data/vienna4x22
```

## Result

`python scripts/eval_vienna_performer.py --root ../data/vienna4x22 --max-notes 250`
(log `logs/vienna_performer.log`; notes capped at 250/performance so the per-piece graph
EB fit stays tractable — the prefix is a fixed score excerpt shared across performers, so
the comparison is fair):

```
Performer ID (leave-one-piece-out, chance = 1/22 = 0.045)
features           accuracy    n
raw (per-perf)       0.136     88
graph (per-perf)     0.080     88
raw (per-seg)        0.179    352
```

**Two findings, both honest:**

1. **Performer identity is recoverable from inferred expression alone** — raw expression
   features reach 3–4× chance (0.136 per performance, 0.179 per segment) under
   leave-one-piece-out, where piece identity cannot be exploited. So the Phase-1 variables
   `[tau, log r, v]` we extract *do* carry performer-discriminative style, on a corpus with
   real labels and no score confound. (Absolute accuracy is modest — 22 classes, 4 examples
   each, a deliberately simple nearest-centroid; a stronger classifier would do better. The
   point is the signal is well above chance and the extraction is validated.)
2. **Graph-denoising does not help — it hurts** (0.136 → 0.080). Shrinking the expression
   toward a smooth graph field removes exactly the fine per-note detail that distinguishes
   performers. This is the *same direction* as the ASAP composer-era negative
   (`downstream_tasks_results.md`, Task 5): the graph prior earns its keep on **per-note
   recovery and calibration**, not as a **feature cleaner for piece-level classification** —
   a consistent, useful boundary on where structure helps.

## Definition of done

- [x] Loader parses real `.match` files into `(Score, y)` (verified: 451 notes on
      `Chopin_op10_no3_p01`, finite `y`, sensible scales).
- [x] Numpy variable maths + meta parsing + grouped classifier tested on synthetic fixtures.
- [x] Eval runs end-to-end on the real corpus, leave-one-piece-out, raw vs graph features,
      against the chance baseline (raw 0.14–0.18 vs chance 0.045; graph does not help).
- [ ] (Optional next) stronger classifier (linear model on per-segment features), the
      `rematched` files, and per-piece velocity normalization if a deeper performer-ID
      result is wanted — not needed for the boundary finding above.

## Note on `../data/` and contamination

Vienna 4x22 is symbolic-only Chopin/Mozart/Schubert excerpts; it does **not** overlap the
MAESTRO Phase-0 LM pretraining set at the performance level (different recordings), and the
performer-ID eval uses no LM features, so there is no train/eval contamination to guard here.
