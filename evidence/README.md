# evidence/ — primary artifacts behind the thesis claims

Committed deliberately, against the repo's general "run artifacts are regenerable —
do not commit" rule, because these are the artifacts that rule is **wrong** for.

## Why this directory exists

The headline claims rest on a **preregistered, one-shot confirmation run** (20 ASAP
pieces never used for any modelling decision; protocol frozen in
`docs/graphgp_first_design.md` before the data existed). Two senses of
"reproducible" must not be conflated:

- **Mechanically:** yes. Fits are deterministic given the inputs here, the code, the
  public datasets, and the Phase-0 checkpoint (at a fixed BLAS thread count). A
  third party can re-execute and re-derive every number.
- **Epistemically:** no. The confirmation set is spent — the run's one-shot status
  is a fact about history, not about code. If these artifacts were lost, the numbers
  could be recomputed but the *record of the single evaluation* could not.

Hence: the primary artifacts live in version control.

## Contents

- `logs/` — the report logs cited throughout `docs/draft.tex` and
  `docs/graphgp_first_design.md` (cited there under their original `logs/...`
  paths; these are byte-identical archival copies). Key files:
  `confirmation_verdict.log` (THE headline evidence), `graphgp_v2_report*.log`
  (dev ladder, one code state), `dev12_report.log` (12-seed robustness),
  `downstream_gpfirst_report.log` + `overnight_performer.log` (all six tasks),
  `guarded_ab_verdict.log` (guard A/B + τ-tail), the two-stage record logs.
- `results/graphgp_conf/`, `results/kernels_conf_*/` — the raw per-cell
  confirmation outputs (held-out targets, predictions, predictive stds per
  (piece, seed)); every confirmation number is recomputable from these alone.
- `inputs/conf_inputs_{lm,featlm}.pkl` — the confirmation masks and strict
  mask-aware means: with the ASAP data, the extraction script, and the Phase-0
  checkpoint, these make the confirmation run re-executable end to end.

## Not included (large; mechanically regenerable, with recipes)

- `.cache/asap_arrays_named50.pkl` — `scripts/extract_asap_arrays.py --n-eval-pieces 50`
  (seed 0; first 30 eval pieces verified byte-identical to the development cache).
- `.cache/conf_emb_ma.pkl` (63 MB) — `scripts/eval_kernels.py --stage precompute
  --arrays-cache .cache/asap_arrays_named50.pkl --eval-start 30 --n-eval-pieces 20
  --mask-seed-base 2000 --dump-embeddings ...`.
- `checkpoints/maestro_scaled/best.pt` — the Phase-0 LM checkpoint (archived
  locally; consider a release artifact).

## Integrity

These files must never be edited. Any future evaluation happens on a NEW
confirmation set and lands as new files; see the preregistration and one-shot
rules in `docs/graphgp_first_design.md`.
