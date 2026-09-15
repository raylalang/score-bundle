# Roadmap

Updated 2026-09-16. One ongoing task at a time. Order stays deliberate:
understanding, then legibility, then modelling — with the 09-10/11
exploration week run on Ray's explicit mandate and now closed
(`results/exploration_week_2026-09.md` is the one-page ledger).

## NOW — Package, adopt, prep (3 days to the meeting; decided 2026-09-16)

Direction decided by Ray: **adoption path**. The inference-level winners
of the exploration week become opt-in capabilities of the pipeline (the
`fit_guarded` precedent: default off, recommended for new runs, published
paths bit-untouched; any change to a *reported* protocol needs its own
registration).

- Day 1: package the week (ledger, this roadmap, verdict banners on the
  stale docs: `sm_estimator_note.tex`, the two kernel memos).
- Day 2: adoption engineering, opt-in and unit-pinned — correlated
  τ-noise block in `gp.py` (ρ=0 bit-equality + brute-force conditioning
  tests), `fit_t_em` robust wrapper, deploy-time t-predictive and
  completion-guard helpers.
- Day 3: draft.tex Future-Work passages updated from measured needs to
  measured answers (two measured-wrong recommendations corrected: floors,
  Mahalanobis-alone); `meeting_qa_prep.md` rewritten for this meeting;
  short professor note drafted. Meeting asks: the corpus/pool decision
  for a future preregistered correlated-τ claim; blessing of the opt-in
  adoption defaults.

## DONE — Exploration week (2026-09-10/11, Ray's mandate)

Seven development studies, all shipped, none touching registered
artifacts (ledger: `results/exploration_week_2026-09.md`):
timing tail defused at both levels (deploy-time t; EM t with the
articulation-spillover discovery; floors measured dead), the failed C4
claim's follow-up measured and hardened to registration-grade (AR(1) τ
noise: both axes, three independent runs), the adaptation boundary
closed (coverage test + disagreement guard), and the spectral-kernel
question fully closed — Slot A dead (kill test), Slot B no-gain
(`results/phase3_smprior_dev.md`), learned filter declined by the
evidence (`results/spectral_bump_dev.md`). The kernels' contribution is
conceptual, not numerical.

## DONE — Consolidate (Ray, 2026-09-04)

Work through `docs/gp_curriculum.md`, in order, one stage per sitting.
Exit criterion: the Stage-6 one-sheet test (redraw the whole model from
memory, every arrow annotated).

- [ ] Stage 1  GP regression from zero (Alvarado §2 + draft §3.6)
- [ ] Stage 2  Kernels and the evidence (Wilson & Adams §1–2 + draft §3.3)
- [ ] Stage 3  The spectral view, SM kernel (Wilson & Adams §3–4.1)
- [ ] Stage 4  Non-stationarity, GSM (Remes §1, §2.2, §5.1)
- [ ] Stage 5  Music-audio assembly (Alvarado §2.2–3 + draft §3.10 opening)
- [ ] Stage 6  Multi-output and graphs, ours (draft §3.2–3.6 + Borovitskiy)
- [ ] One-sheet test passed

Pacing guide: stages 1–2 are the foundation and worth two sittings each
if needed. Self-check answers are at the bottom of the curriculum file.
Questions raised while studying are welcome any time and are never
treated as edit requests.

## DONE — Simplify the thesis draft (executed 2026-09-04)

The six-move restructure of `docs/thesis_confusion_audit.md` is executed,
one move per commit (cc5c801..ec917c9 on worktree-audit-week-0813): main
line = Intro / Background / Model (Phase-1 only) / Data (+URMP) / Phase-1
Results / Downstream / **Phase-2 Results (new Ch 7)** / Discussion /
Future / Conclusion, pp. 4–45; appendices A–G hold notation, two-stage
lineage, methodology record, Phase-1 + Phase-2 dev studies, Phase-3, and
extra tables (pp. 46–70). Verification passed: number-multiset audit
before/after = zero numbers lost; dev/confirmation labels intact; compile
clean x2, 0 overfull; page map re-verified from the compiled ToC; prep
pointers re-synced (`docs/meeting_qa_prep.md`); 179 tests green. The main
line is ~42 pp, not the audit's ~30 estimate — the Phase-2 chapter keeps
its mandated content (~10 pp); further shrinking would mean deleting, which
the rule forbids.

## CLOSED — The spectral-kernel modelling step (both slots measured)

Slot A (estimator v2) died by its pre-committed kill test on 2026-09-04
(`results/sm_estimator_dev.md`); Slot B (the Phase-3 within-note curve
prior) was measured on 2026-09-10 and adds nothing at matched rank
(`results/phase3_smprior_dev.md`: paired +0.112* against, harm where the
scaffold rate is unreliable); a learned filter on the graph spectrum is
switched off by the evidence on 90% of cells
(`results/spectral_bump_dev.md`). GSM stays parked; any Slot-B revival
should marginalize the band location first. The reviewed papers' value to
the thesis is the concepts they forced (curve-level scoring, the
coherence bound, the realized-vs-process estimand rule).

## Parked (done or deliberately idle — not to be picked up unprompted)

- Phase 1 and Phase 2: confirmed, one-shots spent, records in
  `results/`. The Phase-2 pool is gone; never rerun.
- Phase-3 development studies: recorded in `results/phase3_*.md`;
  frontier documented in the thesis.
- Correlated-τ registration: registration-grade dev evidence
  (`results/corrnoise_tau_dev.md`); waits for the corpus/pool decision —
  a meeting ask.
- Tonal-metric registration: drafted with power check
  (`docs/phase2_tonal_prereg_DRAFT.md`); waits for the same corpus
  decision.
- Meeting materials: `docs/meeting_qa_prep.md` (to be rewritten on Day 3
  for the current meeting) and `docs/slides/deck_kernels.pdf`.
