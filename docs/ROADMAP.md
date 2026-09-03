# Roadmap

Updated 2026-09-04. One ongoing task at a time. The professor update comes LAST, after the work is done, not before. Order is deliberate:
understanding, then legibility, then modelling. New results work is
parked until all three say otherwise.

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

## THEN — The spectral-kernel modelling step (only after NEXT, and after discussing with the professor)

Bring the reviewed kernel families into the model as the within-note
curve prior. Decision inputs, all already written: the review
(`docs/kernel_papers_review.md`), the unification memo with its honest
costs and the minimal cheap test (`docs/gp_everywhere_memo.md`), and
the slot choice (curve-level estimator vs the audio model's curve
prior). Any evaluated claim gets its own registration.

## FINALLY — update the professor

Only after the work above is done. A short declarative note (drafts
exist in the 2026-09-04 session; plan-only, no attributions): what was
consolidated, the simplified draft, and the modelling direction. No
update is sent before then.

## Parked (done or deliberately idle — not to be picked up unprompted)

- Phase 1 and Phase 2: confirmed, one-shots spent, records in
  `results/`. The Phase-2 pool is gone; never rerun.
- Phase-3 development studies: recorded in `results/phase3_*.md`;
  frontier documented in the thesis; idle until THEN.
- Tonal-metric registration: drafted with power check
  (`docs/phase2_tonal_prereg_DRAFT.md`); waits for a corpus decision.
- Meeting materials: `docs/meeting_qa_prep.md` (lit-review framing,
  measured results under RESERVE) and `docs/slides/deck_kernels.pdf`;
  reusable at the next meeting.
