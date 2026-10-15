# Roadmap

Updated 2026-10-15. Submission tentatively **2027-02-02**; progress
presentation expected around **2027-01-07**. One ongoing task at a time. Order stays deliberate:
understanding, then legibility, then modelling. New studies run only on
Ray's explicit mandate (the 2026-09-03 rule stands); the September
exploration week and the October waveform studies were both mandated.

Vocabulary: the thesis and the deck use the content names — the
**piano**, **strings-and-winds**, and **waveform** settings, "stages" for
the chronology, the music model named directly. The internal Phase-0..3
numbering survives only in code, results files, and the frozen names of
registered artifacts; this file uses the content names.

## THE PLAN to submission (restored 2026-10-15 from the 2026-10-14 session)

The plan the deck's Plan frame and the 2026-10-14 session carried, in one
place, with the decision gates that are Ray's. Rows marked *(proposed)*
are sequencing suggestions, not agreed; everything else was stated in
the deck or the brief. New runs happen only on Ray's go.

| when | what | gate |
|---|---|---|
| October (done) | first curve-from-waveform studies at scale; the curve as a bundle channel; joint note boundaries; **the first span of the bridge**, measured three times: the first run withdrawn by erratum (window leakage), the corrected protocol shows structure transfers the shared tuning offset and nothing measurable beyond it on short chains (a rigorous negative) | — |
| October (now) | the direction-and-scope meeting: objective, positioning, integration scope, variables, evidence bar, registration pool (brief §H) | Ray + professor |
| October (done) | tuning-scalar follow-up of the prior-value experiment: run under the corrected protocol; verdict = tuning bookkeeping, coverage repaired | — |
| November | decide the registration pool (correlated timing noise, tonal metric: Bach10 / disclosed URMP re-use / both), then run the registered timing confirmation, one shot | professor |
| November | decide the channel design for the waveform strand: scalars or curve level | Ray |
| November | a transient model at the note boundary (the pre-named next lever for the ~18 ms acoustic-vs-musical gap) | Ray's go |
| November *(proposed)* | the registered waveform claim: the masked-imputation design is clean and registration-friendly, but the measured first span is a negative, so there is currently no positive waveform claim to register; decide which follow-up (NEXT) could supply one, or whether the negative itself is the chapter's measured answer | Ray |
| December | **whole-phrase joint inference from the waveform** — the "December row": all notes of a performance under the graph prior, the transient model at the joints, loudness/articulation (amplitudes) brought under the prior instead of read out post hoc, the estimand bridge between the harmonic constant and the annotation functional; consolidate thesis chapters as results land | — |
| December *(proposed)* | adopt the decomposition framing (score + expression + timbre + noise) in the intro/abstract, held since 2026-10-14 for after the meeting | Ray |
| 2027-01-07 (approx.) | **progress presentation**: deck from `deck.tex`, results that stand on their own only | — |
| January *(proposed)* | writing freeze: no new runs after the progress talk; Discussion / Limitations / Conclusion finalized; Codex review pass; Overleaf synced | Ray |
| 2027-02-02 (tentative) | **thesis submission** | — |

Decisions still open from the 2026-10-14 session, verbatim list:
(a) tuning-scalar follow-up — DONE 10-15, verdict negative; (b)
masked-imputation as the registered waveform claim — see the November
row; (c) the brief's H questions; (d) whether
the piano chapter should also carry explicit independent/temporal
baseline rows under the final model in the main line (flagged, not
fixed); (e) whether "decomposition" becomes the intro's literal
vocabulary. The worktree branch is merged (2026-10-15); its raw cell
pickles and shard logs are gitignored and still live only in
`.claude/worktrees/audit-week-0813` — copy or archive before removing it.

## NOW — Direction and scope (meeting brief dated 2026-10-14)

`docs/meeting_qa_prep.md` is the direction-and-scope brief (sections
A–H); `docs/slides/deck.pdf` (44 frames, the one consolidated thesis
deck) is its visual spine. The decisions it asks the professor for, in
priority order:

1. the overarching objective — an interpretable, score-aligned
   representation of expressive performance with uncertainty;
2. positioning — evaluate the inverse-inference methodology, with the
   representation's usefulness as motivation, not a validated claim;
3. how much graph-prior × waveform integration a master's thesis needs
   (full joint loop vs the three-prior comparison on short chains);
4. which variables to claim from audio first (pitch/vibrato, boundaries,
   loudness/articulation);
5. the evidence bar for the final contribution (one preregistered
   waveform claim vs development studies plus the confirmed stages);
6. the registration pool for the correlated-τ and tonal-metric follow-ups
   (Bach10 / disclosed URMP re-use / both).

Until those are decided: no new modelling runs. Open housekeeping:
record the meeting outcome here and in the brief once it exists; sync
Overleaf with main before the professor reads the draft (74 pp,
restructured and reframed on 2026-10-14); `docs/slides/README.md` still
describes the 16+5-frame content-first build and `docs/zemi_prep.md`
the 28-slide zemi deck — both are superseded by `deck.tex`.

## DONE — The first span of the bridge, measured three times (2026-10-14/15)

The brief's proposed experiment (`scripts/eval_prior_value_dev.py`;
cross-note prior over the per-note intonation constants inside
`fit_chain`): independent / temporal-chain / score-graph priors at
matched marginal variance, identical likelihood, data, optimizer;
boundaries fixed; ambiguity = one note removed per chain; 47 chains on
4 development tracks; hypotheses + falsification rule committed before
every run. Development, no claims.

- **First run (10-14, `results/prior_value_dev.md`): WITHDRAWN by
  erratum.** It masked the hidden note's audio samples but left its
  soft-window design columns in the likelihood; the leak dominated the
  prior and every arm's estimate was uncorrelated with the truth.
- **Protocol v2 (10-15, `results/prior_value_v2_dev.md`)**: the hidden
  note's columns removed (`fit_chain(drop_notes=)`, test-pinned);
  structure halves the error, 14.4 → 6.8/7.6 cents (−5.8*/−6.2*, 79%);
  graph = temporal; recurring-pitch stratum trend ns.
- **Protocol v2 + free per-chain tuning scalar in every arm
  (`results/prior_value_v2_tuning_dev.md`)**: all arms tie (graph vs
  indep −0.02 [−0.33, +0.32]); the falsification rule fires — **on
  short chains, score structure adds nothing measurable beyond the
  shared tuning offset**. Clean ties; conditional bands calibrated.

Thesis `sec:phase3-priorvalue` carries the corrected, two-sided story;
contributions, RQ3, abstract, Future Work reworded. Brief §D.4/F.1/G
rewritten. `docs/boundary_prior_proposal.md` written (not run; the
offset scalar is its core control).

## NEXT (needs Ray's go) — where could note-level structure show?

The first span's negative narrows the question. Candidates, each with
the common-mode term in every arm by design, each a proposal before a
run: (a) the boundary-prior extension (proposal written, same null
expected, establishes timing rigorously); (b) a phrase-scale intonation
comparison where recurring pitches occur at a distance (the score
graph's distinguishing edge needs that); (c) the other channels
(vibrato, dynamics) at chain scale. Ray chooses; none runs unprompted.

## DONE — The waveform setting goes active (2026-10-07..14, Ray's go)

After the zemi (2026-10-08) and the supervisor's pointer to
differentiable audio generators, the DDSP direction was scoped
(`docs/ddsp_review.md`: gradients through our own synthesizer, the exact
amplitude-marginalized likelihood and the GP priors kept, no neural
encoder, no spectral surrogate loss). All development, no claims,
nothing registered touched:

- Packaged: `src/score_bundle/phase3/curve.py` (torch, import-guarded;
  gradient MAP + full joint Laplace; numpy-path equality test-pinned;
  `fit_note`, `fit_chain`, `CurveFit.average`).
- Curve prior evidence-fit (`results/phase3_prior_fit_dev.md`): mixed at
  n=3, wins at n=376 → dev default.
- Batch curve study (`results/phase3_curve_dev.md`, 376 notes): median
  3.16 cents vs pYIN frames 4.09; winds win decisively, strings still
  favour the tracker; parameter-band coverage 0.32–0.36 = the estimand
  gap at curve level.
- Curve as the bundle's waveform channel
  (`results/phase3_curve_integration_dev.md`): −1.36* on 14/14 pairs;
  tie vs the scalar read-out; discrepancy floor halves (3.09 → 1.80).
- Boundary pilot (`results/phase3_boundary_pilot_dev.md`): gradients
  through soft windows converge start-independently; the optimum sits
  ~25 ms from the annotation (acoustic vs musical onset).
- Chain study (`results/phase3_chain_dev.md`, 47 chains / 154
  boundaries): the transition-REGION hypothesis rejected productively
  (widths collapse to 1 ms yet beat the fixed-width control on 77%);
  edge error ~18 ms vs the pilot's 25; **deployment works** — from
  score-nominal tempo-line inits with no annotation access, median
  boundary error 39.4 → 19.5 ms. Chains cost ~1.8 cents guarded curve
  RMSE vs single-note fits. Next lever (pre-named): transient/broadband
  columns at the joints.
- Status at the end of these studies (superseded the same evening, see
  the next block): the graph prior was not yet inside the waveform
  inference; it met waveform output only post hoc, as one bundle channel.

## DONE — Thesis restructure and refocus (2026-10-14)

Three moves on the draft, additive/wording only, decimal multiset
checked: the waveform work is a main-line chapter (Ch. 8, with the
chain, deployment, and end-goal figures); results chapters are
content-named; "Phase N" wording discontinued thesis-wide (one
traceability sentence in the Introduction maps the internal numbering
to the content names; internal labels and figure filenames unchanged).
Then the refocus: abstract and motivation state the objective/method
split, the problem statement carries p(z|x,S) ∝ p(x|z,S) p(z|S) with
the progressively-rawer observation ladder, a three-question RQ
hierarchy replaces the single central question, related work gains the
inverse-analysis-vs-DDSP positioning (three verified citations).
`deck.tex` consolidated as the one full thesis deck (44 frames; the
zemi deck frozen as `deck_zemi_v1`). Held for after the meeting (Ray's
call): adopting the decomposition framing (score + expression + timbre
+ noise) in the intro/abstract.

## DONE — Package, adopt, prep (2026-09-16..18)

Adoption path (Ray): the exploration week's inference-level winners are
opt-in capabilities, default off, published paths bit-untouched
(`gp.noise_corr` + `gp.posterior_observations`, `gp.fit_t_em` scored
with `metrics.student_t_nll`, `downstream.completion_guard`; ρ=0
bit-equality and brute-force conditioning pinned in
`tests/test_adoption_optin.py`). BH pass (all main stars survive),
power table, `docs/corrnoise_prereg_DRAFT.md` freeze-ready, house-style
figures, the interface reframe (channels ARE sine parameters; the story
is two-sided).

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
memory, every arrow annotated). The checklist below is Ray's own record.

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
line = Intro / Background / Model / Data (+URMP) / piano results /
Downstream / strings-and-winds results (new Ch 7) / Discussion / Future
/ Conclusion; appendices hold notation, two-stage lineage, methodology
record, the development studies, and extra tables. Verification passed:
number-multiset audit before/after = zero numbers lost; dev/confirmation
labels intact; compile clean x2, 0 overfull; 179 tests green. (The
2026-10-14 restructure above then promoted the waveform chapter to the
main line: 74 pp.)

## CLOSED — The spectral-kernel modelling step (both slots measured)

Slot A (estimator v2) died by its pre-committed kill test on 2026-09-04
(`results/sm_estimator_dev.md`); Slot B (the within-note curve prior in
the waveform rig) was measured on 2026-09-10 and adds nothing at matched
rank (`results/phase3_smprior_dev.md`: paired +0.112* against, harm
where the scaffold rate is unreliable); a learned filter on the graph
spectrum is switched off by the evidence on 90% of cells
(`results/spectral_bump_dev.md`). GSM stays parked; any Slot-B revival
should marginalize the band location first. The reviewed papers' value to
the thesis is the concepts they forced (curve-level scoring, the
coherence bound, the realized-vs-process estimand rule).

## Parked (done or deliberately idle — not to be picked up unprompted)

- Piano and strings-and-winds: confirmed, one-shots spent, records in
  `results/`. The URMP confirmation pool is gone; never rerun.
- Correlated-τ registration: registration-grade dev evidence
  (`results/corrnoise_tau_dev.md`); waits for the corpus/pool decision —
  meeting ask 6.
- Tonal-metric registration: drafted with power check
  (`docs/phase2_tonal_prereg_DRAFT.md`); waits for the same decision.
- Decomposition framing in the intro/abstract: held for after the
  meeting (Ray's call).
- Joint loop over all notes under the graph prior, transient columns at
  the joints, amplitudes under the prior: thesis Future Work
  (`sec:future-phase3`); the first span is measured, the NEXT item above
  is the next.
- Historical meeting materials: `docs/zemi_prep.md`, `deck_zemi_v1`,
  `deck_week`, `deck_kernels` — git history and frozen copies only.
