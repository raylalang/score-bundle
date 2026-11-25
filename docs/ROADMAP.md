# Roadmap

Updated 2026-11-19. Submission tentatively **2027-02-02**; progress
presentation expected around **2027-01-07**. **From 2026-10-23 the professor is
hands-off; every gate below is Ray's. Writing freeze in force.** One ongoing task at a time. Order stays deliberate:
understanding, then legibility, then modelling. New studies run only on
Ray's explicit mandate (the 2026-09-03 rule stands); the September
exploration week and the October waveform studies were both mandated.

Vocabulary: the thesis and the deck use the content names — the
**piano**, **strings-and-winds**, and **waveform** settings, "stages" for
the chronology, the music model named directly. The internal Phase-0..3
numbering survives only in code, results files, and the frozen names of
registered artifacts; this file uses the content names.

## THE PLAN to submission (rewritten 2026-11-17 after Ray's week at a conference)

Ray's calendar (2026-11-17): zemi on **2026-12-03** where about 95 percent
of the results must be done; **no major experiments after December**;
progress talk about **2027-01-07**; submission **2027-02-02**; thesis
presentation about three months out. Feasibility, assessed 11-17: the main
study is complete (piano and strings-and-winds confirmed, two disclosed
re-used one-shots run, thesis 80 pp, clean compile); the open result lines
are the motif study and one waveform one-shot, both bounded, all compute
done by 11-25. Decisions of 11-17 (Ray): the motif study becomes a thesis
chapter with one pre-specified test on untouched performances; the
phrase-scale waveform claim is registered and run on the URMP confirmation
tracks although the power check says 0.6 to 0.7 at best (verdict reported
verbatim either way); the main deck carries the full spine and Ray cuts
the zemi version himself.

| when | what | gate |
|---|---|---|
| October (done) | three spans of the bridge; track-scale attribution; the two re-used-pool one-shots (10-27); thesis cut and makeover (10-28) | — |
| 11-04 (done) | **motif study, stage one, first pilot** (`results/motif_pilot_dev.md`, proposal deck frame 10; dev only, 21 Beethoven first movements, 336 fits): with whole occurrences hidden, annotated motif edges lower the timing log score on those notes by 0.117 nats [−0.245, −0.006] against the time-only graph; articulation RMSE −5 percent; velocity bands too narrow (one shared weight) | — |
| 11-17 (done) | **pilot two**: the per-channel motif weight (the family as its own diagonal coregionalization block), four seeds, both schemes, 672 fits (`scripts/run_motif_pilot2.sh`, H2 in the script docstring); record `results/motif_pilot2_dev.md`. Measured: the velocity penalty is gone under the per-channel block (−0.086*, coverage 0.944) but the timing log-score gain is not kept (−0.005), and on four seeds the shared weight's timing gain is −0.071 ns (the pilot-one star was two seeds); the robust effect is articulation RMSE (−0.035* shared, −0.019* per channel). Registration primary fixed = articulation RMSE, arm = `time_motif` | — |
| 11-18 | MEETING (brief run order). **Stage two done 11-17**: edges from the ISMIR model's held-out predicted occurrences, masks and cells unchanged (H3), `results/motif_stage2_dev.md`: H3 named the per-channel form, which moved timing the wrong way (+0.036 [−0.010, +0.095]), so H3 not supported; the shared form, recorded without a prediction, gave about half of the annotated effect (timing 0.48, articulation 0.4), intervals reaching zero (corrected 11-19 after the Codex audit) | — |
| 11-18 (done) | **waveform one-shot** registered (`phrase-prior-registration-2026-11-18`) and RUN ONCE the same morning, 04:47-10:21, 12 shards: **W1 PASS** (graph − temporal −0.82 cents [−1.59, −0.05] on 48 held-out phrases; dev −1.63, expectation −0.9), W2 tie, W3 ties; `results/phrase_prior_confirmation_results.md`, `evidence/phrase_prior_confirmation/`; thesis `sec:phase3-heldout`, deck slide 31, 88 pp | — |
| 11-20 → done 11-18 midday | **motif registration** frozen and tagged (`motif-registration-2026-11-18`, two days early on Ray's go) (`docs/motif_prereg_design.md`: untouched performances of the 21 pieces, capped at four per piece, unit = piece; primary = articulation RMSE on hidden motif notes with occurrences hidden, fixed 11-17 from pilot two's power table, 0.95 at 18 pieces; arm = the shared weight; the timing log score secondary at 0.33) | Ray's go given 11-17 |
| 11-23 → done 11-18 | **motif one-shot** RUN ONCE 13:43-15:31 (696 fits, 14 shards): **M1 PASS** articulation ΔRMSE −0.0286 [−0.0494, −0.0109] on 18 movements (dev −0.0345, 15 of 18 below zero, per-piece corr. 0.88 with dev); M2 timing log score miss (−0.018, interval covering zero, as the power table said); velocity penalty gone; per-channel form reproduces dev. `results/motif_heldout_results.md`, `evidence/motif_heldout/`; thesis `sec:motif-heldout`, `tab:motif-heldout`, `sec:motif-meaning`, `tab:motif-dev`, `fig:motif-graph`, 89 pp one file; deck slide 39. **Every result line of the thesis is in.** | — |
| 11-24 to 11-29 | thesis: `sec:motif-heldout` and `sec:motif-meaning` from the motif verdict, the held-out sentences in the abstract, intro, conclusion; Ch. 8 held-out section from the waveform verdict (its opening sentence, the stage-table row, the spans caption). **Pulled forward 11-18 (done):** abstract paragraph, intro (RQ paragraph, contributions paragraph, stage-table row, outline), Discussion, Future Work section on the motif family, Conclusion paragraph; 87 pp, checker and compile clean | — |
| 11-30 to 12-02 | deck: the two verdicts on slides 31 and 39 and the Plan row; speaker notes. **Pulled forward 11-18 (done):** the motif section on the main line (divider 33, slides 34-39), Summary bullet, Plan slide to the calendar, notes for every slide, 45 pp. Ray syncs Overleaf and cuts the zemi deck | Ray |
| 11-19 to 11-29 (the week with nothing to wait for) | **Reading and judgement, no compute.** (1) **DONE 11-19:** Codex audit, two passes, 33 + 8 findings all verified and fixed (d15c6e0, 72a9388), no verdict changed. Codex audit by hand of the material written on 11-18: prompt `docs/codex_audit_prompt_2026-11-19.md` (numbers against records, claim strength against the fixed rules, equations against the code, disclosure, cross-document consistency, examiner questions); findings land in `docs/codex_audit_findings_2026-11-19.md`, fixes go through the style checker. (2) Ray reads the 92-page draft once as an examiner, marking every sentence he cannot defend in one line. (3) Zemi deck cut from the 45-page deck once the slot length is known. (4) Optional: one held-out results figure for the motif chapter (per-movement articulation effect, development against held-out, the 0.88 correlation). | Ray |
| 11-25 → done 11-25 (Tier 1) | **Whole-thesis audit and trim** — Codex plan `docs/codex_trim_plan_2026-11-25.md`: 32 findings verified and applied, Tier 1 applied (92 → 75 pp, repetition removed, no held-out number lost); Tier 2 (~62 pp: lineage appendix retired, sweep/drift plots and second decomposition figure cut, Future Work folded into Discussion) awaits Ray. (Ray: the draft is far too long; cut what does not serve the thesis statement). Codex prompt `docs/codex_trim_prompt_2026-11-25.md` (audit of the chapters not yet audited, repetition list, keep/compress/move/cut per section, tiers of about 75 and 60-65 pages); its output `docs/codex_trim_plan_2026-11-25.md` comes back to be verified against the records and applied through the style checker, with deck and notes following. Explainer page refreshed 11-25 (https://claude.ai/artifact/NZjcsNTRXZgPnrmEwV2XLZ). | Ray |
| 11-25 → done 11-25 | **Fat audit, round 3** — merged and applied (`docs/thesis_fat_merge_2026-11-25.md`), 75 → 65 pp, 16 correctness items fixed (pooled RMSE dominated by articulation, piano criterion, URMP intervals, motif scope, waveform selection and control), every held-out number kept. (Ray: the thesis is not a compilation of the work; cut process history, protocol restatements, scoping studies, implementation detail, self-commentary). Claude's read `docs/thesis_fat_audit_claude_2026-11-25.md` (about 9 pp, 4 correctness items incl. a stale 'no annotation access' caption); Codex reads independently with `docs/codex_fat_prompt_2026-11-25.md` into `docs/codex_fat_audit_2026-11-25.md`; the revision merges both. | Ray |
| 11-25 → done 11-25 | **Final line edit** — fluff and jargon cut, terms unified (NLL for log score, the two-stage pipeline, hidden notes vs held-out sets, temporal chain, time-only graph, common offset $c_0$, boundary not joint), variables disambiguated ($n_c$ channels, $\sigma_x^2$ audio noise, $V_{\mathrm{ref}}$, type$(k)$, hidden note $i$, AR(1) noise via $\sigma_{e,i\tau}$), every symbol and abbreviation defined (pYIN, NLLS, DTW, AUROC, MAP, PIT, GMRF, SPDE; URMP, Vienna, pYIN cited), RQ links at chapter openings, Conclusion answers RQ2/RQ1/RQ3/motif in order; Oxford -ize; 66 pp, compile with zero overfull boxes; canonical terms recorded in `docs/thesis/style_guide.md` | done |
| 11-25 → done 11-25 | **Deck and speaker notes brought in line with the thesis** — same canonical terms and variables, bookkeeping vocabulary and dates out of the slides, prose semicolons and the em-dash removed; slide 18 corrected: it presented the GP estimator (estimator v2, dropped in September) as the channel estimator, the held-out results used the per-note NLLS sinusoid fit; the two-component kernel now introduced on slide 24 as the waveform curve prior; Plan slide brought to current state; speaker notes rewritten in full (no process history, no code identifiers); 45 pp, no new overfull boxes | done |
| **12-03** | **zemi** (95 percent of the results done) | — |
| December | writing only: Codex review by hand, main deck to bullet style, Limitations, figures; no new runs | — |
| 2027-01-07 (approx.) | **progress presentation** from `deck.tex` | — |
| 2027-02-02 | **thesis submission**; presentation about three months from 11-17 | — |

Not run, by decision of 11-17 (Future Work only): the transient model at
the note boundary, the whole-phrase joint loop (the former "December
row"), the scalar-vs-curve channel decision for the waveform strand, the
music-model embeddings inside the motif study (the claim is graph against
graph on identical features, so the embeddings are not needed for it).
The decomposition framing was adopted on 10-23 (the former "December
(proposed)" row).

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

## NOW — The last results (11-17 to 11-25), then writing

The calendar above is the NOW list; the sections below are the record.

## Direction and scope (meeting brief dated 2026-10-14, decided 10-23)

`docs/meeting_qa_prep.md` is the direction-and-scope brief (sections
A–H); `docs/slides/deck.pdf` (37 pages, the one consolidated thesis
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
   (Bach10 / disclosed URMP re-use / both) — DECIDED 10-23, RUN 10-27.

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

## DONE — The second and third spans (2026-10-20, Ray's go 2026-10-18: "B and C")

- **B, boundaries** (`scripts/eval_boundary_prior_dev.py`,
  `results/boundary_prior_dev.md`; `fit_chain(b_prior_cov=, drop_joints=)`
  test-pinned): 47 chains, hidden middle joint removed from the design,
  offset scalar in every arm. Arms tie (graph vs indep −2.45 ms
  [−8.38, +3.32]); all beat the tempo line by 10–14 ms* (the common
  offset); clean deployment identical at 19.6 ms. Calibration: masked
  bands too wide (σ_b = population scale), observed-joint bands too
  narrow (0.7 ms, the estimand gap at the boundary).
- **C, phrase scale** (`scripts/eval_prior_value_phrase_dev.py`,
  `results/prior_value_phrase_dev.md`): 48 cells from 30 of the 78 dev
  tracks, hidden pitch recurring ≥2 notes away. Primary contrast
  passes (graph vs temporal −1.63* cents, 65%), but graph = indep
  (−0.07) and temporal < indep (+1.56*): the win is the temporal
  prior's harm repaired; no dose-response on the edge weight.
- **Three-span verdict**: weak form of the hypothesis survives (score
  structure ≥ naive temporal structure), strong form fails so far
  (nothing beyond the common offset). Thesis `sec:phase3-priorvalue`,
  brief §D.4/F.1/G, CLAUDE.md updated.

## DONE — Track scale and the band fix (2026-10-22/23, Ray: "all experiments in parallel")

- **E1** (`eval_phase2_real.py run-chain`, `results/phase2_chain_prior_dev.md`):
  the strings-and-winds masked evaluation with a time-only chain arm,
  all arms with the per-piece bias column. Score graph beats the chain
  beyond the common offset: intonation −0.50*, extent −0.22*, rate
  −0.21*, loudness −0.007*; chain vs no-graph mixed.
- **E1b** (`run-timegauss`, `results/phase2_timegauss_prior_dev.md`):
  the score graph with its pitch term removed EQUALS the score graph
  on intonation (−0.02 ns) and is slightly better on vibrato; the
  time-Gaussian graph reproduces the confirmed graph value (−0.878 vs
  −0.896). **The graph's value is the time window (2 beats on piano, 2 seconds on
  strings and winds, whose onsets are annotated in seconds), not pitch
  proximity**, consistent with the three waveform spans.
- **E3** (`BOUNDPRIOR_SIGMA=chain`, `results/boundary_prior_chainsig_dev.md`):
  chain-scale prior width: independent band 1.00 → 0.88, structured
  0.75; verdict unchanged. One sentence in the thesis.
- **Declined** (reasons in the CLAUDE.md state dump): the curve-level
  waveform test; the MIDI-DDSP attribute comparison.
- **Piano counterpart** (`eval_graphgp.py --configs b_featlm_timegauss`,
  `results/piano_timegauss_dev.md`): the headline model with the pitch
  term removed is slightly better (RMSE −0.008*, NLL −0.019*, 120
  cells; articulation carries it). Same answer on both corpora.

## DECIDED 2026-10-23 (Ray; the professor is hands-off from here)

- "The graph's value is smoothness in score time over about two beats"
  (wording corrected 11-19 after the Codex audit: two beats on piano, two
  seconds of performance time on strings and winds)
  is the thesis's STATED attribution (abstract, contributions,
  Discussion, Conclusion); the confirmed headline models keep their
  score graph (no protocol change, no new registration). The ℓ_b sweep
  is not run.
- The decomposition framing (score + expression + timbre + noise) is
  ADOPTED in the abstract, the introduction, the conclusion and deck
  slide 3.
- Two more registered one-shots, DONE 2026-10-27, on the disclosed
  re-used 13-piece URMP confirmation pool: correlated timing noise
  (`docs/corrnoise_prereg_design.md`, frozen; PASSED, the failed timing
  claim repaired) and the tonal metric (`docs/phase2_tonal_prereg_design.md`,
  frozen; calibration primary MISSED, recovery held, stays exploratory).
  Records `results/corrnoise_confirmation_results.md`,
  `results/phase2_tonal_confirmation_results.md`. Bach10 is NOT
  available: the archive on disk is the Sibelius-resynthesised
  version (synthetic audio), unusable for intonation or timing claims.
- Writing freeze NOW (2026-10-23), not January: every number audited
  against its record, every forward-looking sentence resolved, Codex
  review, Overleaf sync. The transient model at the joints and the
  whole-performance joint loop stay Future Work.

## DONE — The second and third spans (2026-10-20, Ray's go 2026-10-18: "B and C")

- **B, boundaries** (`scripts/eval_boundary_prior_dev.py`,
  `results/boundary_prior_dev.md`; `fit_chain(b_prior_cov=, drop_joints=)`
  test-pinned): 47 chains, hidden middle joint removed from the design,
  offset scalar in every arm. Arms tie (graph vs indep −2.45 ms
  [−8.38, +3.32]); all beat the tempo line by 10–14 ms* (the common
  offset); clean deployment identical at 19.6 ms. Calibration: masked
  bands too wide (σ_b = population scale), observed-joint bands too
  narrow (0.7 ms, the estimand gap at the boundary).
- **C, phrase scale** (`scripts/eval_prior_value_phrase_dev.py`,
  `results/prior_value_phrase_dev.md`): 48 cells from 30 of the 78 dev
  tracks, hidden pitch recurring ≥2 notes away. Primary contrast
  passes (graph vs temporal −1.63* cents, 65%), but graph = indep
  (−0.07) and temporal < indep (+1.56*): the win is the temporal
  prior's harm repaired; no dose-response on the edge weight.
- **Three-span verdict**: weak form of the hypothesis survives (score
  structure ≥ naive temporal structure), strong form fails so far
  (nothing beyond the common offset). Thesis `sec:phase3-priorvalue`,
  brief §D.4/F.1/G, CLAUDE.md updated.

## CLOSED 2026-10-23 — after three spans

(a) taken: the three-span verdict plus the track-scale attribution is
the waveform chapter's measured answer; (d) was run (band fix,
partial); (b) and (c) are not run (writing freeze).

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
zemi deck frozen as `deck_zemi_v1`). The decomposition framing (score +
expression + timbre + noise) was held for the meeting and ADOPTED on
2026-10-23.

## DONE — Package, adopt, prep (2026-09-16..18)

Adoption path (Ray): the exploration week's inference-level winners are
opt-in capabilities, default off, published paths bit-untouched
(`gp.noise_corr` + `gp.posterior_observations`, `gp.fit_t_em` scored
with `metrics.student_t_nll`, `downstream.completion_guard`; ρ=0
bit-equality and brute-force conditioning pinned in
`tests/test_adoption_optin.py`). BH pass (all main stars survive),
power table, `docs/corrnoise_prereg_design.md` freeze-ready, house-style
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
- Correlated-τ registration: SPENT 2026-10-27 on the disclosed re-used
  pool, N1/N2 PASS (`results/corrnoise_confirmation_results.md`); adoption
  as the timing channel's default noise model would need its own
  registration. Never rerun.
- Tonal-metric registration: SPENT 2026-10-27, T2 FAIL / T1 PASS
  (`results/phase2_tonal_confirmation_results.md`); exploratory status
  kept. Never rerun.
- Decomposition framing in the intro/abstract: ADOPTED 2026-10-23.
- Joint loop over all notes under the graph prior, transient columns at
  the joints, amplitudes under the prior: thesis Future Work
  (`sec:future-phase3`); the first span is measured, the NEXT item above
  is the next.
- Historical meeting materials: `docs/zemi_prep.md`, `deck_zemi_v1`,
  `deck_week`, `deck_kernels` — git history and frozen copies only.
