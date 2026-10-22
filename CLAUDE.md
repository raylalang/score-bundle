# CLAUDE.md — score-bundle

Guidance for working in this repo. Read `README.md` and `docs/music_lm_design.md` first.
Codex will review your output once you are done.

## What this is

A research codebase for **Bayesian, score-informed performance transcription**: given a
known symbolic score and a performance, infer per-note expressive variables (timing,
articulation, dynamics; later intonation/vibrato) with calibrated uncertainty. **The
thesis model is one multi-output Gaussian process on the score graph** (GP-first,
`src/score_bundle/gp.py`): channels coupled by coregionalization, side information
(score features, LM embeddings) marginalized into the kernel, everything fit by exact
per-piece evidence; the earlier two-stage pipeline (plug-in mean + per-channel graph
residual) is its nested special case and ablation chain. The model is one generative
process run two ways — forward = synthesis, inverse = transcription. See
`docs/architecture.svg`.

All thesis updates are delivered through `docs/thesis/draft.tex` — there is
no external companion note. (A Notion page existed early on and is RETIRED;
do not create, sync, or reference one.)

## Phases (and what's implemented)

(Vocabulary, 2026-10-14: the thesis and the deck no longer say "Phase N" — they use
the content names: the **piano**, **strings-and-winds**, and **waveform** settings,
"stages" for the chronology, the music model named directly. The Phase-0..3
numbering below survives in code, results files, and the frozen names of registered
artifacts; keep it there, never reintroduce it into `draft.tex` or `deck.tex`.)

- **Phase 0 — music LM (implemented).** From-scratch **PyTorch** decoder-only Transformer
  over note-structured MIDI tokens (`src/score_bundle/lm/`, hand-written causal attention).
  Provides per-note embeddings `h_i` that feed the Phase-1 GP as a feature kernel (in the
  two-stage development form, a plug-in mean `μ_LM`). Tokenizer
  and batching are framework-agnostic NumPy; the model is PyTorch.
- **Phase 1 — core, piano (implemented + evaluated).** Score graph → Gaussian graph prior →
  closed-form posterior with per-note uncertainty; baselines; calibration metrics. Held-out
  ASAP eval (`scripts/eval_asap_robust.py`, `src/score_bundle/imputation_eval.py`) shows
  `LM mean + graph residual` best on RMSE *and* calibration — see
  `docs/phase1_calibration_results.md` (**read the 2026-07-02 correction first**: the LM
  mean must use the *leak-free pre-velocity read-out* (`emb_leakfree`) — a readout at the
  VELOCITY token leaks each note's own velocity into `μ_LM`; any readout fit on observed
  rows must never contain its own target (bidirectional models need the leave-one-out
  readout in `lm/masked.py`) — and the EB fit needs `noise_floor` /
  `noise_floor_frac=0.05`). NB the graph posterior needs a **predictive-variance floor**
  (held-out `y=f+ε` has variance `diag(Σ_y)+noise_var`) or NLL/coverage blow up; the EB
  fit needs the matching **noise_var floor** or a minority of fits collapse; and a rare
  knife-edge marglik collapse (one cell in 120) is screened by
  `fit_laplacian_field_guarded` / `impute_methods(guard=True)` — default off, use it for
  new runs, published tables stay guard-off (measured bit-identical there). Honest
  baselines: `rich_score_features` (`baselines.py`) ties the LM mean on RMSE (the LM's
  edge is calibration + the `v` channel; both survive as feature kernels of the GP);
  Stage-2 masked pretraining is an honest negative at matched budget.
  Downstream tasks (completion / anomaly / denoising):
  `src/score_bundle/downstream.py`, `docs/downstream_tasks_results.md` (re-validated
  under the GP; see the banner there + `scripts/eval_downstream_gpfirst.py`).
  Kernel comparison (2026-07-09, `scripts/eval_kernels.py`,
  `docs/kernel_comparison_results.md`): Matérn/diffusion/normalized-Laplacian all tie
  the additive default (spectral machinery: `prior.SPECTRAL_KERNELS`,
  `model.SpectralGaussianField` — covariance form, since the diffusion precision
  overflows); harmonic chord+voice-leading edges (`graph.build_adjacency_harmonic`)
  win both axes in the two-stage regime; tonal-distance *replacement* of the pitch
  metric hurts.
  **THE THESIS MODEL is GP-first (2026-07-09/10):** one multi-output graph GP
  (`src/score_bundle/gp.py` `MultiOutputGraphGP`; runner `scripts/eval_graphgp.py`;
  primary doc `docs/graphgp_first_design.md`) — ICM coregionalization ⊗
  shape-normalized spectral kernel + feature kernels (score features + mask-aware LM
  embeddings = marginalized Bayesian linear mean) + per-channel floored noise, all by
  exact per-piece evidence. **Preregistered one-shot confirmation** on 20 untouched
  pieces (`logs/confirmation_verdict.log`): RMSE 0.376 vs 0.393 (two-stage best,
  paired −0.014*), graph NLL contribution −0.074*, coverage 0.925; pooled-NLL tie =
  one diagnosed τ-outlier cell (Gaussian-tail limitation; `gp.fit_guarded` exists,
  verified no-op on healthy fits, cannot catch that tail — documented, not patched).
  Attribution (measured): per-piece Bayesian feature weighting recovers, the graph
  calibrates; harmonic edges are redundant under LM-in-kernel; the whole two-stage
  pipeline is the GP's nested special case (unit-pinned) and its ablation chain.
  Boundary: per-piece adaptation fails at excerpt extrapolation (completion) — the
  cross-piece head stays the honest tool there. Dev numbers = development set
  (selection-reused; every model's NLL flatters there). GP path leak-audited bitwise
  (`scripts/audit_graphgp_leakfree.py`). Fits deterministic per BLAS thread count
  only (v2 = OMP_NUM_THREADS=2).
  Post-adoption dev studies (2026-07-13..16, all dev-only, confirmation untouched):
  masking-level sweep 50%→LOO (`scripts/run_mask_sweep.sh`, `eval_gp_loo.py`,
  `docs/masking_sweep_results.md`) — ordering stable at every level; theory-alignment
  audit (`docs/graphgp_theory_alignment.md`) — no unforced deviation from the graph-GP
  literature, "additive" ≡ graph Matérn ν=1; music-theory feature block
  (`rich_score_features(theory=True)`, 14 cols, default OFF) — honest negative, and
  embedding probes (`scripts/probe_embeddings.py`, linear + RFF) show the embeddings
  encode rhythm/voicing/register but NOT tonality; sustain-overlap edge family
  (`build_adjacency_harmonic(overlap_weight=)`, config `c_overlap`) — trend, ns,
  not adopted; calibrated deep baselines (`scripts/eval_deep_baseline.py`,
  `docs/deep_baseline_results.md`) — hetero-MLP + 5-ensemble lose to the GP on both
  axes (−0.090*/−0.30* paired); statistical recheck (`scripts/robustness_recheck.py`)
  — per-channel significance, BCa/Wilcoxon/sign agreement, BH-surviving stars,
  composer-clustered CIs; 30-piece fresh replication set (positions 50–79 of the
  same shuffle, `.cache/asap_arrays_named80.pkl`, identity-gated;
  `scripts/run_replication_set.sh`).
  Aria frozen-feature upper-bound baseline is an import-guarded stub (`lm/aria_baseline.py`).
- **Phase 1 addendum — posterior decomposition (2026-07-31..08-07, in the thesis).**
  Exact per-component split of the posterior (`gp.posterior_components`,
  `posterior_component_cov`, `fit(b_diagonal=True)`; thesis eq:components–eq:decomp-total
  + §5.3, component index ζ). Consolidated record: `docs/posterior_decomposition_results.md`
  — features carry the mean, the graph calibrates, graph×LM components near-orthogonal
  (complements); coupling's value is the velocity channel only; dominance is per-piece
  (ARD switches kernels off/on). Harmonic-edge question **CLOSED**
  (`docs/kernel_multirate_results.md`): a density gradient — wins at ≤30% hidden,
  tie at the 40% operating point, nothing + one guard-invisible collapse at 50% —
  so no adoption, no second confirmation set spent.
- **Phase 2 — intonation/vibrato/loudness/timing/vibrato-delay:
  CONFIRMED (one-shot spent 2026-08-27; the pool is GONE — never rerun).**
  Code: `src/score_bundle/phase2/` — `intonation.py` (pyin `extract_f0`,
  import-guarded; `fit_vibrato_note` ungated for [c, log γ, log f];
  `fit_vibrato_note_gated` = eq:vibrato exactly, supplies δ_vib),
  `urmp.py` (loader 44/44), `splits.py` (**FROZEN** composition-level
  split; confirmation = 13 pieces, SPENT), `targets.py`, `warp.py`
  (onset-anchored LOO warp = τ; OLS predictive variance = its noise row).
  Six-channel bundle [c, log γ, log f, ℓ, τ, δ_vib]; estimator variances
  used AS GIVEN (learned scale = documented fallback, worse-calibrated —
  three times measured); |c|>150 octave rule → missing; cell masks
  everywhere.
  **Registration**: tag `phase2-registration-2026-08-17`
  (`docs/phase2_prereg_design.md`, frozen + one dated erratum); claims
  C1–C4, as-given primary, one shot.
  **Confirmation verdict (2026-08-27,
  `results/phase2_confirmation_results.md` incl. the scored verdict
  section; evidence archived `evidence/phase2_confirmation/`):**
  C1 intonation recovery PASS (−0.877*, dev −0.891* reproduced);
  C2 vibrato calibration PASS both channels (−2.990*/−0.564*; the
  flagged seed-sensitive extent star HELD); C3 coverage PASS (0.88–0.91
  all six); C4 timing calibration FAIL (−0.030 ns; plus starred adverse
  τ recovery +0.003*) — reported verbatim. Headline CONFIRMED by the
  registered rule. Thesis §3.9 carries it (tab:phase2-conf).
  Dev record: `results/phase2_real_results.md` (registration-grade,
  diff-clean vs tag). Post-adoption dev studies, all results-doc'd:
  drift study (`results/phase2_drift_dev.md` — committee's "sine too
  simple" quantified; drift real but graph-white → no new channel);
  tonal metric exploratory win on intonation
  (`results/phase2_tonal_dev.md`; adoption = own registration; power
  check in `docs/phase2_tonal_prereg_DRAFT.md`: calibration-primary,
  recovery underpowered <~20 pieces); fresh-seed robustness
  (`results/phase2_seeds23_dev.md`); δ_vib decision
  (`results/delta_vib_dev.md`); tracker calibration
  (`results/tracker_calibration_dev.md`).
- **Phase 3 — the waveform as the observation: ACTIVE since 2026-10-09 (dev
  studies only, all exploratory, no claims; nothing registered touched).**
  `src/score_bundle/phase3/`: `synth.py` + `waveform_model.py` = synthesis basis
  Φ(z) + exact amplitude marginalization incl. the O(mp²) Woodbury form
  (`collapsed_loglik_lowrank`, dense-equality unit-pinned); `curve.py` = the
  differentiable step of the DDSP direction (`docs/ddsp_review.md`: gradients
  through OUR synthesizer, the exact marginalized likelihood and the GP priors
  kept; no neural encoder, no spectral surrogate loss) — PyTorch, import-guarded,
  gradient MAP + full joint Laplace over the within-note curve on a knot grid,
  numpy-path equality test-pinned (`tests/test_phase3_curve.py`,
  `test_phase3_chain.py`); API `fit_note`, `fit_chain`, `CurveFit.average`
  (the realized-average read-out that feeds the bundle). Measured record,
  August: `results/phase3_waveform_feasibility.md` (2-note pilot),
  `results/phase3_waveform_dev.md` (376 notes / 7 tracks: waveform median
  2.29 cents from quasi-truth with NO tracker, beats the estimator chain on
  winds; coverage vs quasi-truth invariant ~0.02 under mean/noise/deviation-prior
  fixes; self-check on model-true synthetics covers at 0.90 exactly ⇒ the gap
  is an ESTIMAND gap), `results/phase3_integration_dev.md` (scalar waveform c
  as 7th bundle channel + visible-notes discrepancy floor: −1.21*; no-floor
  control worse-calibrated ⇒ the floor is necessary). October (all `results/
  phase3_*_dev.md`, runners `scripts/run_phase3_{curve,chain}_eval.sh`,
  sharded): evidence-fitted curve prior = dev default (`phase3_prior_fit_dev`,
  wins at n=376); batch curve study (`phase3_curve_dev`: median 3.16 cents vs
  pYIN frames 4.09, winds win, strings favour the tracker, parameter-band
  coverage 0.32–0.36 = the estimand gap at curve level); curve read-out as the
  bundle's waveform channel (`phase3_curve_integration_dev`: −1.36* on 14/14
  pairs, floor halves 3.09→1.80 cents); boundary pilot
  (`phase3_boundary_pilot_dev`: soft-window gradients converge
  start-independently; optimum ~25 ms from the annotation = acoustic vs
  musical onset); chain study (`phase3_chain_dev`, 47 chains / 154 boundaries:
  transition-REGION hypothesis REJECTED, sharp edges win; edge error ~18 ms;
  **deployment works** — score-nominal tempo-line inits, no annotation,
  39.4→19.5 ms median; chains cost ~1.8 cents guarded curve RMSE vs
  single-note fits; next lever, pre-named: transient columns at the joints).
  **The first span of the bridge (2026-10-14/15, dev, measured THREE times on
  the same 47 chains): the score graph couples per-note intonation constants
  INSIDE the waveform inference on short chains** — the prior-value experiment
  (`scripts/eval_prior_value_dev.py`, runner `run_prior_value_eval.sh [N]
  [sd|-] [v2]`; independent / temporal-chain / score-graph priors at matched
  marginal variance, same likelihood/data/optimizer, boundaries fixed, one note
  removed per chain vs clean, 4 dev tracks, hypotheses + falsification rule
  committed before each run). **First run WITHDRAWN by erratum**
  (`results/prior_value_dev.md`): masking the audio samples but leaving the
  note's soft-window columns in the likelihood leaked ~4 samples' worth of
  column mass whose marginalized-amplitude evidence dominated the prior —
  every arm's estimate uncorrelated with the truth. Protocol v2 =
  `fit_chain(drop_notes=)` (hidden note acoustically absent, test-pinned:
  independent prior returns exactly its mean). **Verdict**
  (`results/prior_value_v2_dev.md`, `prior_value_v2_tuning_dev.md`): without
  a common-mode term structure halves the hidden note's error (14.4→6.8/7.6,
  −6*, 79%) and graph = temporal; with a free per-chain tuning scalar in every
  arm ALL ARMS TIE (graph vs indep −0.02 [−0.33,+0.32]) → **on 3–5-note chains
  score structure adds nothing measurable beyond the shared tuning offset**
  (a rigorous negative; the falsification rule fired). Conditional bands
  calibrated (0.89–0.96); clean rows tie. Lesson for every future prior-value
  design: the common-mode scalar is mandatory in every arm, and a masked note
  must be dropped from the design, not just its samples. **Second span,
  boundaries (2026-10-20, `scripts/eval_boundary_prior_dev.py`,
  `results/boundary_prior_dev.md`; `fit_chain(b_prior_cov=, drop_joints=)`,
  test-pinned)**: same verdict — arms tie on the hidden joint (graph vs indep
  −2.45 ms ns), all beat the tempo line by 10–14 ms* (common offset), clean
  deployment identical at 19.6 ms; masked bands too wide (σ_b = population
  scale), observed-joint bands too narrow (0.7 ms = the estimand gap at the
  boundary). **Third span, phrase scale (2026-10-20,
  `scripts/eval_prior_value_phrase_dev.py`, `results/prior_value_phrase_dev.md`;
  48 cells from ALL 78 dev tracks, hidden pitch recurring ≥2 notes away)**:
  graph beats temporal −1.63* (the pre-stated primary), but graph = indep
  (−0.07) and temporal < indep (+1.56*) → the win is the temporal prior's harm
  repaired, no dose-response on edge weight. **Three-span verdict: weak form
  survives (score structure ≥ naive temporal), strong form fails so far
  (nothing beyond the common offset).** Still open: the curve level, the other
  channels, whole-performance scale — the curves themselves carry independent
  priors, amplitudes are read out post hoc. Thesis: main-line Ch. 8
  (`ch:phase3`) + Future Work `sec:future-phase3`. The three spectral-kernel
  papers (`related_works/`, `docs/kernel_papers_review.md`,
  `docs/gp_everywhere_memo.md`) are the design vocabulary for the within-note
  curve prior (Slot B measured no-gain, `results/phase3_smprior_dev.md`).
- **Exploration week (2026-09-10/11, dev-only) + adoption (2026-09-16).**
  Seven development studies, ledger `results/exploration_week_2026-09.md`:
  timing tail defused at both levels (deploy-time t predictive; EM t noise
  with a measured articulation spillover via the coupling; predictive
  floors measured DEAD), the failed C4 claim's follow-up measured and
  hardened (AR(1) τ noise row: both axes, three independent runs,
  registration-grade — awaits a corpus decision), the adaptation boundary
  closed (low-rank Mahalanobis + disagreement guard; Mahalanobis alone is
  half the answer), and the spectral-kernel question CLOSED (Slot A dead,
  Slot B no-gain, learned filter switched off by the evidence). Adopted as
  **opt-in** capabilities (the fit_guarded precedent — default off,
  published paths bit-unchanged, any reported-protocol change needs its
  own registration): `gp.fit_t_em` (LOO-EM robust t noise; score its fits
  with `metrics.student_t_nll`, never Gaussian), `gp.noise_corr` +
  `gp.posterior_observations` (correlated per-channel noise, bit-equal to
  the study machinery), `downstream.completion_guard`. Tests:
  `tests/test_adoption_optin.py` (incl. ρ=0 bit-equality and brute-force
  conditioning pins).
- **Thesis state (2026-10-21, main=acd3076+): 76 pp, compile clean.** Main line =
  Intro / Background / Model / Data / piano results / Downstream / strings-and-winds
  results / **the waveform as the observation (Ch. 8)** / Discussion / Future Work /
  Conclusion; appendices A–G. Refocus: objective = an interpretable, score-aligned
  representation with uncertainty; method = Bayesian score-conditioned inverse
  synthesis p(z|x,S) ∝ p(x|z,S)p(z|S) with a progressively-rawer observation
  ladder; three-question RQ hierarchy. `docs/slides/deck.tex` is the one full
  thesis deck (52 pages incl. 5 backup frames, pages 48-52; **every deck.tex change must be
  followed by a change to `docs/slides/speaker_notes.md`, same commit — Ray's
  standing rule 2026-10-21**). `docs/meeting_qa_prep.md` = the 2026-10-14
  direction-and-scope brief (asks A–H); `docs/ROADMAP.md` carries NOW/NEXT. Held
  for after the meeting (Ray's call): the decomposition framing (score +
  expression + timbre + noise) in the intro/abstract.
- **STATE DUMP for context compaction (2026-10-22, main=6f9f52b+; written at
  Ray's request before a `/compact`).** Meeting of 10-21 was cancelled; Ray's
  instruction: "do all experiments you have in mind in parallel; include the
  results in our docs, but only the ones worth considering; you can decline".
  Facts that decide the plan: the strings-and-winds model's feature kernel
  carries a per-piece bias column (`eval_phase2_real.py`: `np.ones((n,1))`
  appended to `rich_score_features`), in BOTH the graph and the no-graph arm,
  so the confirmed intonation gain (−0.877 cents*, graph vs no-graph, 13
  pieces) is already a gain BEYOND the common offset at whole-track scale with
  tracked targets — the opposite of the chain-scale waveform verdict (three
  spans: nothing beyond the common offset). Running (both dev-only, hypotheses
  in the script docstrings before the run, separate artifacts, nothing
  registered touched): **E1 track-scale three priors** = the strings-and-winds
  masked-imputation evaluation with a third arm, a time-only chain graph
  (`gp_chain`, `gp_chain_asgiven`), beside `gp`/`gp_asgiven` and `nograph`, on
  the 78 dev tracks × 6 channels (variant plumbed like the tonal study: own
  cells dir `results/phase2_cells/chain`, own results file
  `results/phase2_chain_prior_dev.md`; runner `run_phase2_eval.sh N chain`).
  Question: at track scale, beyond the common offset, does the score graph
  beat a temporal prior (strong form) or only tie it (weak form)? **E3
  boundary band fix** = experiment B rerun with a chain-scale prior width
  (σ_b² = median over the track's chains of the within-chain τ variance;
  env `BOUNDPRIOR_SIGMA=chain`, prefix `boundprior_chainsig`, results
  `results/boundary_prior_chainsig_dev.md`); calibration-only, expected to
  move masked coverage from 1.00 toward 0.90 without changing the tie.
  DECLINED with reasons: the curve-level waveform test (the chain model has no
  per-note vibrato parameters to couple; the buildable version is a
  chain-scale regression on read-outs, already answered by the three spans)
  and the MIDI-DDSP attribute comparison (TensorFlow setup risk + attribute
  mismatch; poor use of a day during writing). After the runs: readings
  appended to the two records; docs (thesis Ch. 8 / brief / ROADMAP / this
  file / speaker notes if a slide changes) updated ONLY for results worth
  considering; the standing rules (speaker notes follow deck changes; deck =
  minimal narrative + full math; no Phase-N wording in thesis/deck) apply.
  Speaker notes now cover every slide 2–52 (`docs/slides/speaker_notes.md`).
- Real dataset loaders: **MAESTRO** (Phase-0 LM) and **ASAP** (Phase-1 aligned task) are
  **implemented** — `lm/data.py` (`load_maestro_meta`, `maestro_note_events`,
  `iter_maestro_note_streams`, `maestro_split`) and `features.py` (`load_asap_meta`,
  `load_asap`, `asap_performance_variables`, `asap_clean_performances`). Aria-MIDI / ATEPP /
  GiantMIDI loaders remain open. Datasets live under `../data/`
  (`/home/ray/Research/data/{maestro-v3.0.0,asap-dataset}`); pass the root explicitly.

## Canonical notation (keep consistent everywhere — code, docs, thesis)

| Symbol | Meaning |
|--------|---------|
| `S = {s_i}`, `s_i=(p_i,b_i,d_i)` | score support (pitch, beat onset, beat duration) |
| `y_i = [τ_i, log r_i, v_i]` | Phase-1 per-note variables (onset residual, articulation, velocity) |
| `Q_G` | graph prior precision; additive `λI + ηL_G`, or Matérn `σ_g⁻²(κ²I+L_G)^α` |
| `λ, η` | additive ridge term, Laplacian weight |
| `σ_g, κ, α` | Matérn scale, inverse-length, exponent |
| `g(ν)`, `t` | spectral kernel: covariance eigenvalues of `L_G`'s spectrum; diffusion time (`K = σ_g² exp(−t L_G)`) |
| `B`, `c_f`, `g(ν; s)`, `ς²` | GP-first: coregionalization matrix; feature-kernel scales; shape-normalized spectral kernel (g(0)=1, shape `s`); per-channel noise |
| `Σ_e` | observation-noise covariance; `Σ_y` posterior covariance; `m` posterior mean |
| `σ` | **posterior standard deviation only** (not a prior scale) |
| `μ_LM`, `h_i` | LM-predicted plug-in mean (development form), LM per-note embedding (GP feature kernel) |
| `ζ`, `y^(ζ)`, `C^(ζ)`, `m^(ζ)` | posterior-decomposition component index (`ζ ∈ {G, feat, emb}` in Phase 1; `ζ′` the paired index — NOT a/b, which are amplitudes/beat onset), component draw/prior covariance/posterior mean |
| `ω^(ζ)`, `ρ^(ζζ′)` | covariance share of a component (sums to 1); per-note component correlation |
| `J_i`, `π_j`, `q_0.2`, `f_ref` | Phase-2 frame rule: note-i frame set, tracker per-frame confidence, lowest-quintile threshold, tuning reference 440 Hz = A4 (MIDI 69) |
| `φ̂_i, ψ̂_i` | Phase-2 local tempo-line slope/intercept at note i (eq:localwarp) |
| `z, a, Φ(z), x, ε, A_i(t)`, `H` | Phase-3 positions, amplitudes, synth, audio, noise, amp envelope, harmonic count |

Do **not** reuse `S` for a covariance, `σ` for a prior scale, `α` for the additive weight,
or `a_i(t)` for the amplitude envelope. (These were deliberately disambiguated.)
Thesis renames (2026-09-04 confusion audit; code identifiers and results docs
unchanged): the vibrato onset delay is typeset `d_i^vib` (was `δ^vib`), the
tracker confidence `π_j` (was `P_j`), the Phase-3 harmonic count `H` (was `K`),
and the chord+voice-leading graph is the **chord graph** (was "harmonic graph").

## Conventions

- **NumPy-first statistical core.** The Phase-1 package (graph / prior / model / **gp** /
  metrics /
  tokenizer / data) must import and test with **numpy only**; `scipy` and `scikit-learn` are
  optional and import-guarded. The **Phase-0 LM is PyTorch** (`lm/model_torch.py`) —
  import-guarded so the package still imports without torch, but training and the LM tests
  require it (`pip install -e '.[train]'`). Never add a hard dependency to the Phase-1 core.
- **src layout.** Run things with `PYTHONPATH=src` or `pip install -e .`.
- Dataclasses for structured data; clear docstrings that tie modules back to the concept
  note / `docs/music_lm_design.md` sections.
- Determinism in tests: pass an explicit `np.random.default_rng(seed)`.
- Keep the tokenizer behind its current interface (`encode`/`decode`/`token_type`) so the
  scheme can change without touching the model or the Phase-1 bridge.

## Datasets

- **Phase 0 pretraining:** start on **MAESTRO** (clean, Disklavier); scale with **ATEPP**,
  **GiantMIDI-Piano**, **Aria-MIDI** (~100k h, transcribed). Avoid Lakh as primary.
- **Phase 1 (thesis task):** **ASAP** — the only corpus with aligned score↔performance;
  MAESTRO supplies audio for the overlapping subset (Phase 3).
- **Phase 2:** **URMP** — downloaded and MD5-verified (Dryad doi:10.5061/dryad.ng3r749;
  the direct/API download is behind an Anubis proof-of-work gate, needs a browser),
  extracted at `../data/urmp/Dataset/` (44 pieces). NB arrangements of one composition
  share **identical track recordings**, so the dev/confirmation split MUST stay at the
  composition level (`phase2/splits.py`, frozen).
- **aria model:** frozen-feature **baseline / upper bound**, never the backbone.
- MAESTRO + ASAP loaders are wired (see Phases list). Always hold out eval pieces;
  transcribed corpora may overlap ASAP/MAESTRO (contamination). ASAP's `metadata.csv`
  carries a `maestro_midi_performance` cross-reference; `asap_clean_performances` drops any
  ASAP eval performance whose MAESTRO twin was in Phase-0 pretraining. MAESTRO's own split
  is composition-safe; `maestro_split(strict_dedup=True)` additionally drops 5 title-colliding
  eval pieces.

## Run

```bash
# one-time: create the conda environment (see environment.yml)
conda env create -f environment.yml
conda activate score-bundle
pip install -e ".[dev]"     # editable install (or just: export PYTHONPATH=src)

pytest                      # full suite (numpy-only paths must stay green)
python examples/phase0_pretrain_lm.py
python examples/phase1_imputation.py

# real Phase-0 pretraining on MAESTRO (single-GPU; tokenizes + trains MusicGPT)
python scripts/train_lm.py --maestro-root ../data/maestro-v3.0.0 \
    --d-model 256 --n-layer 4 --epochs 10 --cache-dir .cache/lm
```

The conda env's `train` extra installs torch + pretty_midi + tqdm, so the PyTorch LM
trains for real. Without torch, `examples/phase0_pretrain_lm.py` prints an install hint and
the LM tests no-op; the Phase-1 examples and the numpy core run regardless.

**Long-run rule (2026-08-19, after two silent 21h/31h Phase-2 evals):** long
evaluations run SHARDED with per-cell progress — Phase 2:
`bash scripts/run_phase2_eval.sh [N=8] [tonal]`; Phase 1:
`eval_graphgp.py --shard k/n` + report. Estimate wall time from one cell
before launching; a silent single-process eval projected past ~2 h is a bug,
not a mode (shard equivalence is bit-exact, pinned by
`tests/test_phase2_eval_shard.py`). The Phase-2 confirmation one-shot has a
staged, guarded runner: `scripts/run_phase2_confirmation.sh` — it refuses
without `PHASE2_SPLIT=confirmation` + `PHASE2_CONFIRMATION_I_AM_SURE=yes`
and was run ONCE on 2026-08-27 (Ray's explicit go); the pool is spent —
never run it again.

**Env gotchas (this machine):** activating `score-bundle` sets
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (a conda env var) so numpy/torch find the conda
`libstdc++` (GLIBCXX_3.4.29). The NVIDIA driver is CUDA 12.8, so torch must be the **cu128**
build (`pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`);
the default PyPI cu130 wheel disables CUDA. Datasets live in `../data/`.

## Design decisions worth respecting

- **aria = dataset + baseline, not foundation.** Use the Aria-MIDI *dataset* for
  pretraining and the aria *model* as a frozen-feature **upper-bound baseline**. We build
  our own small, note-aligned, score-conditioned LM as the object of study (reasons in
  `docs/music_lm_design.md` §6.1: representation mismatch, score-conditioning, eval
  contamination, confound control).
- **The contribution is structure + calibration**, not raw accuracy. Evaluations must
  report calibration (coverage, PIT, NLL), not just error, and must isolate the graph
  prior's marginal value via held-out imputation against the baselines.
- Guard against **train/eval contamination** when using transcribed corpora.

## When adding code

- Put new modules under `src/score_bundle/...`; add tests under `tests/test_*.py`.
- Prefer extending the existing interfaces (`GraphGaussianField`, `MidiTokenizer`,
  `GPTConfig`) over parallel implementations.
- Update `README.md`, `docs/`, and the notation table here if you introduce new symbols.
