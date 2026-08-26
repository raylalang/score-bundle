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
- **Phase 2 — intonation/vibrato/loudness/timing/vibrato-delay: CONFIRMED (one-shot spent 2026-08-27).**
  `src/score_bundle/phase2/`: `intonation.py` (`extract_f0` = librosa pyin,
  import-guarded; `fit_vibrato_note` = the thesis NLLS estimator), `urmp.py` (loader,
  44/44), `splits.py` (**FROZEN** composition-level dev/confirmation split, unit-pinned;
  confirmation = 13 pieces, SPENT 2026-08-27), `targets.py` (f0 → per-note channels).
  Measured groundwork: tracker calibration vs URMP GT (2–5 cents/instrument, confidence
  predictive → as-given variances + lowest-quintile frame filter,
  `results/tracker_calibration_dev.md`); τ feasibility (onset-anchored warp, 76/78
  tracks, 79 ms, lag-1 +0.59, `scripts/eval_tau_feasibility.py`). Real-audio
  results (`scripts/eval_phase2_real.py`, `results/phase2_real_results.md`,
  6-channel, as-given default): graph value significant on intonation/vibrato
  recovery (−0.89*/−0.26*/−0.30*) and vibrato/timing calibration, coverage
  0.88–0.91 on all six; honest cells reported (ℓ recovery ns with NLL +0.042*
  against — starred adverse; τ recovery ns; δ_vib graph-neutral; brass
  intonation ns in the learned-scale family table); learned-scale variant
  stars all six recovery contrasts but is the worse-calibrated fallback.
  GT-validated octave-failure rule (|c|>150 → missing); learned noise scale
  collapses on real data with failure cells present, as-given healthy.
  Within-note drift study (2026-08-27, `scripts/eval_drift_dev.py`,
  `results/phase2_drift_dev.md`, thesis §3.9 ¶"What the per-note compression
  discards"): the committee's "sine model too simple" comment quantified —
  intonation drift is real music (two-thirds of notes significant in tracker
  AND GT independently, 97% sign agreement, ~10 cents/note median) and loudness
  moves even more (137% of the channel's across-note sd, but 65% is decay
  envelope); decisively, the slopes are GRAPH-WHITE (lag-1 +0.03/+0.06 vs
  +0.59 for τ) → no new channel, mismatch priced into residual-based cell
  variances (why calibration held), per-note resolution is the right level for
  the graph model; frame-level structure belongs to Phase 3. Registered
  estimator unchanged.
  Circle-of-fifths EXPLORATORY result (2026-08-18, `results/phase2_tonal_dev.md`):
  tonal metric beats plain on intonation both axes (−0.213*/−0.050*), re-imposes
  the replacement penalty on timing — first geometry-level positive; adoption =
  future preregistered confirmation.
  All DEV-labeled. **REGISTERED 2026-08-17** (tag `phase2-registration-2026-08-17`,
  commit = frozen claims): 6-channel bundle [c, log γ, log f, ℓ, τ, δ_vib];
  τ adopted (onset-anchored LOO warp, `phase2/warp.py`, noise row = OLS
  predictive variance); δ_vib IN by pre-stated criterion
  (`scripts/eval_delta_vib.py`, gated estimator `fit_vibrato_note_gated` =
  eq:vibrato exactly; 95-97% coverage, 18 ms GT agreement) but carries NO
  claim (graph-neutral); claims C1 intonation recovery / C2 vibrato
  calibration / C3 coverage / C4 timing calibration, as-given variant
  primary, one shot (`docs/phase2_prereg_design.md`). **CONFIRMATION SPENT
  2026-08-27 (Ray's explicit go; 40 unique tracks, 1h21m sharded;
  `results/phase2_confirmation_results.md` + verdict section, evidence
  archived `evidence/phase2_confirmation/`): C1 PASS (c dRMSE −0.877*,
  dev −0.891* reproduces), C2 PASS both channels (dNLL γ −2.990*,
  f −0.564* — the seed-sensitive extent star HELD), C3 PASS (cov
  0.88–0.91 all six), C4 secondary FAIL (τ dNLL −0.030 ns, CI incl. 0;
  plus starred adverse τ recovery +0.003*) → headline CONFIRMED by the
  registered rule; adverse dev ℓ cell did NOT replicate (+0.015 ns);
  δ_vib graph-neutral as registered; as-given stays better-calibrated
  vs quasi-truth. Thesis §3.9 carries the verdict (tab:phase2-conf).**
  Thesis §3.9 carries the measured state. 2026-08-13 audit: every quoted
  number re-verified against its log; reproduction tolerances recorded in
  `docs/posterior_decomposition_results.md`.
- **Phase 3 — waveform likelihood (stubs + helpers).** `src/score_bundle/phase3/`.
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
| `J_i`, `P_j`, `q_0.2`, `f_ref` | Phase-2 frame rule: note-i frame set, tracker per-frame confidence (capital — `p_i` is pitch), lowest-quintile threshold, tuning reference 440 Hz = A4 (MIDI 69) |
| `φ̂_i, ψ̂_i` | Phase-2 local tempo-line slope/intercept at note i (eq:localwarp) |
| `z, a, Φ(z), x, ε, A_i(t)` | Phase-3 positions, amplitudes, synth, audio, noise, amp envelope |

Do **not** reuse `S` for a covariance, `σ` for a prior scale, `α` for the additive weight,
or `a_i(t)` for the amplitude envelope. (These were deliberately disambiguated.)

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
