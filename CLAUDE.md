# CLAUDE.md — score-bundle

Guidance for working in this repo. Read `README.md` and `docs/music_lm_design.md` first.

## What this is

A research codebase for **Bayesian, score-informed performance transcription**: given a
known symbolic score and a performance, infer per-note expressive variables (timing,
articulation, dynamics; later intonation/vibrato) with calibrated uncertainty, using a
**score graph** as a structured prior. The model is one generative process run two ways —
forward = synthesis, inverse = transcription. See `docs/architecture.svg`.

The companion thesis plan lives in Notion ("Score-Bundle Models"); keep code and that
note conceptually in sync.

## Phases (and what's implemented)

- **Phase 0 — music LM (implemented).** From-scratch **PyTorch** decoder-only Transformer
  over note-structured MIDI tokens (`src/score_bundle/lm/`, hand-written causal attention).
  Provides a learned prior mean `μ_LM` and per-note embeddings that feed Phase 1. Tokenizer
  and batching are framework-agnostic NumPy; the model is PyTorch.
- **Phase 1 — core, piano (implemented + evaluated).** Score graph → Gaussian graph prior →
  closed-form posterior with per-note uncertainty; baselines; calibration metrics. Held-out
  ASAP eval (`scripts/eval_asap_calibration.py`, `src/score_bundle/imputation_eval.py`) shows
  `LM mean + graph residual` best on RMSE *and* calibration — see
  `docs/phase1_calibration_results.md`. NB the graph posterior needs a **predictive-variance
  floor** (held-out `y=f+ε` has variance `diag(Σ_y)+noise_var`) or NLL/coverage blow up.
  Aria frozen-feature upper-bound baseline is an import-guarded stub (`lm/aria_baseline.py`).
- **Phase 2 — intonation/vibrato (stubs + helpers).** `src/score_bundle/phase2/`.
- **Phase 3 — waveform likelihood (stubs + helpers).** `src/score_bundle/phase3/`.
- Real dataset loaders: **MAESTRO** (Phase-0 LM) and **ASAP** (Phase-1 aligned task) are
  **implemented** — `lm/data.py` (`load_maestro_meta`, `maestro_note_events`,
  `iter_maestro_note_streams`, `maestro_split`) and `features.py` (`load_asap_meta`,
  `load_asap`, `asap_performance_variables`, `asap_clean_performances`). Aria-MIDI / ATEPP /
  GiantMIDI loaders remain open. Datasets live under `../data/`
  (`/home/ray/Research/data/{maestro-v3.0.0,asap-dataset}`); pass the root explicitly.

## Canonical notation (keep consistent everywhere — code, docs, Notion)

| Symbol | Meaning |
|--------|---------|
| `S = {s_i}`, `s_i=(p_i,b_i,d_i)` | score support (pitch, beat onset, beat duration) |
| `y_i = [τ_i, log r_i, v_i]` | Phase-1 per-note variables (onset residual, articulation, velocity) |
| `Q_G` | graph prior precision; additive `λI + ηL_G`, or Matérn `σ_g⁻²(κ²I+L_G)^α` |
| `λ, η` | additive ridge term, Laplacian weight |
| `σ_g, κ, α` | Matérn scale, inverse-length, exponent |
| `Σ_e` | observation-noise covariance; `Σ_y` posterior covariance; `m` posterior mean |
| `σ` | **posterior standard deviation only** (not a prior scale) |
| `μ_LM`, `h_i` | LM-predicted prior mean, LM per-note embedding |
| `z, a, Φ(z), x, ε, A_i(t)` | Phase-3 positions, amplitudes, synth, audio, noise, amp envelope |

Do **not** reuse `S` for a covariance, `σ` for a prior scale, `α` for the additive weight,
or `a_i(t)` for the amplitude envelope. (These were deliberately disambiguated.)

## Conventions

- **NumPy-first statistical core.** The Phase-1 package (graph / prior / model / metrics /
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
