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

- **Phase 0 — music LM (implemented).** From-scratch decoder-only Transformer over
  note-structured MIDI tokens (`src/score_bundle/lm/`). NumPy forward+sampling (no deps,
  tested) + a trainable PyTorch twin. Provides a learned prior mean `μ_LM` and per-note
  embeddings that feed Phase 1.
- **Phase 1 — core, piano (implemented).** Score graph → Gaussian graph prior → closed-form
  posterior with per-note uncertainty; baselines; calibration metrics.
- **Phase 2 — intonation/vibrato (stubs + helpers).** `src/score_bundle/phase2/`.
- **Phase 3 — waveform likelihood (stubs + helpers).** `src/score_bundle/phase3/`.
- Real dataset loaders (ASAP/MAESTRO/Aria-MIDI) are **stubs** in `features.py` /
  `lm/data.py` — the main open work.

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

- **NumPy-first core.** The package must import and the tests must pass with **numpy only**.
  `scipy`, `scikit-learn`, and `torch` are *optional* and must be import-guarded (see
  `lm/model_torch.py`, `baselines.gbm_impute`). Never add a hard dependency to the core.
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
- Loaders are stubs — wiring them is the first real task. Always hold out eval pieces;
  transcribed corpora may overlap ASAP/MAESTRO (contamination).

## Run

```bash
# one-time: create the conda environment (see environment.yml)
conda env create -f environment.yml
conda activate score-bundle
pip install -e ".[dev]"     # editable install (or just: export PYTHONPATH=src)

pytest                      # full suite (numpy-only paths must stay green)
python examples/phase0_pretrain_lm.py
python examples/phase1_imputation.py
```

The conda env's `train` extra installs torch + pretty_midi + tqdm, so the PyTorch LM
trains for real. Without torch, `examples/phase0_pretrain_lm.py` falls back to a NumPy
forward+sample demo.

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
