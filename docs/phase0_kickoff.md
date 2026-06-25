# Phase 0 kickoff prompt (paste into Claude Code)

Run Claude Code from the repo root and paste the prompt below.

---

You are working in the `score-bundle` repo. Before coding, read `CLAUDE.md`, `README.md`,
and `docs/music_lm_design.md` (especially §3 tokenization, §4 architecture, §6 the Phase-1
integration, and §6.1 "why not aria"). Respect the canonical notation table and the
**NumPy-first** rule in `CLAUDE.md` (the core package and `pytest` must work with numpy
only; torch / pretty_midi / aria are optional and import-guarded).

**Step 0 — environment (conda).** Create and activate the environment, install editable,
and confirm the suite is green. Show me the test summary before continuing.

```bash
conda env create -f environment.yml
conda activate score-bundle
pip install -e ".[dev]"
pytest -q
```

If the default torch wheel doesn't match my GPU, install the right build from
https://pytorch.org/get-started/locally/ and re-run.

**Goal:** get Phase 0 (the from-scratch music LM) training on real data and feeding the
Phase-1 prior. Work in small, tested increments.

1. **Real MIDI ingestion.** Add a MAESTRO loader in `src/score_bundle/lm/data.py` (use
   `pretty_midi`) that yields `NoteEvent` streams compatible with `MidiTokenizer`. Add a
   deterministic train/val/test split **by piece/performer** (no leakage). Unit-test that a
   real file round-trips through `encode`/`decode` within quantization tolerance.

2. **Pretrain the LM.** Add `scripts/train_lm.py` (or extend
   `examples/phase0_pretrain_lm.py`): tokenize MAESTRO, train `lm.model_torch.MusicGPT` with
   next-token cross-entropy, log train/val loss + perplexity (tqdm), checkpoint, and sample a
   continuation each epoch. Start tiny (`d_model=256, n_layer=4`) and confirm validation
   perplexity drops. Single-GPU.

3. **Embeddings → prior mean.** With `lm.features`, extract per-note embeddings on an ASAP
   subset, fit the `h_i → μ_LM` head for `y = [τ, log r, v]`, and plug `μ_LM` into
   `GraphGaussianField(mean=μ_LM)`. Report held-out **imputation RMSE and calibration
   (coverage / PIT / NLL)** for: zero-mean prior, ridge-feature mean, and LM mean — each
   with and without the graph residual.

4. **aria baseline (upper bound).** Add an optional path that loads EleutherAI `aria` as a
   frozen feature extractor and runs the same comparison (`aria alone`,
   `aria-features + graph prior`). Guard the import; treat its numbers as an upper bound and
   note possible train/eval contamination. If aria isn't easily loadable, stub the interface
   with a TODO.

**Constraints**
- NumPy-first core: `pytest` must pass with numpy only.
- Don't break the `MidiTokenizer` / `GraphGaussianField` / `GPTConfig` interfaces.
- Hold out eval pieces; never evaluate on anything the LM trained on.
- Keep notation consistent with `CLAUDE.md` (`μ_LM`, `Σ_y`, `η`, `σ_g`, …).
- Add tests for every new module; update `README.md` / `CLAUDE.md` if you add symbols or
  scripts.

Start with Step 0, then propose a short plan for steps 1–4 before writing code.
