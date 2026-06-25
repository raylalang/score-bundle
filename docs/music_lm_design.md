# Phase 0 — A from-scratch symbolic music language model

This document specifies the Phase-0 model the professor asked for: a symbolic music
language model (LM) built from the ground up — our own tokenizer, architecture, and
training loop — rather than a downloaded black box. It states what we build, why, and
exactly how it connects to the Phase-1 score-graph prior.

## 1. Why build it, and why from scratch

The LM plays three roles we agreed on:

1. **Generative backbone / learned prior.** An autoregressive model over performance
   tokens learns the distribution of expressive piano performance — including the
   timing and dynamics the thesis cares about — capturing long-range musical structure
   a 2-hop graph cannot. It is the forward (synthesis) direction's learned core.
2. **Representation learner.** Its hidden states give a per-note embedding `h_i` that
   replaces hand-built score features and conditions the Phase-1 prior.
3. **Foundation / skill-building.** Owning the whole stack lets us (a) attach the
   Bayesian/uncertainty layer to the internals, (b) make interpretability and
   calibration claims that a black box would forbid, and (c) tokenize specifically for
   the score(base)/performance(fiber) split.

"From scratch" is therefore not dogma: the thesis is about *structure + calibrated
uncertainty*, and both require access to model internals and a representation we control.

## 2. Scope and modality

- **Symbolic**, not audio: tokenized MIDI. (Audio LMs from scratch are out of scope.)
- **Piano first**, matching Phase 1 (ASAP/MAESTRO/ATEPP), with optional large-corpus
  pretraining (Lakh MIDI, GiantMIDI-Piano) for the representation.
- **Small scale**, single-GPU. The deliverable is a controllable, inspectable model, not
  a state-of-the-art generator.

## 3. Representation / tokenization (the key decision)

We use a **flattened compound, note-structured** scheme: each note is emitted as a fixed
4-token group, in onset order:

```
[ TIME_SHIFT(Δ) , PITCH(p) , DURATION(d) , VELOCITY(v) ]
```

with sentinel tokens `PAD`, `BOS`, `EOS`. Quantization:

- `TIME_SHIFT`: inter-onset interval since the previous note, quantized to a grid
  (default: beats / 24, clipped to a max).
- `PITCH`: MIDI number (piano range 21–108 by default).
- `DURATION`: performed duration, same time grid.
- `VELOCITY`: MIDI velocity bucketed into `n_vel_bins` (default 32).

Rationale: the group is **isomorphic to a note's performance variables** `(τ, log r, v)`
plus pitch, so (a) detokenization is trivial, and (b) the hidden state at each group's
last token (`VELOCITY`) is a natural per-note embedding aligned with the graph nodes.
This is closest in spirit to REMI (Huang & Yang 2020) and the MIDI-like performance
encoding of Oore et al. (2018), simplified for a fixed per-note stride.

Alternatives considered: pure MIDI-like event streams (variable stride, harder to map to
notes); Compound Word / Octuple (multi-stream embeddings, more plumbing). We start simple
and can swap the tokenizer behind the same interface.

## 4. Architecture (decoder-only Transformer, from scratch)

A standard GPT-style decoder, implemented ourselves (no `transformers` library):

- token embedding + learned positional embedding
- `n_layer` blocks: pre-LayerNorm → causal multi-head self-attention → residual;
  pre-LayerNorm → MLP (GELU, 4× width) → residual
- final LayerNorm → linear LM head (weight-tied to the token embedding)

Reference defaults (tiny): `d_model=256`, `n_layer=4`, `n_head=4`, `block_size=512`,
dropout 0.1. This is deliberately nanoGPT-scale (Karpathy) so it trains on one GPU and
stays legible.

The model lives in `lm/model_torch.py`, implemented from the ground up in PyTorch:
hand-written causal self-attention (`nn.Linear` q/k/v + masked softmax — no
`nn.MultiheadAttention`, no `transformers`), a GELU MLP, pre-LayerNorm blocks, and a
weight-tied head. `forward` / `embed` / `generate` and a `train_lm` loop ship with it. The
tokenizer and batching (`lm/tokenizer.py`, `lm/data.py`) stay framework-agnostic NumPy, so
the representation is decoupled from the training framework.

## 5. Pretraining objective

Standard autoregressive next-token prediction (cross-entropy), teacher-forced over the
flattened token stream, with causal masking and the usual shifted targets. Optimizer:
AdamW, cosine schedule, gradient clipping. Validation by held-out perplexity and by
sampling plausible continuations.

We may add a light **field-structured factorization** later (predict the four token types
with type-specific heads), but plain next-token prediction is the baseline.

## 6. Integration with the Phase-1 graph prior (the payoff)

This is where Phase 0 earns its place. After pretraining, for a given score/performance we
extract per-note embeddings `h_i` (hidden state at each note's last token). They feed the
structured model two ways:

1. **Learned prior mean.** A small linear/MLP head maps `h_i → μ_i`, a prediction of the
   expressive variables `y_i = [τ_i, log r_i, v_i]`. The graph prior then models the
   *residual* with calibrated structured uncertainty:

   ```
   y = μ_LM(h) + ξ,     ξ ~ N(0, Q_G^{-1})
   ```

   so the posterior is `N(m, Σ_y)` exactly as in Phase 1, but centered on the LM mean
   instead of zero. The LM supplies the expressive signal; the GMRF supplies the
   correlated, calibrated residual uncertainty.
2. **Learned features for the graph.** `h_i` can also define edge weights / node features,
   replacing or augmenting the hand-built score-distance graph.

This makes the contribution sharper: the experiment becomes *does a score-graph residual
prior on top of a learned LM mean improve recovery and calibration over the LM alone (and
over the hand-built baselines)?* — a stronger, more modern comparison than zero-mean
priors vs ridge features.

## 6.1 Why not just use a large pretrained model (aria)?

Nothing technically stops us from freezing EleutherAI's **aria** (a self-supervised
representation model trained on ~100k h of transcribed piano) and using its embeddings as
`h_i`. It would likely give *better raw features* than a small from-scratch model. We still
build our own, and use aria as a **baseline / upper bound**, for four concrete reasons:

1. **Representation mismatch.** Our prior attaches a GMRF over score *notes* and reads a
   per-note `μ_i` off the model; that needs a note-aligned representation. aria is an event
   stream built for generation, so per-note embeddings require reverse-engineering note
   boundaries and pooling — an uncontrolled transform between model and prior.
2. **Not score-informed.** aria models `p(performance)`; our task conditions on the known
   score and infers deviations (the base/fiber split). aria has no score input.
3. **Evaluation contamination.** Aria-MIDI is ~1M transcribed YouTube recordings that almost
   certainly overlap our eval sets (ASAP/MAESTRO). Numbers from aria's weights cannot be
   trusted as *recovery* vs memorization; a controlled split avoids this.
4. **Confound control / provenance.** A 100k-h black box could carry all the signal, masking
   the structured prior's marginal value. A small owned backbone isolates it and lets us
   state exactly what was trained on.

**Honest caveat:** from-scratch does not by itself make a transformer interpretable, and aria
will likely win on raw feature quality. So we *report* both — `aria-features + graph prior`
and `aria alone` — alongside our model. If the structured prior improves **calibration** even
on aria features, that is a strong result; if our small model + prior approaches aria with a
fraction of the data and full provenance, that is also a result. Use the **Aria-MIDI dataset**
freely; treat the **aria model** as a reference point, not the foundation.

## 7. Data

- **Pretraining (representation):** start on MAESTRO (clean, Disklavier-captured); scale with
  ATEPP, GiantMIDI-Piano, and **Aria-MIDI** (~1.18M files / ~100k h transcribed solo piano,
  purpose-built for pretraining). Avoid Lakh as primary (multi-instrument, off-domain).
  Caveat: everything but MAESTRO is AMT-transcribed (noisier).
- **Downstream (Phase 1):** ASAP aligned score↔performance for the expressive-variable
  targets (the only corpus with aligned scores; MAESTRO supplies audio for the overlap).

Tokenizer and batching are framework-agnostic (NumPy), so the same token streams feed
either model implementation.

## 8. Deliverables and milestones

- `lm/tokenizer.py`, `lm/data.py` — tokenizer + batching (runnable, tested).
- `lm/model_torch.py` — from-scratch PyTorch GPT: hand-written attention, forward / embed /
  generate + training loop (this is what we pretrain).
- `lm/features.py` — per-note embeddings → prior mean (connects to Phase 1).
- `examples/phase0_pretrain_lm.py`, `examples/phase0_lm_features_to_prior.py`.

Rough schedule (folds into June–early July): week A tokenizer + data; week B NumPy model +
tests; week C PyTorch model + a small pretraining run; week D embeddings → prior-mean
integration and a first comparison.

## 9. Risks / gates

- **Compute/data limits.** A tiny from-scratch LM may underfit. *Gate:* keep it small and
  verify it adds signal — the LM prior mean (or features) must beat the hand-built
  baseline before the rest of the pipeline depends on it.
- **Tokenization lock-in.** Keep the tokenizer behind a stable interface so the scheme can
  change without touching the model or the Phase-1 integration.
- **Pretraining/eval contamination.** Large transcribed corpora (Aria-MIDI, GiantMIDI) may
  overlap the Phase-1 eval pieces. *Gate:* train on held-out splits; when using aria as a
  baseline, treat its numbers as an upper bound, not clean recovery.
- **Scope.** The LM is a backbone/representation, not a route to full AMT in this thesis.

## 10. References

- Huang et al. (2018), *Music Transformer*, arXiv:1809.04281 (ICLR 2019).
- Huang & Yang (2020), *Pop Music Transformer (REMI)*, arXiv:2002.00212 (ACM MM 2020).
- Oore, Simon, Dieleman, Eck & Simonyan (2018), *This Time with Feeling: Learning
  Expressive Musical Performance*, arXiv:1808.03715.
- Karpathy, *nanoGPT* — https://github.com/karpathy/nanoGPT (canonical from-scratch GPT).
- Raffel (2016), *Lakh MIDI Dataset*. Kong et al. (2022), *GiantMIDI-Piano*, TISMIR.
- *Aria-MIDI: A Dataset of Piano MIDI Files for Symbolic Music Modeling* (2025),
  arXiv:2504.15071 — https://github.com/loubbrad/aria-midi.
- EleutherAI, *aria* — Scaling Self-Supervised Representation Learning for Symbolic Piano
  Performance (ISMIR 2025) — https://github.com/EleutherAI/aria.
