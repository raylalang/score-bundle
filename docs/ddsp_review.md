# Differentiable synthesis and our Phase 3 (reading + fit memo)

2026-10-09, after the zemi. The supervisor's pointer: look into
differentiable audio generators. The anchor work is **DDSP:
Differentiable Digital Signal Processing** (Engel, Hantrakul, Gu,
Roberts, ICLR 2020): a harmonic-plus-noise synthesizer implemented in
an autodiff framework, so gradients flow THROUGH synthesis; models are
trained with multi-scale spectral (magnitude-spectrogram) losses, and
in the original paper a neural encoder predicts f0, loudness, and the
harmonic amplitude distribution. Companion readings: **MIDI-DDSP** (Wu
et al., 2022) — hierarchical control of musical performance via DDSP,
the closest cousin to our problem — and the **Hayes et al. review of
DDSP for music and speech synthesis** (2023).

## The honest fit

Our Phase 3 already IS a differentiable-synthesis design in everything
but implementation: `phase3/synth.py` is an additive harmonic
synthesizer (cumulative-phase oscillator bank — the same construction
as DDSP's harmonic branch), and the module docstring has said from the
start that inference over the nonlinear z belongs in an autodiff
framework. What we have that DDSP does not:

- an **exact amplitude-marginalized Gaussian likelihood** (DDSP point-
  estimates amplitudes with an encoder and scores with a heuristic
  spectral loss; we integrate the timbre out in closed form and get a
  calibrated likelihood);
- a **GP prior over the pitch curve** (DDSP's f0 comes from a pitch
  tracker — CREPE — at training time; ours is the latent being
  inferred, tracker-free);
- the across-note graph GP on top.

What DDSP has that we need:

- **gradients through the synthesizer.** Our current MAP search is
  Nelder-Mead over up to ~33 knot values — the scaling bottleneck and
  the reason note boundaries are still out of reach. The cumulative-
  phase synth and the collapsed likelihood are both smooth in z, so a
  torch port gives exact gradients and turns curve inference into
  seconds of Adam, scalable to joint curves + boundaries + whole
  phrases;
- the **multi-scale spectral loss** as a robustness option worth
  knowing about (phase-blind, so more forgiving of model mismatch;
  our exact likelihood is phase-aware — sharper when the model is
  right, more brittle when it is wrong). A measured comparison is a
  natural later study;
- an engineering pattern for whole-track synthesis (per-note windows,
  cf. the Alvarado-Stowell construction already in our vocabulary).

## What we do NOT take

No neural encoder (our inference is the GP posterior, not an amortized
network), no spectral loss as the primary objective (we have an exact
likelihood; the spectral loss becomes a baseline/robustness study), no
learned reverb/noise modules yet (URMP stems are dry and close-miked;
the Gaussian noise term carries the residual for now).

## First implementation step (started today)

`scripts/proto_phase3_torch.py`: a torch port of the chunked harmonic
design and the amplitude-collapsed log likelihood, equality-pinned
against the numpy path, with gradient-based MAP over (c, knots) under
the same two-component GP prior as the prototype — same three notes,
wall-time and accuracy compared against the Nelder-Mead version. GPU:
A6000s only (the A100s are occupied; checked before use, per the
standing rule).

Next steps if the prototype holds: fit the prior hyperparameters by
evidence (differentiable too), batch evaluation on the dev tracks, then
joint note boundaries — the open problem the gradients exist to unlock.
