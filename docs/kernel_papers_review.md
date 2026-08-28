# Review: the supervisor's two kernel pointers, three papers

2026-08-28. Papers in `related_works/`: `gp-kernel-pattern-discovery.pdf`
(Wilson & Adams, ICML 2013), `non-stationary-spectral-kernels.pdf`
(Remes, Heinonen & Kaski, NeurIPS 2017), `gp-music-audio-model.pdf`
(Alvarado & Stowell, arXiv 1606.01039, 2016). Prompted by the
supervisor's pointers "periodic kernel Gaussian process" and
"generalized spectral mixture", in the context of the vibrato/pitch-curve
model (draft eq:cents-curve/eq:vibrato) and the Phase-3 likelihood.

## 1. Wilson & Adams 2013 — the spectral mixture (SM) kernel

**What it is.** Bochner's theorem: every stationary kernel is the
Fourier transform of a spectral density. Instead of hand-picking a
kernel, model the spectral density itself as a Q-component Gaussian
mixture; the transform has closed form,

    k(τ) = Σ_q w_q exp(−2π² τ² v_q) cos(2π τ μ_q),

so each component is a damped cosine: μ_q the frequency, v_q how fast
coherence decays, w_q the weight. Gaussian mixtures are dense in
spectral densities, so the family approximates ANY stationary kernel;
inference stays exact GP regression; the marginal likelihood prunes
unused components (their Q=10 runs typically keep ~7).

**What they show.** Pattern discovery + long-range extrapolation (CO2,
airline, sinc, negative-covariance AR series) where SE/Matérn/rational
quadratic/periodic kernels interpolate but cannot extrapolate; the
learned spectral density is interpretable (each peak = a discovered
periodicity).

**Relevance to us.**
- The periodic kernel the supervisor named is the v_q → 0 sharp-peak
  limit of one SM component; a vibrato = one component at f_vib with a
  finite v (coherence decays — vibrato is not phase-locked forever),
  plus a near-zero-frequency component = the drift/trend. So "vibrato +
  drift" is a TWO-COMPONENT SM prior on the cents curve, learned by the
  same per-piece evidence machinery we already run.
- Their ARD-by-marginal-likelihood story is exactly our per-piece
  evidence discipline (kernels switch off when unneeded).
- Limitation for us: stationary — one fixed vibrato rate/coherence per
  note. That is precisely what the drift study and the Phase-3 study
  found insufficient. Hence paper 2.

## 2. Remes, Heinonen & Kaski 2017 — the generalized spectral mixture (GSM)

**What it is.** The non-stationary extension, via the generalised
Fourier transform (spectral SURFACE S(s,s') instead of a density). The
practical kernel makes the SM parameters input-dependent functions,
each itself a GP: log w_i(x), log ℓ_i(x), logit μ_i(x) ~ GP, giving

    k_GSM(x,x') = Σ_i w_i(x) w_i(x') k_Gibbs,i(x,x') cos(2π(μ_i(x)x − μ_i(x')x')),

a product of three PSD terms (input-dependent amplitude, Gibbs
lengthscale, input-dependent frequency). Reduces exactly to SM under
constant functions. Inference: MAP on the marginalized posterior with
whitened gradients; Kronecker structure for gridded inputs.

**What they show.** Their FIRST experiment is our problem in miniature:
a simulated oscillation whose frequency changes over time — the GSM
kernel recovers the time-varying frequency where SM/SE cannot. Then
texture extrapolation and climate fields.

**Relevance to us.**
- This is the principled version of everything the drift study measured:
  amplitude that grows (w_i(t)), rate that drifts (μ_i(t)), coherence
  that varies (ℓ_i(t)) — WITHIN a note, as smooth functions, not new
  per-note scalars.
- The estimator connection: our gated sine fit eq:vibrato is the
  degenerate limit (one component, constant functions, ℓ → ∞, hard
  onset gate). A GSM-prior regression on the cents curve would replace
  the NLLS estimator wholesale and output distributions over
  (c(t), γ(t), f(t)) instead of point scalars — "estimator v2".
- Cost/care: each hyperparameter is now a latent function → MAP over
  function values, far heavier than our closed-form fits; per-note
  curves are short (10²–10³ frames), which keeps it feasible but makes
  the GP-hyperprior lengthscales influential. Their whitening trick is
  the relevant implementation detail.

## 3. Alvarado & Stowell 2016 — GPs for music audio

**What it is.** The two ideas above, assembled for music, in the
WAVEFORM domain. A recording is f(t) = Σ_m φ_m(t) f_m(t): per-note
sigmoid change-windows φ_m (onset/offset parameters) gating independent
per-note GPs — kernel k_f = Σ_m φ_m k_m φ_m. Each per-note kernel is the
exponentiated-cosine family: k_EC = σ² exp[z cos(ωτ)] has a genuinely
HARMMONIC spectrum (fundamental ω plus integer partials, envelope set by
z — one parameter for spectral richness); k_ECQ multiplies in an SE term
so the amplitude envelope can vary within the note. Pitch estimation =
maximize the marginal likelihood over each note's ω; also missing-data
imputation on real polyphonic audio (8 kHz).

**Relevance to us.**
- Nearest neighbour to Phase 3 in the literature, with an instructive
  contrast: they put the harmonic structure IN THE KERNEL (waveform =
  one GP, stationary-per-note, harmonic covariance); we put it in the
  deterministic basis Φ(z) with marginalized amplitudes and keep the
  nonlinearity in explicit position variables z. Their route marginalizes
  everything but loses named per-note variables; ours keeps the named
  expressive channels (c, γ, f, τ...) that the thesis is about — worth
  saying to the supervisor as a deliberate design difference, not an
  omission.
- Their pitch estimation (marglik over ω per note) is procedurally our
  Phase-3 grid over c on the collapsed likelihood — independent
  validation of the inference pattern.
- Their change-windows are "manually specified" — they lack a score. We
  HAVE the score: our onsets/durations supply φ_m for free. The
  score-informed setting removes their main practical burden.
- Their k_EC is also the natural WAVEFORM-domain quasi-periodic prior if
  we ever want the harmonic content itself to be a GP rather than
  explicit amplitudes.

## Synthesis — the ladder, and where each rung fits our model

1. SM (stationary, learned spectrum) → the right PRIOR FAMILY for a
   note's cents curve: vibrato component + trend component.
2. GSM (input-dependent spectrum) → the right family once the drift
   study's finding is taken seriously: rate/extent/coherence varying
   within the note. Supersedes the parametric sine estimator in
   principle.
3. Alvarado & Stowell → the same ideas at waveform level with per-note
   gating; validates the Phase-3 inference pattern; our score supplies
   what they must hand-specify; our explicit-z design is the deliberate
   alternative to their all-in-the-kernel design.

**Where the kernels would plug in (two concrete slots):**
- Slot A — estimator v2 (Phase-2 style, per-note cents curve): GP
  regression with SM/GSM kernel on tracked frames; outputs distributions
  over slowly-varying (c, γ, f)(t). Any adoption is post-hoc to the
  spent confirmation and needs its own registered evaluation.
- Slot B — Phase-3 curve prior: the marginalized pitch-curve deviation
  prior already in the collapsed likelihood is a GP with a crude bump
  kernel; SM/GSM/EC are the principled replacements, still Gaussian, so
  the Woodbury machinery carries over.

**Honest boundaries to state.** These kernels model WITHIN-note
structure; across notes the slopes are white (drift study), so the graph
prior's role is untouched. GSM's flexibility costs identifiability on
short notes (a free frequency function can absorb drift and vice versa)
— the same confound our gated estimator manages with hard rules; priors
on the hyper-GPs would have to do that job. And the exp-cos family
assumes exact harmonicity — strings are mildly inharmonic, which our
explicit-Φ route also currently assumes.
