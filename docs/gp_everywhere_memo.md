# Memo: "GPs everywhere" — the simplification the papers suggest

2026-09-03. Design memo ONLY — nothing here is a work order. Written
because the related_works papers suggest a reframing that could make the
thesis SIMPLER, and Ray asked that such ideas enter the plan.

## The idea in one paragraph

The thesis currently uses three different modelling languages: bespoke
NLLS estimators within notes (sine fits, chunk RMS, warp regressions), a
graph GP across notes, and a deterministic-basis likelihood for audio.
The papers point at one language for all three levels: **a GP at every
level.** Within a note: the pitch curve gets a quasi-periodic /
spectral-mixture prior (vibrato = one component, drift = one component)
— replacing the sine fit and its hand identifiability rules with a
posterior. Across notes: the graph GP, unchanged — this level is
confirmed and already GP-native. At the waveform: the collapsed
likelihood already is a GP marginal; the change-window construction of
Alvarado & Stowell shows how per-note windows (our score!) assemble
whole-track models.

## What it would simplify

- One inference story (posterior + evidence) at every level — easier to
  explain, easier to defend, matches the thesis's "one generative
  process" framing.
- The estimator's hand rules (min-cycles, gates, octave thresholds)
  become priors and posteriors — uncertainty flows automatically into
  the bundle's noise rows instead of via delta-method formulas.
- The Phase-3 deviation prior stops being a bump-basis hack: it becomes
  the same within-note kernel, used at waveform level.

## What it would cost (honestly)

- The confirmed Phase-2 pipeline is frozen; a GP estimator is
  "estimator v2" — new development, new registered evaluation, on a
  corpus question that is itself open (the pool is spent).
- Within-note GP hyperparameters on 30–150-frame notes are weakly
  identified (rate-vs-drift confound); the hand rules we would delete
  are currently doing that work. Priors would have to replace them, and
  tuning priors is its own project.
- GSM-style input-dependence costs the exact-conjugate, numpy-only
  simplicity the thesis is built on; even plain SM adds a nonconvex
  hyperparameter search per note.
- Measured caution: our Phase-3 study showed richer within-note models
  improve accuracy but do NOT fix cross-estimand calibration — the
  binding constraints there are the estimand bridge and the τ noise
  row, which no kernel choice touches.

## Minimal honest test (if ever pursued)

One development study, no new claims: SM-prior GP regression on cents
curves for the ~5,000 identifiable dev notes; compare against the NLLS
estimator on (i) agreement with quasi-truth, (ii) calibration of its
own posterior, (iii) behavior on the estimator-missing notes (where the
GP degrades gracefully instead of refusing). If it does not beat the
sine fit against quasi-truth, the idea dies cheaply.

## Recommendation

Discuss with the professor before building anything. The right sequence
is the plan's: comprehension first, thesis legibility second. This memo
exists so the option is on paper, not so it gets built this week.
