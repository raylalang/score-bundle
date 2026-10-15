# Meeting prep (2026-10-14) — research direction and scope

> Purpose of this meeting: resolve direction and scope, not report
> progress. The deck (`docs/slides/deck.pdf`, 44 frames) is the visual
> spine; this brief is the discussion skeleton. Standing language rules:
> say "the GP estimator" (never a coined name), "correlated timing noise
> across notes" (never bare "AR(1)"). Before the meeting: **sync
> Overleaf** (draft.tex changed substantially: restructure, Chapter 8,
> vocabulary, and today the introduction/RQ reframe).

## A. Research objective (the one-breath version)

An **interpretable, score-aligned representation of expressive
performance, inferable from a recording, with explicit uncertainty on
every recovered parameter.**

The score says which notes are intended; the audio contains how they
were played. We recover the *how* — timing, articulation, dynamics,
intonation, vibrato, and within-note trajectories — as named musical
quantities aligned to the written notes, not as disconnected acoustic
features. Recovering the notes is transcription; recovering *how the
notes were played* is this thesis.

Distinction to keep explicit in the room: the **scientific objective**
is the representation; the **methodological contribution** is the
probabilistic inverse-inference framework that recovers it. They are
complementary. Honesty guards: the representation is interpretable *by
construction* (named parameters), not semantically validated; the
framework is validated stage by stage, and its full integration at the
waveform is not yet complete (the first span is measured and is a
negative for note-level structure at chain scale, §G; the rest is
open).

## B. Why it matters

Possible uses (possible, not validated): performance analysis and
comparison across performers; pedagogy (showing a student *what* differs
from a reference, with confidence); performance editing and expressive
resynthesis; structured supervision for audio models; computational
musicology at corpus scale. The uncertainty is what makes any of these
actionable — an expression estimate without an error bar cannot support
a judgment.

## C. Proposed methodology: Bayesian score-conditioned inverse synthesis

    p(z | x, S)  ∝  p(x | z, S) · p(z | S)

- `S` the score, `x` the observation of the performance, `z` the latent
  expressive variables.
- `p(z | S)`: the score-structured prior — one multi-output Gaussian
  process on the score graph (notes related in the score are related in
  performance; channels coupled; score features and music-model
  embeddings as kernels; all hyperparameters by per-piece evidence).
- `p(x | z, S)`: the observation model, made progressively rawer across
  the thesis — aligned MIDI → tracker-derived per-note estimates with
  variances → the waveform itself through a differentiable harmonic
  synthesizer with amplitudes (timbre) marginalized exactly.

The two ingredients are complementary: the score contributes structure
(which notes should co-vary, where expression has room), the waveform
contributes evidence; the posterior reports what the combination cannot
decide.

## D. Research progression (why each stage was necessary)

1. **Piano (stage 1 — isolate the prior).** Aligned MIDI gives the
   variables nearly noise-free, so the pure question "does score
   structure provide informative priors over expression?" is answered
   without any audio complication. Masked-note recovery vs independent /
   temporal / no-graph baselines, both accuracy and calibration.
2. **Strings and winds (stage 2 — measurement uncertainty).** The same
   prior, unchanged, meets noisy, incomplete, per-note estimates from
   tracked audio (six channels incl. intonation and vibrato, each with
   its own variance; missing cells where estimation is impossible).
   Tests the prior under realistic observation noise.
3. **Waveform (stage 3 — the endpoint).** Replace the extraction chain
   with an explicit differentiable harmonic likelihood: candidate pitch
   trajectories and note boundaries scored directly against the audio,
   timbre marginalized analytically. Development studies only.
4. **The intended endpoint (first span built 2026-10-14, measured
   honestly 2026-10-15):** the score-graph prior joined with the
   waveform likelihood — all notes of a performance inferred jointly
   from score + audio. Status: the graph couples the per-note
   intonation constants *inside* the waveform inference on short
   chains, where its measured value is the shared tuning offset and
   nothing beyond it (the prior-value experiment, §G); the curve-level
   and boundary coupling, longer spans, the remaining channels, and
   whole-performance scale are still open.

## E. Current results (verified against the repository)

**Confirmed (preregistered, one-shot, untouched pools):**
- Piano (ASAP, 20 pieces): RMSE 0.376 vs 0.393 for the two-stage
  baseline (paired, significant); coverage 0.925 at nominal 0.90; the
  graph's calibration contribution independently significant.
- Strings and winds (URMP, 13 pieces): intonation recovery −0.877 cents
  (significant); vibrato calibration significant on both channels
  (−2.990 / −0.564 log score); coverage 0.88–0.91 on all six channels.
- **Failed claim, reported verbatim:** the secondary timing-calibration
  claim did not confirm (−0.030, ns; plus a small adverse significant
  timing-recovery contrast).

**Development / exploratory (no claims):**
- Curve-from-waveform, 376 notes, no tracker: median 3.16 cents vs the
  tracker's own frames at 4.09 against the corpus annotation; decisively
  better on winds (2.26 vs 5.15), worse on strings (4.90 vs 3.09);
  heavier failure tail.
- The inferred curve as a bundle channel: held-out intonation improves
  on 14/14 pairs (−1.36 cents, significant); the empirical calibration
  floor halves (3.1 → 1.8 cents).
- Joint note boundaries on 3–5-note chains: from score-only tempo-line
  starts (no annotation access), median boundary error halves, ~40 →
  ~20 ms; the transition-region hypothesis was *rejected* by the data
  (sharp edges win); repeated-pitch boundaries are unidentifiable to a
  harmonic likelihood (stratified out).
- Correlated timing noise across notes (the failed claim's follow-up):
  improves timing calibration significantly on development data across
  three independent runs; awaiting the registration-pool decision.
- Tonal (circle-of-fifths) metric helps intonation (development only).

**Negative results and limitations (kept, not hidden):**
- Waveform parameter bands are internally calibrated (model-true
  synthetics cover at 0.90) but cover the annotation frames at only
  0.32–0.36: the acoustic-vs-musical estimand gap plus model mismatch
  (measured residual correlation 0.944 at the harmonics). Handled by a
  measured per-track discrepancy floor wherever the posterior is
  consumed — not by trusting the thin band.
- Joint chains currently cost curve accuracy vs single-note fits
  (guarded RMSE 4.9–5.2 vs 3.16 cents).
- ~18 ms systematic boundary offset: the likelihood finds the acoustic
  transition, the annotation marks the musical onset.
- Music-theory features: honest null; the embeddings encode rhythm,
  voicing, register — not tonality. A learned many-component spectrum
  adds nothing over the hand-structured two-component kernel.
- Per-note compression measurably discards within-note drift/decay
  (the RQ1 "what it fails to preserve" evidence).

## F. Open technical problems (the discussion core)

1. **Graph prior × waveform likelihood — the bridge, span by span.**
   The first span (per-note intonation constants, short chains) is
   built and measured, and the measurement is a negative for note-level
   structure (§G): at chain scale the prior transfers the tuning offset
   and nothing more. Design question now: which variable or span is the
   first where note-level structure could carry information — boundary
   offsets (proposal written), longer spans, the curve level, the other
   channels — and how to make the common-mode control part of every
   design.
2. **Identifiability from audio.** Measured so far: pitch trajectories
   strongly identifiable (phase accumulation); boundaries identifiable
   except at repeated pitches; loudness/articulation live in amplitudes
   that are currently marginalized, not inferred. Which variables
   *should* we claim from audio?
3. **Model mismatch and the estimand gap.** The harmonic model's
   constant vs the annotation chain's functional differ by 2–4 cents;
   residuals are far from white. Current treatment: empirically
   calibrated floors. Open: a principled bridge (transient columns at
   joints are the pre-named next lever for boundaries).
4. **Value of score structure given strong acoustic evidence.** The
   waveform is so informative about pitch that the prior may matter
   mainly where evidence is weak (joints, low SNR, repeated pitches,
   masking). This is exactly what the proposed next experiment measures.
5. **Reliable uncertainty on real performances.** Internal calibration
   is verified; external calibration needs either the floor mechanism
   (validated at development level) or a likelihood that models the
   correlated residual.

## G. The next experiment — RUN, with its follow-ups (2026-10-14/15, development)

**The three-prior comparison exists and is measured three times on
the same 47 chains** (independent / temporal-chain / score-graph priors
over per-note intonation constants, matched marginal variance,
identical waveform likelihood and optimizer, boundaries fixed; ambiguity
= one note removed per chain, clean audio as the control; four
development tracks). Hypotheses and a falsification rule were committed
before every run.

- **The first run (10-14) contained a protocol error, found on 10-15
  and recorded as an erratum.** It masked the hidden note's audio but
  left its soft-window design columns in the likelihood; the window
  tails beyond the mask guard leaked a few samples' worth of evidence,
  which with marginalized amplitudes dominated the prior. Every arm's
  masked estimate was uncorrelated with the truth and worse than
  predicting zero. Its "structure wins" headline is withdrawn.
- **Corrected protocol, no tuning scalar** (`results/prior_value_v2_dev.md`):
  a cross-note prior halves the hidden note's error, 14.4 → 6.8–7.6
  cents (paired −5.8 / −6.2, significant, better on 79%); score graph
  and temporal chain tie; the recurring-pitch stratum trends the
  predicted way but is not significant (n = 25).
- **Corrected protocol, free per-chain tuning scalar in every arm**
  (`results/prior_value_v2_tuning_dev.md`): the independent arm now
  inherits the chain mean and reaches 6.4 cents; **all three arms tie**
  (graph vs independent −0.02 [−0.33, +0.32]; recurring stratum +0.01).
  The falsification rule fires: **on 3–5-note chains, score structure
  carries no measurable note-level intonation information beyond the
  shared tuning offset.**
- Clean audio: exact ties in every run. Calibration honest: structured
  conditional bands 0.89–0.94; the zero-mean independent band 0.77
  (miscentred), repaired by the scalar (0.96).

**What this says for the discussion.** The methodological hypothesis has
its first waveform-level test at the smallest scale, and the answer is
a rigorous negative for intonation constants: the prior's value there
is tuning bookkeeping, done with calibrated uncertainty. Where the
hypothesis could still hold, untested: longer spans with recurring
pitches at a distance (the whole-phrase loop), the other channels
(vibrato, dynamics), and the curve level — each needing the common-mode
term in every arm by design, which this experiment shows is mandatory.

**Proposed follow-ups (not run):** (i) the boundary-prior extension
(`docs/boundary_prior_proposal.md`, same design over the joint offsets,
with the offset scalar as its core control; expectation sharpened to
the same null); (ii) a phrase-scale version of the intonation
comparison where recurring pitches occur at a distance.

## H. Questions for the professor (prioritized)

1. **Objective:** Is "an interpretable, score-aligned representation of
   expressive performance, with uncertainty" the right overarching
   scientific objective for the thesis?
2. **Positioning:** Should the thesis primarily evaluate the
   inverse-inference methodology, with representation usefulness as
   motivation rather than a validated claim? (The draft now states
   exactly this split.)
3. **Scope of integration:** How much graph-prior × waveform integration
   is realistically necessary for a master's thesis — the full joint
   loop, or the three-prior comparison on short sequences plus the
   measured component studies?
4. **Variables:** Which expressive variables should be prioritized for
   direct waveform inference — pitch/vibrato (strong evidence), timing
   via boundaries (works, estimand gap), loudness/articulation (requires
   bringing amplitudes under the prior)?
5. **Evidence bar:** What constitutes sufficient evidence for the final
   contribution — a preregistered confirmation of one waveform claim, or
   development-level studies with the confirmed stage-1/2 results
   carrying the claims?
6. *(Standing, if time)* The registration-pool decision for the
   correlated-timing-noise and tonal-metric follow-ups (Bach10 /
   disclosed URMP re-use / both).

## If asked (quick answers)

- **"Why are the waveform bands so thin?"** Thousands of samples observe
  the curve through phase; we measured that the band is still too thin
  against the annotation (0.32–0.36 coverage; residual correlation
  0.944 diagnosed) and carry a measured calibration floor wherever the
  posterior is used. Thin-but-audited beats wide-but-underived.
- **"Is this DDSP?"** We took the implementation idea — gradients
  through a synthesizer (Engel et al. 2020) — and kept our exact
  marginalized likelihood and GP priors. No neural encoder, no spectral
  surrogate loss; inverse analysis, not generation. MIDI-DDSP (Wu et
  al. 2022) is the closest cousin and runs the opposite direction;
  interpretable-DDSP works (Yeh et al. 2024; Dal Rì and Conci 2026)
  share the interpretability goal on the synthesis side.
- **"What does the score give you once you have audio?"** Structure
  where evidence is weak: joints, repeated pitches, short notes, missing
  cells — and the across-note coupling no per-note method has. That is
  the next experiment.
- **"Can it generate?"** The prior defines a distribution over
  expression given a score; everything validated is the conditional
  (inference) direction. Generation quality is neither claimed nor
  measured.

---

*Historical prep (September measured-answers week, the zemi) lives in
git history of this file and `docs/zemi_prep.md`; numbers above trace to
`results/*.md`, `docs/ddsp_review.md`, and the thesis draft.*
