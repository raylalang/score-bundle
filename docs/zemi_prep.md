# Zemi prep (2026-10-08, 20 minutes, English) — private

Deck: `docs/slides/deck_week.pdf`, 22 slides, no backups. Audience: lab
members with no prior context. Say "the GP estimator", never "SM-GP";
say "correlated timing noise across notes", never bare "AR(1)".

## Opening line

"One model: a Gaussian process on the score graph. Score and performance
in, per-note expressive variables out, each with an error bar you can
trust. I'll show the model, what it does on piano and on strings and
winds, and where it's going: reading pitch curves straight from audio."

## Timing budget (1200 s)

| slides | content | budget |
|---|---|---|
| 2–8 | expression concept · thesis + three problems · approach · model | 6.5 min |
| 8–11 | cents curve · channels · data figure · Phase 1 results | 4 min |
| 12–14 | GP estimator · one-note figure · Phase 2 results | 4 min |
| 15–16 | Phase 3 · curve-from-waveform prototype | 3 min |
| 17–18 | correlated timing noise + picture | 2 min |
| 19–20 | timeline · open questions | 1 min |

If long: slides 15 and 18 compress to a sentence. Slide 16 (prototype)
does not get cut.

## Speaker notes, slide by slide

- **2 Expressive performance.** The concept picture: written vs
  played, the deviations ARE the expression. Not transcription: the
  score is known. AUDIO DEMO: present from docs/slides/deck_week.pptx (the full deck
  as PowerPoint, both clips embedded on this slide: flat = score MIDI
  at uniform velocity, expressive = a real ASAP performance). Or play docs/slides/audio/{flat,expressive}.wav in any player. Sine
  synthesis, so timing and dynamics carry the contrast, say so.
- **3 Thesis + three problems.** Say the thesis sentence verbatim, then
  one problem per phase with its own variables: piano (timing,
  articulation, velocity, read exactly), strings/winds (six, estimated,
  noisy), audio (no tracker). Every later slide answers one of the
  three; each phase gets a setting slide before its results.
- **(new) Phase 1 setting.** Disklavier = exact measurement, no
  estimation anywhere; the concept picture was this data. Phase 1 asks
  only: good estimates, trustworthy error bars, on held-out notes.
- **3 Approach.** One GP prior on a score graph; only the observation
  block changes per phase (MIDI, f0 targets, waveform).
- **4 Graph.** Edges = close in score time and pitch: neighbours play
  similarly.
- **5 Coupled GP.** Shape the Laplacian spectrum with one parameter;
  couple channels with learned B: an observed channel at one note
  informs the others at neighbours.
- **6 Covariance.** One per-piece matrix: graph + features (weights
  re-inferred per piece, the accuracy source) + noise. Sixteen numbers,
  exact evidence, conjugate prediction.
- **7 Picture.** Blue graph branch, amber features, vermilion
  likelihood (the only block that changes later).
- **8 Phase 1.** Preregistered one-shot on 20 untouched pieces: RMSE
  0.376 vs 0.393 (pooled, paired), calibration contribution
  significant, coverage 0.925 at nominal 0.9. Features carry the mean,
  the graph carries calibration.
- **9 Cents curve.** Frame rule: voiced, confidence above the lowest
  quintile, >= 4 frames. If asked about GT: the corpus's own annotation
  of the same isolated stem (corrected pYIN, 46 ms / 10 ms) — cleaner,
  still a measurement, hence "quasi-truth" and own-reference scoring.
- **10 Channels.** gamma = oscillation amplitude; logs because positive
  with multiplicative errors. No oscillation -> missing cells, the GP
  conditions on fewer cells.
- **11 Data figure.** A: tracker quantizes, GT shows the oscillation
  underneath. B: identifiable. C: refused.
- **12 GP estimator + naming note (say once).** Kernel: damped cosine
  at the vibrato rate + drift — a special case of the spectral mixture
  kernel; the method is ours (structure imposed), not the SM paper's
  spectral learning. We also ran their method as intended on 9,804
  notes: exact tie at curve level, vibrato band found unaided on 18
  percent — the structure must be imposed and costs nothing. Channels
  are realized quantities read from the posterior (ensemble scale is
  chi-squared-2 spread around the realized amplitude).
- **13 One-note figure.** The GP tracks what the player did; the rigid
  sine reference drifts out of phase. The posterior decomposes into
  centre + drift + vibrato: that split IS the channel read-out.
- **14 Phase 2 results.** Six channels, estimator variances used as
  given. Deltas vs the no-graph ablation: intonation −0.877 cents,
  vibrato NLL −2.990/−0.564, coverage 0.88–0.91 on all six. Timing did
  not improve — the loose end, after Phase 3.
- **15 Phase 3.** Likelihood becomes the audio; amplitudes marginalized
  exactly, O(mp^2). Dev studies: 2.29 cents median with no tracker;
  beats the estimator chain on winds.
- **16 Prototype (do not cut).** Built this week: within-note GP prior
  + waveform likelihood, MAP + diagonal Laplace, no tracker, no
  estimator. Violin 2.5 vs pYIN 3.4 cents, clarinet 1.0 vs 4.6, cello
  4.8 vs 2.7 (the miss, say it plainly). Knots must beat Nyquist for
  5–6 Hz vibrato: 16 knots/s.
- **17 Timing.** Why it didn't calibrate: alignment error correlated
  along score time; diagonal noise can't represent it. One parameter
  rho per piece, chosen by the evidence (97–100 percent of pieces):
  NLL −0.111 significant, timing error 99 -> 80 ms, coverage to
  nominal. Three reproductions.
- **18 Picture.** One track's held-out timing, with and without the
  correlation.
- **19 Timeline.** Sequencing, not dates; gated by the open questions.
- **20 Open questions.** Corpus for confirming the timing result;
  scalar channels vs curve level (the prototype's direction); the
  robustness options as defaults.

## Likely questions

- **"Why a GP and not a parametric fit?"** We measured both ways: a
  rigid sine fit is more stable at producing the three scalars (it IS
  those scalars); the GP is at least as good at describing the curve
  and never refuses a note, reporting wide variance instead. The deeper
  question is whether scalars are the right interface at all — that is
  open question 2. (Full record: results/sm_estimator_dev.md.)
- **"What is the ground truth?"** A curve, not channel values: corrected
  pYIN on isolated stems. Channels are estimator-defined.
- **"Why does one correlation parameter cut timing error 20 percent?"**
  A neighbour's warp error predicts yours; the correlated noise row
  lets the model subtract it.
