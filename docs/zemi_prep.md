# Zemi prep (2026-10-08, 20 minutes, English) — private

Deck: `docs/slides/deck_week.pdf`, spine = frames 2–26, backups 27–39.
Naming rule for the room: say **"the GP estimator"**; describe the
kernel ("two components: a damped cosine at the vibrato rate plus a
slow drift"); the one attribution sentence is on frame 11. Never
"SM-GP", never a coined acronym. Never bare "AR(1)": say "correlated
timing noise across notes".

## Opening line

"This is the thesis in one pass: one Gaussian process on the score
graph, three phases, and then the current question — two competing
estimators for the pitch-curve channels, and what their head-to-head
taught us about the interface."

## The naming correction, said once (at frame 12)

"One correction from last time, checked against the paper: the kernel I
use is the two-component special case of the spectral mixture kernel of
Wilson and Adams, but the METHOD is not theirs — their method learns a
many-component spectrum to discover structure; I imposed the structure
by hand. So I call this simply the GP estimator. And I ran their method
as intended, on 9,804 notes: it ties my hand-structured kernel at curve
level exactly, and finds the vibrato band unaided on 18 percent of
identifiable notes. The imposed structure costs nothing in fit and is
what defines the channels."

## Timing budget (1200 s)

| frames | content | budget |
|---|---|---|
| 2–8 | problem · approach · model (graph, coupling, covariance, architecture picture) · Phase 1 | 6 min |
| 9–14 | cents curve · channels · data · side-by-side · GP estimator · confirmation | 5.5 min |
| 15–20 | head-to-head · win · loss · verdict · vocabularies · design question | 5 min |
| 21–26 | timing fix · mechanism picture · Phase 3 · curve-from-waveform prototype · timeline · open questions | 3.5 min |

If running long: frames 22 (mechanism picture) and 23 (Phase 3) compress
to one sentence each. Frame 24 (the prototype) does NOT get cut: say
"built this week: the within-note GP prior and the waveform likelihood
meeting directly, no tracker, no estimator -- 2.5, 4.8, 1.0 cents
against ground truth on violin, cello, clarinet, versus pYIN's 3.4,
2.7, 4.6. The cello is the honest miss. This is the curve-level depth
of the design question, running."

## One-breath verdict (frame 17)

"The swap is conclusively off; at curve level the evidence favours the
GP without deciding it; the direction is inconclusive and its deciding
tests are known."

## Three likely questions

- **"Wasn't the test unfair to the GP?"** Two senses: the scoring was
  own-reference, so neither model was graded against the other's
  numbers. What is asymmetric is the output vocabulary: the three
  channels invert the sine fit exactly and pin the GP only to its
  coherent limit, which IS the sine model (frames 18/19). Fair for the
  swap decision, tilted for model comparison — hence the design
  question, not a re-score.
- **"What is the ground truth?"** The corpus annotates a pitch curve
  (corrected pYIN on isolated stems — gross errors hand-fixed), never
  channel values; channels are estimator-defined, hence own-reference.
- **"What next?"** The open-questions frame: pool for the timing
  registration, opt-in defaults, and the estimator design depth — with
  the learned-spectrum run (9,804 notes: exact tie at curve level, 18% band rediscovery)
  as the honest, measured version of the kernel question. Record:
  `results/sm_proper_dev.md`.

Deep Q&A reservoir: `docs/meeting_qa_prep.md` Part 2 (note its banner —
read "GP estimator" wherever it says SM).

---

## Speaker notes, frame by frame (the depth lives HERE, not on the slides)

- **2 Problem.** Score known, so not transcription: infer what the
  performer controlled, per note, with an error bar you can trust.
  Piano: timing, articulation, velocity. Strings/winds add pitch
  control: intonation and vibrato.
- **3 Approach.** One GP prior on a score graph; only the observation
  block changes per phase (MIDI, f0 targets, waveform). The badges show
  status: two phases confirmed, one scoped.
- **4 Score graph.** Edges = close in score time and pitch, so
  "neighbours play similarly". Laplacian L_G carries that geometry.
- **5 Coupled GP.** Shape the Laplacian spectrum with one parameter
  (graph Matern); couple channels with a learned B, so an observed
  channel at one note informs the others at neighbours.
- **6 Covariance.** The whole model is one per-piece matrix: graph term
  + feature terms (marginalized linear means, weights re-inferred per
  piece, THE accuracy source) + noise. Sixteen numbers, exact evidence,
  conjugate prediction. Baselines are exact special cases.
- **7 Architecture picture.** Walk the colors: blue graph branch, amber
  features, vermilion likelihood (the only block that changes later).
- **8 Phase 1.** Confirmed one-shot on 20 untouched pieces: RMSE 0.376
  vs 0.393 (pooled over channels, paired per piece), graph's log-score
  contribution significant, coverage 0.925 at nominal 0.9. Attribution:
  features carry the mean, the graph carries calibration. Known wart:
  the timing-outlier tail (fixed later in the talk).
- **9 Cents curve.** Frame rule: voiced, confidence above the lowest
  quintile, at least 4 frames. Depth if asked: GT = corpus annotation of
  the same isolated stem, corrected pYIN (46 ms window, 10 ms hop; only
  insertion/deletion/octave errors hand-fixed, cents-level structure is
  tracker output -- why we also say quasi-truth, and why the later test
  is own-reference).
- **10 Channels.** gamma is half the peak-to-peak swing; logs because
  positive with multiplicative errors. Refusal -> missing cells; the
  across-note GP just conditions on fewer cells.
- **11 Data figure.** A: tracker quantizes and drops frames, GT shows
  the oscillation underneath. B: identifiable. C: 11 frames, refused.
- **12 Side-by-side.** Same input, same output interface. Status line:
  approach 1 confirmed inside the pipeline, approach 2 measured,
  inconclusive. Detail of approach 1 in backups (grid + closed-form
  amplitudes, curvature covariance, delta method, gated variant for the
  vibrato onset, hard identifiability rules).
- **13 GP estimator + naming correction (say it here).** Kernel: damped
  cosine at the vibrato rate + drift. Coherent limit v1->0, w2=0 IS the
  sine model with phase marginalized. Fit by exact evidence, grid over
  the rate (multimodal), variances from curvature. NAMING: checked
  against the paper -- the kernel is the Q=2 special case of the
  spectral mixture kernel; the METHOD is not theirs (they learn a
  many-component spectrum, Q~10, to discover structure). And we ran
  their method as intended on 9,804 notes: exact tie at curve level,
  vibrato band found unaided on 18 percent. The structure must be
  imposed; it costs nothing in fit.
- **14 Confirmation.** C-claims spoken as names. Intonation recovery
  -0.877 cents; vibrato calibration -2.990/-0.564; coverage 0.88-0.91
  on all six; timing calibration failed (-0.030, CI includes zero),
  reported verbatim. Suspect named then: alignment error correlated
  along score time.
- **15 Head-to-head.** Own-reference because channels are
  estimator-defined. Measures: M1 parameter accuracy (pass/fail,
  pre-committed), M2 calibration, M3 refused notes, M4 curve level.
- **16 Win.** The GP tracks the curve the sine cannot (6.3 vs 9.6 cents
  median frame error); posterior splits into centre + drift + vibrato,
  read-outs are realized quantities (ensemble scale is chi-squared-2
  spread around the realized amplitude, median 0.83x).
- **17 Loss.** Same note, two measurements: sine says 2.5 Hz twice (its
  grid floor -- 13-18 percent of weak notes sit there), GP says 2.0 then
  9.5 Hz. Survives restriction to strong vibrato: intrinsic to
  point-picking a multimodal decomposition.
- **18 Verdict.** Swap conclusively off; curve level favours the GP
  without deciding it (paired mean grazes zero, an overconfidence tail
  remains); direction inconclusive, deciding tests known.
- **19 Vocabularies.** 5 vs 7 numbers. Channels invert the sine exactly
  (lossless round trip); they pin the GP to its coherent limit, which
  IS the sine -- everything the GP adds lives in coordinates the bundle
  does not carry. Interpretability is conditional on a well-identified
  decomposition, the same conditionality that closed the swap.
- **20 Design question.** Four depths: read-out redesign (marginalize
  the rate, gate by strength), extend the vocabulary (coherence
  mu1/sqrt(v1) + drift power channels -- then the GP inverts from the
  bundle like the sine does), curve-level channels (the two GP levels
  meet directly; Phase 3 is already curve-native), keep as is.
- **21 Timing fix.** One correlation parameter in the timing noise row,
  chosen per piece by the evidence (97-100 percent of pieces, median
  rho 0.45-0.60): NLL -0.111 significant, timing error 99 -> 80 ms,
  coverage to nominal. Three independent reproductions. Say "correlated
  timing noise across notes", never bare AR(1).
- **22 Mechanism picture.** One track's held-out timing: diagonal vs
  correlated noise intervals.
- **23 Phase 3 status.** Collapsed likelihood: amplitudes never
  estimated, O(mp^2) exact. Dev studies: 2.29 cents median with no
  tracker, beats the estimator chain on winds; 7th-channel integration
  better on 14/14 pairs.
- **24 Prototype (do not cut).** Built this week: within-note GP prior
  on a knot grid + collapsed waveform likelihood, MAP + diagonal
  Laplace, no tracker, no estimator. Violin 2.5 vs pYIN 3.4, clarinet
  1.0 vs 4.6, cello 4.8 vs 2.7 (the honest miss). Found on the way:
  knots must beat Nyquist for 5-6 Hz vibrato (hence 16 knots/s).
- **25 Timeline.** Sequencing, not dates; the first two items are gated
  by the open questions.
- **26 Open questions.** Pool (Bach10 / disclosed re-use / both);
  opt-in defaults; estimator design depth -- with the learned-spectrum
  run as the measured closure of "why not just use the paper's method".
