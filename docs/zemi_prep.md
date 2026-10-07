# Zemi prep (2026-10-08, 20 minutes, English) — private

Deck: `docs/slides/deck_week.pdf`, spine = frames 2–24, backups 25–37.
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

## The naming correction, said once (at frame 11)

"One correction from last time, checked against the paper: the kernel I
use is the two-component special case of the spectral mixture kernel of
Wilson and Adams, but the METHOD is not theirs — their method learns a
many-component spectrum to discover structure; I imposed the structure
by hand. So I call this simply the GP estimator. And I ran their method
as intended: at pilot scale (253 notes) it ties my hand-structured
kernel at curve level and finds the vibrato band unaided on about a
quarter of identifiable notes. The imposed structure costs nothing in
fit and is what defines the channels."

## Timing budget (1200 s)

| frames | content | budget |
|---|---|---|
| 2–7 | problem · approach · model (graph, coupling, covariance) · Phase 1 | 6 min |
| 8–13 | cents curve · channels · data · side-by-side · GP estimator · confirmation | 5.5 min |
| 14–19 | head-to-head · win · loss · verdict · vocabularies · design question | 5.5 min |
| 20–24 | timing fix · mechanism picture · Phase 3 · curve-from-waveform prototype · open questions | 3.5 min |

If running long: frames 21 (mechanism picture) and 22 (Phase 3) compress
to one sentence each. Frame 23 (the prototype) does NOT get cut: say
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
  coherent limit, which IS the sine model (frame 17/18). Fair for the
  swap decision, tilted for model comparison — hence the design
  question, not a re-score.
- **"What is the ground truth?"** The corpus annotates a pitch curve
  (corrected pYIN on isolated stems — gross errors hand-fixed), never
  channel values; channels are estimator-defined, hence own-reference.
- **"What next?"** The open-questions frame: pool for the timing
  registration, opt-in defaults, and the estimator design depth — with
  the learned-spectrum pilot (ties at curve level, 27% band rediscovery)
  as the honest, measured version of the kernel question. Record:
  `results/sm_proper_dev.md`.

Deep Q&A reservoir: `docs/meeting_qa_prep.md` Part 2 (note its banner —
read "GP estimator" wherever it says SM).
