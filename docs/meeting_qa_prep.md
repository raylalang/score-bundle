# Meeting brief (written 2026-10-14, brought to 2026-11-18 for tomorrow's meeting)

> Purpose: report the state and the decisions taken, and hear objections.
> The deck (`docs/slides/deck.pdf`, 37 pages, speaker notes for every
> slide in `docs/slides/speaker_notes.md`) is the visual spine; this
> brief is the discussion skeleton. Standing language rules: say "the GP
> estimator" (never a coined name), "correlated timing noise across
> notes" (never bare "AR(1)"). Before the meeting: **sync Overleaf**
> (draft.tex changed on 10-21, 10-23, 10-27, 10-28: three spans, track-scale
> attribution, decomposition framing, number audit, the two re-used-pool
> one-shots in `sec:phase2-reuse`, the 10-28 cut to 74 pages).

## Run order for 2026-11-18 (written 2026-11-17, Ray back from a week at a conference)

Before the room: sync Overleaf with the whole `docs/thesis/` directory
(87 pages as of the morning of 11-18, the motif chapter and its framing
in); open `docs/slides/proposal_endgoal.pdf` (13 pages, internal),
`docs/slides/deck.pdf` (42 pages, the motif section is slides 32 to 36)
and `docs/ROADMAP.md` (the calendar table).

Morning of 11-18, before the meeting (done): the phrase-scale waveform
claim was frozen and tagged (`phrase-prior-registration-2026-11-18`,
pushed) and the one-shot launched at 04:47 on 12 shards (about 4 h),
and finished at 10:21: **W1 passed**, graph minus temporal −0.82 cents
[−1.59, −0.05] on 48 held-out phrases (development −1.63, the
pre-stated expectation for the winds-heavy mix about −0.9, power 0.3 to
0.5); graph minus independent −0.13, a tie as on development; clean
rows zero; by family strings −1.55, winds −0.54, brass −0.44, none
starred alone. Say: "the weak form holds on tracks no waveform study
had read; the strong form does not, there either." Thesis
`sec:phase3-heldout`, deck slide 30 fourth row. The thesis framing (abstract, intro, discussion, future
work, conclusion) and the deck's motif section were written the same
morning, verdict-free, so the writing week has only the held-out
sentences left.

1. **Where things stand, 60 s, no slide.** Thesis 87 pages, every
   confirmation pool spent, the main study complete, the motif chapter
   drafted with two sections pending the held-out run; one week lost to
   the conference; the zemi on 12-03 is the point where the results are
   done, December is writing.
2. **The proof of concept promised on 10-28, about 8 min** (frame 10
   of the proposal deck, speaker notes in
   `docs/slides/proposal_endgoal_notes.md`): the first pilot of 11-04
   (21 movements, 336 fits; with whole occurrences hidden the annotated
   motif edges lower the timing log score on those notes by 0.117 nats,
   interval [−0.245, −0.006]; articulation RMSE −5 percent; velocity
   bands too narrow under one shared weight; development only). If the
   pilot-two report is in by the morning (`results/motif_pilot2_dev.md`,
   the per-channel weight, four seeds, measured 11-17, proposal deck
   frame 11): the velocity penalty went away (−0.086 nats, coverage
   0.94) and the timing gain did not hold (−0.005; on four seeds the
   shared weight's own timing gain is −0.071 with an interval covering
   zero, the first pilot's star was two seeds); what holds on every seed
   pair and in both forms is articulation accuracy (RMSE −0.035 shared,
   −0.019 per channel, intervals clear of zero), and the evidence puts
   the family on the articulation channel. So the pre-specified test's
   primary is articulation RMSE, power 0.95 at 18 pieces. Stage two
   (frame 12, also measured 11-17): the ISMIR model's held-out
   occurrences in place of the annotations give about half the effect
   (timing 0.48, articulation 0.4 of the annotated one, intervals
   reaching zero); the rule asked for half with a clear interval and was
   not met by a small margin.
3. **The calendar to 12-03, one table** (ROADMAP): pilot two (done),
   stage two done (11-17), two
   registrations (the waveform phrase-scale claim on the URMP
   confirmation tracks, frozen, launched and PASSED the morning of 11-18;
   the motif claim on untouched performances of the same
   21 pieces, frozen 11-20, run 11-23), the thesis chapter and deck
   already carrying everything but the two verdicts, nothing new after
   December. Say the power check plainly: the waveform one-shot has 0.3
   to 0.5 on the drawn cells (strings 14, winds 25, brass 9 of 48), and
   its verdict is reported verbatim. Confirm the motif arm: the shared
   weight (power 0.95 against 0.93 for the per-channel form).
4. **The asks:** (a) the zemi slot's length and language on 12-03;
   (b) whether he will read the draft before the January talk, and by
   when; (c) the progress-talk date.

What is NOT on the table: modelling beyond the two registered one-shots,
a longer thesis, any rerun of a spent claim.

## Status note (2026-11-04)

No meeting on 11-04; Ray sent a written update instead. The proposal
deck (`proposal_endgoal.pdf`) is an INTERNAL document: Ray explained the
proposal orally on 10-28 and did not share the deck. The 11-04 pilot is
the proof of concept promised then. The run order below is kept for the
next meeting that has a slot for it.

## Run order written 2026-11-04 (superseded by the 11-18 order above)

Before the room: sync Overleaf with the whole `docs/thesis/` directory
(the draft is now split into `docs/thesis/chapters/`, rewritten in
academic voice on 10-28, 80 pages); open `docs/slides/proposal_endgoal.pdf`
(12 pages) and `docs/slides/deck.pdf` (37 pages).

1. **Where things stand, 60 s, no slide.** Thesis 80 pages, frozen except
   for one bounded study; the deck of 10-28 unchanged; every confirmation
   pool spent.
2. **The end-goal proposal, about 10 min** (`proposal_endgoal.pdf`,
   speaker notes in `docs/slides/proposal_endgoal_notes.md`): frames 2 to 3
   (the gap and the three candidates), 5 to 7 (the motif edge family and
   what it does to a hidden note), 8 (methodology), then **frame 10, the
   first measured pilot, run this morning** (21 movements, 336 fits: with whole occurrences hidden, the motif edges lower the timing log score on those notes by 0.117 nats, interval [−0.245, −0.006]; articulation RMSE −5 percent; velocity bands too narrow; development only). Frame 9 is
   the mock figure drawn last week; show it only to say the pilot replaced
   it. Frame 11 (the ISMIR model as stage two) if asked.
3. **The decision to ask for:** whether the motif study becomes the
   thesis's end-goal chapter (stage one on the annotations now, stage two
   with predicted occurrences after the January talk), or whether the
   thesis closes on the attribution result and the motif work stays a
   Future Work item. Say which outcome the pilot points to and why.
4. **The three asks** of 10-28 if they were not answered: reading the
   draft before January, the progress-talk slot, the writing freeze.

What is NOT on the table: new modelling runs outside the motif study, a
longer thesis, any rerun of a confirmation pool.

## Since 2026-10-28, in one breath

- The thesis was rewritten in academic voice on 10-28 (one file per
  chapter under `docs/thesis/chapters/`, a style guide and a checker that
  proves every number, label, equation and table body unchanged; 80
  pages). No content was added or cut.
- The end-goal proposal deck (`proposal_endgoal.pdf`) was written on
  10-28: the feedback application as drawn is decoration, its proper form
  (an across-performer model) is deferred to Future Work, and the
  proposal is the motif study: one more edge family in the score graph
  from the BPS-motif annotations, notes of two occurrences of one motif
  joined by a Gaussian in relative position inside the occurrence.
- **Today (11-04): the first pilot of that study was run** on the
  development side (`scripts/eval_motif_pilot_dev.py`,
  `results/motif_pilot_dev.md`): the ASAP Beethoven first movements
  aligned to the BPS-motif note lists (21 of 23 at 0.94 to 1.00 match;
  3-1 and 24-1 excluded as confirmation pieces), four graph arms on
  identical masks (time-only, time + pitch, each with and without the
  annotated motif edges, weight learned by evidence), two masking
  schemes, two seeds, 336 fits. The pre-stated primary (H1, timing log
  score on hidden motif notes when whole occurrences are hidden, motif
  arm vs time-only) is supported at the edge: −0.117 [−0.245, −0.006];
  the pitch term gives −0.014 there. Articulation RMSE 0.637 → 0.611
  (interval excludes zero). Velocity calibration worsens (+0.069 nats,
  coverage 0.88 → 0.82): one shared weight is the wrong design, a
  per-channel weight is the next step. Under the thesis's random masking
  the family is invisible. The evidence switches the family off on eight
  pieces and asks for it strongly on six. Proposal deck frame 10.

## Run order of 2026-10-28 (kept for the record)

Before the room: sync Overleaf; open `docs/slides/deck.pdf` (37 pages)
and this brief; optionally have the three clips in
`results/resynthesis_demo/` ready in a player (recording, resynthesis,
flat), 8 seconds each.

1. **Orientation, 60 s, no slide** (§A): the objective in one breath,
   and that the professor is hearing the state and the decisions, with
   objections welcome.
2. **Deck, main line, about 15 min.** Slide 3 (the thesis and the three
   problems), 4 (one model, three depths), 13 (piano result), 19
   (strings-and-winds result), 20 (the two re-used-pool one-shots: the
   timing claim repaired, the tonal metric missed), 30 (three spans), 31
   (time window: the graph's value is two-beat smoothing), 32 (summary),
   33 (plan to submission). Slides 7 to 9 (the model) and 22 to 23 (the
   waveform likelihood) only if the mathematics is asked for.
3. **Decisions as statements** (§H, 1 to 6), each in one sentence, then
   listen.
4. **The three asks of the professor**, after the decisions: (a) will he
   read the 74-page draft before the January progress talk, and by when;
   (b) the date and length of the progress talk slot (planned about
   2027-01-07); (c) any objection to the writing freeze, which is now
   in force until submission on 2027-02-02.
5. **If time remains:** play the resynthesis demo (quick answer below),
   then §F open problems as discussion.

What is NOT on the table: new modelling runs (the confirmation pools are
spent twice over), a longer deck, a new chapter.

## Since 2026-10-14, in one breath

- The score prior was measured inside the waveform inference at three
  spans (chains, joints, phrases): every prior transfers the context's
  common offset; the score graph beats a naive time prior at phrase
  scale; beyond the offset nothing measurable (§G, deck slide 30).
- At whole-track scale on both corpora, the score graph beats a time
  chain but EQUALS the same graph with its pitch term removed; on piano
  the pitch-less graph is slightly better. The graph's measured value is
  smoothness in score time over about two beats (§G, deck slide 31,
  thesis Appendix "Three priors at track scale").
- Decisions taken 2026-10-23 (Ray): that attribution is stated in the
  thesis, headline models keep their score graph; the four-layer framing
  (score, expression, timbre, noise) adopted; writing freeze in force;
  two more registered one-shots on the disclosed re-used URMP pool
  (correlated timing noise; tonal metric), frozen and RUN on 10-27.
- Results of 10-27 (deck slide 20, thesis `sec:phase2-reuse`): the
  correlated-timing-noise claim PASSED (timing log score −0.121
  [−0.156, −0.089], RMSE −0.011 s, coverage 0.908, ρ > 0 on 79 of 80
  cells): the failed August timing claim is repaired, confirmed once on
  the re-used pool. The tonal-metric calibration primary MISSED (−0.023
  [−0.049, +0.001]); its recovery claim held (−0.312 cents, starred);
  the metric stays exploratory. The unchanged arms reproduced the
  August table digit for digit.
- Thesis 74 pp (cut from 83 on 10-28 with the deck, 56 to 37 pages:
  obsolete two-stage-era material, tangential downstream tasks, the
  three-span section compressed; no number changed), every number
  audited against its record on 10-23 (three sentences corrected,
  `results/number_audit_2026-10-23.md`).
- Resynthesis demo (10-28, development, no claim,
  `results/resynthesis_demo.md`): the end-goal passage rendered back
  from the model's posterior-mean curves, inferred boundaries and
  amplitude posterior; the harmonic render explains 0.89 of the
  recording's energy over the fitted span; the remainder is timbre and
  noise by construction.

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
   chains and phrases, and the per-joint boundary offsets on short
   chains; its measured value is the context's common offset and
   nothing beyond it, plus a significant edge over a purely temporal
   prior at phrase scale (§G); the curve-level coupling, the remaining
   channels, and whole-performance scale are still open.

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
  three independent runs; registered and CONFIRMED once on the re-used
  pool, disclosed (2026-10-27: −0.121*, RMSE −0.011*).
- Tonal (circle-of-fifths) metric helps intonation: development win on
  both axes; registered one-shot on the re-used pool (2026-10-27) kept
  the recovery gain (−0.312*) and MISSED the calibration primary
  (−0.023 ns). Stays exploratory.

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
   built and measured three times over (§G): intonation at chain scale,
   timing at chain scale, intonation at phrase scale. Each time the
   prior transfers the context's common offset and nothing measurable
   beyond it; at phrase scale the score graph beats a purely temporal
   prior, by repairing that prior's harm. Design question now: is the
   strong form of the hypothesis worth pursuing further (the curve
   level, the other channels, whole phrases under the joint loop), or
   is the weak form — score structure beats naive temporal structure,
   and the common offset is the honest gain — the chapter's measured
   answer?
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

**Both follow-ups RUN (Ray's go 2026-10-18; measured 2026-10-20):**

- **The second span, boundaries** (`results/boundary_prior_dev.md`,
  47 chains, 40 non-unison): same design over the joint offsets from
  the tempo line, hidden joint removed from the design, offset scalar
  in every arm. Arms tie on the hidden joint (graph vs independent
  −2.45 ms [−8.38, +3.32]); every arm beats the tempo line by 10–14 ms
  (significant) — the common offset again; clean deployment identical
  across arms at 19.6 ms. Calibration: masked bands far too wide (the
  prior scale is a population scale), observed-joint bands far too
  narrow (0.7 ms: the acoustic-vs-musical estimand gap at the
  boundary).
- **The third span, phrase scale** (`results/prior_value_phrase_dev.md`,
  48 cells from all 78 development tracks, 6–10-note legato runs with
  the hidden pitch recurring 2+ notes away): the pre-stated primary
  contrast passes — graph beats temporal, −1.63 cents [−2.93, −0.46],
  65% — but the graph ties the scalar-augmented independent arm
  (−0.07) and the temporal chain is significantly worse than
  independent (+1.56): the graph's win is the temporal prior's harm
  repaired, with no dose-response on the edge weight.

**Track scale, 2026-10-22/23 (`results/phase2_chain_prior_dev.md`,
`results/phase2_timegauss_prior_dev.md`; the meeting of 10-21 was
cancelled and these were run in the gap).** The same three-prior
question on the tracked six-channel bundle, 78 development tracks, 153
cells, both arms carrying the per-piece bias column, so beyond the
common offset: the score graph beats a nearest-neighbour time chain
significantly (intonation −0.50 cents, vibrato extent −0.22, rate −0.21,
loudness −0.007), but equals the same graph with its pitch term removed
(intonation −0.02 [−0.10, +0.06]; vibrato slightly worse). The
confirmed strings-and-winds graph value (−0.88 cents) is reproduced by
the time-only Gaussian graph exactly. **The graph's value is temporal
smoothing over about two beats; the score-specific pitch coupling adds
nothing measurable at any scale tested.** Also: the boundary bands'
width fix (chain-scale prior) brings the independent arm to coverage
0.88, the structured arms to 0.75; verdict unchanged
(`results/boundary_prior_chainsig_dev.md`). Piano counterpart measured
(`results/piano_timegauss_dev.md`): the headline piano model with the
pitch term removed is slightly BETTER (RMSE −0.008*, NLL −0.019*,
paired on 30 development pieces), the gain in articulation. Same answer
on both corpora: the graph's value is smoothness in score time over
about two beats, including chord notes on piano; pitch proximity adds
nothing. The confirmed results stand (full model vs alternatives); the
attribution narrows. Adopting a pitch-less graph as the default would
be a reported-protocol change needing its own registration.

**The three-span verdict, for the room:** against the thesis's own
baselines (independent, purely temporal) the score-graph prior is never
worse and beats the temporal prior at phrase scale; against the
common-mode baseline it adds nothing measurable for intonation at
either span or for timing at chain scale. Weak form survives, strong
form fails so far. The positive content is honest and small: a note or
joint with no acoustic evidence inherits its context's common offset
with a calibrated band.

## H. The decisions, taken (2026-10-23) — for information, objections welcome

These were written on 10-14 as questions. The professor is hands-off
from here, so they were decided; state each as a decision and listen.

1. **Objective:** an interpretable, score-aligned representation of
   expressive performance, with uncertainty. Kept.
2. **Positioning:** the thesis evaluates the inverse-inference
   methodology; the representation's usefulness is motivation, not a
   validated claim. Kept, stated in the draft.
3. **Scope of integration:** the three spans plus the track-scale
   attribution ARE the waveform chapter's measured answer; the full
   joint loop over a whole performance is Future Work. Decided.
4. **Variables:** pitch and vibrato (strong evidence) and timing through
   boundaries (works, estimand gap) are the waveform variables;
   amplitudes under the prior are Future Work. Decided.
5. **Evidence bar:** the confirmed piano and strings-and-winds results
   carry the claims; the waveform is development level; two more
   registered one-shots (correlated timing noise, answering the failed
   timing claim; the tonal metric) on the disclosed re-used URMP pool,
   run once, reported whatever the outcome. Decided; RUN 10-27: timing
   noise passed, tonal metric missed its primary (see the top block).
6. **Pool:** disclosed URMP re-use. Bach10 as available on disk is the
   Sibelius-resynthesised version (synthetic audio), unusable for
   intonation or timing; the real Bach10 needs a request form and is not
   pursued this cycle.

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
- **"Can you resynthesize the audio?"** Yes, at the level the model
  represents: the demo of 10-28 renders the end-goal passage from the
  inferred curves, boundaries and amplitude posterior, and the flat
  clip (same envelopes, score-nominal pitch and timing) makes the
  expression audible by contrast. The harmonic render explains 0.89 of
  the recording's energy over the fitted span; the rest is timbre and
  noise, the layers the model integrates out by design. A waveform
  distance is therefore a diagnostic of the signal model, not an
  evaluation of the expression estimate; Chapter 8 evaluates that
  against the reference curves directly. Posterior-predictive
  resynthesis over held-out pieces is the Future Work protocol for the
  joint loop.
- **"Why is the deck shorter than last time?"** Cut on 10-28 from 56 to
  37 pages to findings only: notation and architecture frames,
  per-example figures, two-stage-era tables and the withdrawn-run
  backups removed; the model and waveform-likelihood frames merged with
  their full mathematics kept. The pre-cut deck is in git history.

---

*Historical prep (September measured-answers week, the zemi) lives in
git history of this file and `docs/zemi_prep.md`; numbers above trace to
`results/*.md`, `docs/ddsp_review.md`, and the thesis draft.*
