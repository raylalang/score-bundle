# Speaker's notes for `deck.tex` (the one living thesis deck)

Standing rule (Ray, 2026-10-21): **every change to `deck.tex` is followed
by a change to this file.** Frame numbers are PDF page numbers (page 1
is the title page). The note for each slide states what the slide says,
defines every symbol on it, and gives an example. Written at the
simplest level that is still complete.

Coverage: slides 33 to 49 (the waveform chapter, summary, plan, and the
backup frames). Slides 2 to 32 are not yet written; `docs/zemi_prep.md`
holds the older notes for the piano and strings-and-winds chapters.

---

## Slide 33. Setup: what the audio of one note is

**Data.** The same URMP development tracks as the strings-and-winds
chapter, 7 tracks, one instrument each. Each note is cut out of the
recording at its annotated start and end and resampled to 16,000
samples per second. 376 notes in the pitch-curve studies; 47 chains of
3 to 5 consecutive notes (154 boundaries) in the boundary study.

**The idea.** A sustained note is a stack of harmonics: a wave at the
fundamental frequency, another at twice it, another at three times, and
so on. Example: a violin A at 440 Hz also contains waves at 880, 1320,
1760 Hz.

**Symbols.**
- `f_pitch`: the frequency the score says the note should have. A4 is
  440 Hz.
- `cents(t)`: how far the played pitch deviates from `f_pitch` at time
  `t`, in cents; 100 cents is one semitone. Vibrato is `cents(t)`
  wobbling up and down.
- `f_0(t) = f_pitch · 2^(cents(t)/1200)`: the actual fundamental
  frequency at time `t`. Example: 10 cents sharp on A4 is about 442.5 Hz.
- `φ_k(t) = 2π k ∫_0^t f_0`: the phase of harmonic `k`. Think of an
  odometer: frequency is the speed, phase is the distance travelled.
  Harmonic `k`'s waveform is the cosine and sine of that distance. Phase
  accumulates, so a small pitch error grows into a large phase error
  over time. That is the property the method exploits.
- `z`: the pitch curve, the whole function `cents(t)`. The unknown.
- `Φ(z)`: a matrix with one column per harmonic and per cosine/sine,
  each column the wave that harmonic would produce if the curve were
  `z`. The curve fixes the shape of every column.
- `a`: the amplitudes, one per column, how loud each harmonic is: the
  timbre. They multiply the columns, like volume knobs.
- `ε`: whatever the harmonic stack does not explain, treated as noise.
- `x`: the whole waveform of the note as a vector, one entry per audio
  sample, NOT a single number. A recording is a sequence of pressure
  readings at a fixed rate; at 16,000 per second a 1.2-second note gives
  `m = 19,200` entries, each a value like 0.012 or −0.028. The vermilion
  line on slide 35 is `x` plotted against time; each point on it is one
  entry.
- `x = Φ(z) a + ε`: a matrix equation. `x` has `m` rows; `Φ(z)` has `m`
  rows and `p` columns, each column a waveform of length `m`; `a` has
  `p` entries; `ε` has `m` entries. Sample by sample: the recorded
  reading equals the sum of the harmonic waves at that instant, each
  scaled by its knob, plus noise. "Every audio sample votes" on slide 34
  means every one of the `m` entries contributes to a candidate's score.
- `K`: the number of harmonics, 8 in the studies.
- `m` and `p`, the two sizes of the matrix. `m` is the number of audio
  samples in the note, the length of `x` and the number of rows of
  `Φ(z)`: a 0.5-second note has `m = 8,000`, a 1.2-second note
  `m = 19,200`. `p` is the number of columns of `Φ(z)`, which is the
  number of knobs in `a`: cosine and sine per harmonic gives `2K = 16`,
  and the note is split into 2 time chunks so loudness can change
  within it, so `p = 2 × 8 × 2 = 32`. Scoring one candidate curve costs
  about `m · p²` operations (slide 34), here `19,200 × 32² ≈ 20 million`,
  milliseconds; the naive `m × m` form would cost `m³ ≈ 7 × 10¹²`. The
  intuition: `m` is huge and `p` is small, so thousands of samples pin
  down a few dozen loudnesses very precisely, and the pitch curve is
  what remains to be found.
- Technically (`phase3/curve.py`, `chunked_design_torch`, and the line
  `m, p = Phi.shape` in the likelihood): the time grid `t` is the vector
  of sample instants `[0, 1/16000, 2/16000, …]` and `m = len(t)`; every
  column of `Φ` is evaluated at those instants, so `Φ` has one row per
  instant and `x` one entry per instant. With a sample mask the masked
  rows are deleted from both, and `m` is the number of kept samples.
  The columns: for each harmonic `k = 1…n_harm`, the phase
  `φ_k(t) = 2π k · cumsum(f_0 dt)` gives two columns, `cos φ_k` and
  `sin φ_k` (the pair lets the harmonic take any starting phase); the
  time axis is then split into `n_chunk` equal blocks and the column set
  is repeated once per block, zeroed outside it, so each harmonic's
  loudness can differ per block. Hence `p = 2 · n_harm · n_chunk = 32`,
  and `a` has 32 entries: cosine and sine weight for each of 8 harmonics
  in each of 2 chunks. `Φ a` is the synthesized signal at every instant.
  The likelihood forms `Φ^T Φ`, `p × p`, never anything `m × m`. In a
  chain fit each note's `Φ` is built on its own slice plus a 0.2 s
  margin, multiplied by its fade windows, and the notes' blocks are
  concatenated side by side: `p` becomes the sum over notes and `m` the
  whole segment. That concatenation is where the leak of slide 48 lived.

## Slide 34. Setup: scoring a curve, finding the curve

**The problem.** We want `z` but do not know `a`. **The trick.** `a`
enters linearly, so with a Gaussian prior on the knobs we can average
over every knob setting exactly. The result is a score for a candidate
curve that no longer mentions the timbre:

- `p(x | curve) = N(x; 0, Φ Σ_a Φ^T + σ² I)`. Under this curve the
  audio should look like Gaussian noise with a covariance built from
  the harmonic waves. `Σ_a` is the prior covariance of the knobs, `σ²`
  the noise variance, `I` the identity. `m` is the number of samples,
  `p` the number of columns; the cost grows as `m · p²`, which is cheap.
- "Every audio sample votes": each of the thousands of samples
  contributes to the score. No pitch tracker anywhere; the raw waveform
  is compared with the candidate directly.
- **Candidates.** The score fixes `f_pitch`; the two-component Gaussian
  process of the strings-and-winds chapter says which deviations are
  plausible: a vibrato band near 5 to 6 Hz plus a slow drift. That turns
  "any curve" into "curves a player would produce".
- **Finding the best curve.** Synthesizer and score are differentiable
  in the curve, so: nudge the curve, check whether the score improves,
  repeat. Gradient descent through the synthesizer, the idea of Engel's
  DDSP applied with our exact likelihood. Seconds per note.
- **Uncertainty.** At the best curve, how sharply the score falls off
  when the curve is moved gives the band. Sharp peak, narrow band. This
  is the Laplace approximation ("curvature at the optimum").

## Slide 35. One note: the model against the raw audio

40 ms of a violin note with strong vibrato. Vermilion: the recorded
audio `x`. Grey dashed: the best single constant pitch. Blue: the
inferred pitch curve. The constant drifts out of alignment within 40 ms
because vibrato changes the odometer's speed; the curve stays locked.
This is phase accumulation made visible, and why the method is
sensitive to small pitch deviations.

## Slide 36. The curve from the waveform: three notes

Three development notes: violin, cello, clarinet. Blue: the curve from
the waveform. Orange dots: the corpus's ground-truth pitch frames, never
seen by the fit. Green dots: a standard tracker's frames (pYIN), also
scored against the ground truth. Legend numbers are median distance
from the truth in cents ("c"). Example: clarinet, 1.0 cents for the
waveform curve versus 4.5 for the tracker.

Honesty line in the caption: the 90 percent band around the blue curve
is about half a cent wide, hidden under the line. It is honest for the
curve itself, but the ground-truth frames come from their own
measurement chain, off by 2 to 4 cents, so the band does not and should
not cover them. That gap is carried separately as a calibrated floor
wherever the curve is used downstream.

## Slide 37. Note boundaries from the waveform

Several notes at once: a chain of 3 to 5 consecutive notes. Each
boundary is a crossfade: the earlier note fades out at one edge, the
later note fades in at another; both edges are parameters found by the
same gradient descent.

Panel A: a five-note violin passage with three kinds of vertical line.
Black: where the annotation says the note starts. Grey dashed: where
the score alone, through the local tempo line, predicts it ("what you
would guess from the score with no audio"). Blue: where the fit put it.
Panel B: the per-note pitch curves from the same fit; the 90 percent
band widens exactly at the joints, where two notes overlap and the
audio carries no boundary information.

Numbers: starting from the tempo line, the median boundary error
halves, 40 to 20 ms, over 128 boundaries. The remaining 20 ms is
systematic: the audio shows where the sound changes, the annotation
marks where the musician intended the note. A model of the attack
transient at the joint is the named next step.

## Slide 38. Where this chapter ends: the joint loop

The architecture of the complete intended system. Top path, blue: the
score graph from the model chapter with its spectral shaping, giving
the graph covariance `K_G`, and the coupling matrix `B` across
channels. Bottom path, amber: score features and the music model's
embeddings as linear kernels. These sum into the prior over every
note's expression `z`, the same prior as in the earlier chapters.
Right, orange: the audio likelihood of slides 33 and 34 replaces the
per-note observations; it is not Gaussian in `z`, so the posterior is
found by Laplace or variational inference, with the amplitudes `a`
recovered in closed form afterwards.

Bullets: one posterior for a whole recording; with no per-note targets
the evaluation becomes held-out pieces and resynthesis from the
posterior; solid blocks are measured (curves, boundaries), dashed ones
(transient model; loudness and articulation under the prior) are not
built.

## Slide 39. The end goal on one passage (development)

A real 13-note, 8-second violin passage from the score and the waveform
alone. Panel A: the recording segmented into notes by the inferred
boundaries; annotated onsets drawn only as reference. Panel B: each
note's pitch expression with its band, against annotation frames the
fit never saw. Panel C: the timbre strand, each note's harmonic
envelope read off the amplitude posterior, alternating colours for
adjacent notes. The gap in the middle: notes outside the
chain-eligibility rule, honestly left unfitted. Caveat: chains fitted
independently, envelopes read out after the fit rather than inferred
under the prior. That is the open work.

## Slide 40. Summary

Three sentences, say them exactly. Piano: beats the two-stage baseline
on accuracy and calibration, confirmed once on untouched pieces.
Strings and winds: the graph improves recovery and calibration on the
estimated channels, confirmed once. Audio: the method is in place and
running at development level, no claims.

## Slide 41. Plan

October, done: curve studies at scale, the curve as a bundle channel,
joint boundaries. November: the registration-pool decision, the
one-shot timing confirmation, the scalars-or-curve-level channel
decision, a transient model at the boundary. December: whole-phrase
joint inference, chapters consolidated as results land. This row does
not mention the three spans of 2026-10-15..20; if asked, go to the
backup frames.

## Slide 42. Divider: Backup

Shown only if asked.

## Slide 43. Backup: the curve against the tracker, at scale

376 notes, 7 tracks. Dumbbell plot: per instrument family, blue is the
waveform curve's median error against the annotation frames, green the
tracker's. All notes 3.16 vs 4.09 cents. Winds 2.26 vs 5.15, a clear
win. Brass 3.15 vs 3.51. Strings 4.90 vs 3.09, the tracker wins. Two
honest edges: strings favour the tracker, and the waveform fit's worst
cases are worse (90th percentile 10.6 vs 8.6 cents). Second bullet: the
inferred curve added to the strings-and-winds bundle as a seventh
observation channel improves held-out intonation on 14 of 14
track-and-seed pairs, by 1.36 cents, significant, and the calibration
floor halves, 3.1 to 1.8 cents.

## Slide 44. Backup: boundary accuracy from score-only starting points

One dot per boundary, 128. Horizontal: the error of the score-only
tempo-line guess. Vertical: the error after the joint fit. Below the
diagonal improved: 73 percent. Median 40 to 20 ms. Repeated-pitch
boundaries excluded: identical harmonic stacks on both sides carry no
boundary information. Dots far above the diagonal: the systematic
acoustic-versus-written remainder.

## Slide 45. Backup: the score prior inside the waveform inference, the design

The week of 2026-10-15..20. Question: when a note's audio is missing,
does the score prior help guess it from its neighbours? Three priors
over one number per note, the intonation constant `c` (the note's
average pitch deviation, cents), or for the boundary study one number
per joint, the offset `db` from the tempo line (ms):
- independent: notes know nothing of each other;
- temporal chain: each note linked to the next in time;
- score graph: notes linked when close in time and pitch, so a
  recurring pitch gets a direct link.
Same total spread, same likelihood, data and optimizer. The hidden note
has its audio removed and its model columns dropped, so only the prior
sets it. Every arm also has a free common-mode scalar `s`, one number
shared by all notes, absorbing the player's overall tuning offset.
Example: a run 12 cents sharp; `s` learns 12, the independent prior can
use it too, so the comparison measures what structure adds beyond that.

## Slide 46. Backup: one chain end to end

Panel A: a five-note violin chain, the middle note's samples hatched
(masked) and its columns dropped. Panel B: for each note the posterior
of `c` under the three priors with 90 percent bands, against the
quasi-truth (black bar). The observed notes are pinned by the audio to
the same value under every prior. The hidden note gets the prior's
conditional, what the prior expects given the neighbours. On this chain
all three land near the chain's mean, 16 to 19 cents, while the truth is
29.5: this note is sharper than its context. That is what "prior
conditional" looks like.

## Slide 47. Backup: the three spans, paired

Three panels, one per test. Each row: the mean paired difference in the
hidden element's error between two priors, with a 95 percent bootstrap
interval. Negative: the first-named prior is better. Star: the interval
excludes zero.
- Left, intonation on 47 chains (cents): score graph minus independent
  −0.02, a tie.
- Middle, boundaries on 40 joints (ms): −2.45, a tie.
- Right, intonation on 48 phrases where the hidden pitch recurs two or
  more notes away: score graph minus temporal −1.63, significant;
  temporal minus independent +1.56, significant (the temporal prior
  hurts); score graph minus independent −0.07, a tie.
Third bullet, the positive content: the common offset alone, which
every prior gets through `s`, is worth about 7 cents on intonation and
10 to 14 ms on boundaries against the no-transfer references.

## Slide 48. Backup: why the first intonation run was withdrawn

Panel A: the objective as the hidden note's `c` sweeps −100 to +100
cents on one chain, first protocol. Vermilion: the likelihood term;
grey dashed: the prior term; black: their sum. A nat is one unit of
log-likelihood, an evidence point. The leaked likelihood varies by 38
nats across the sweep, the prior by 4 to 35: the leak outvoted the
prior. The leak: masking the hidden note's samples did not remove its
model columns, and the soft edges of its window left a tiny weight on
the neighbours' audio; with amplitudes free, that tiny weight became
loud wrong evidence. Panel B: under that protocol every prior's
estimate against the truth is a cloud with correlation zero. Panel C:
with the note dropped from the model, the independent prior returns
exactly the prior mean (zero) and the structured priors track the truth
through the common offset.

## Slide 49. Backup: calibration of the hidden element's band

A z-score is the error divided by the claimed uncertainty. Honest bands
put 90 percent of z-scores inside ±1.645 (the shaded band); "cov" above
each strip is the fraction that did. Intonation at chain and phrase
scale: 0.89 to 0.96, honest. Hidden joints in the boundary study: 1.00,
bands too wide, because the prior's spread was set from the track's
whole timing variability. Observed joints on clean audio: 0.05, 27 of
40 beyond ±8; the model claims 0.7 ms precision because it pins the
acoustic transition exactly, while the annotation marks the musical
onset about 20 ms away. The estimand gap of slide 36, at the boundary.

---

## If stopped on one slide

The two most likely probes are 34 (how the timbre is removed) and 47
(what the three spans say). One-line answers: "the amplitudes are
linear, so we integrate them out exactly," and "the score prior gives a
missing note its context's offset with an honest band, beats a
time-only prior at phrase scale, and so far nothing more."
