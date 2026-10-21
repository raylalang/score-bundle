# Speaker's notes for `deck.tex` (the one living thesis deck)

Standing rule (Ray, 2026-10-21): **every change to `deck.tex` is followed
by a change to this file.** Frame numbers are PDF page numbers (page 1
is the title page). The note for each slide states what the slide says,
defines every symbol on it, and gives an example. Written at the
simplest level that is still complete.

Coverage: slides 33 to 51 (the waveform chapter, summary, plan, and the
backup frames). Slides 2 to 32 are not yet written; `docs/zemi_prep.md`
holds the older notes for the piano and strings-and-winds chapters.

---

## Slide 33. Setup: the data and the signal model

**What is on the slide.** The data line; the one equation
`x = Φ(z) a + ε` with its sizes (`x ∈ R^m`, `Φ(z) ∈ R^{m×p}`, `a ∈ R^p`,
`ε ~ N(0, σ² I_m)`); and a five-row table naming each symbol. The
message: the curve fixes the SHAPE of every column, the amplitudes only
SCALE them. The three slides that follow unpack, in order, how `z`
becomes `Φ(z)`, how a curve is scored without knowing `a`, and how the
best curve and its band are found.

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
- `ε`: the noise term, a vector with `m` entries, one per sample: the
  part of the recording the harmonic stack does not reproduce,
  `ε = x − Φ(z) a`. The model assumes each entry is an independent
  Gaussian with mean zero and common variance, `ε ~ N(0, σ² I)`; that
  is what makes slide 34's likelihood a Gaussian with covariance
  `Φ Σ_a Φ^T + σ² I` (spread from the unknown amplitudes, plus the noise
  floor on every sample). `σ²` is not chosen by hand: per note, the
  code solves the best amplitudes by least squares on the flat-curve
  design and takes the residual's variance as `σ²`, refitting it once
  halfway through the curve fit. In a real recording `ε` contains bow
  and breath noise, room reverberation, harmonics above the eighth,
  slight inharmonicity, and the mismatch between piecewise-constant
  amplitudes and the true envelope. Caveat, said on slide 38: it is not
  white; the residual at the harmonics is strongly correlated across
  samples (0.944 measured), so the likelihood counts correlated samples
  as independent votes and is overconfident, which is why the
  calibrated floor exists.
- `x`: the whole waveform of the note as a vector, one entry per audio
  sample, NOT a single number. A recording is a sequence of pressure
  readings at a fixed rate; at 16,000 per second a 1.2-second note gives
  `m = 19,200` entries, each a value like 0.012 or −0.028. The vermilion
  line on slide 37 is `x` plotted against time; each point on it is one
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
  `sin φ_k`; the pair lets the harmonic take any starting phase: by
  `A cos(φ − θ) = (A cos θ) cos φ + (A sin θ) sin φ`, a sinusoid of any
  amplitude `A` and any starting phase `θ` is a cosine column times one
  weight plus a sine column times another, so the two weights
  `a_cos = A cos θ`, `a_sin = A sin θ` absorb the unknown starting phase
  (example: `a_cos = 0.3, a_sin = 0.4` is `A = 0.5`, `θ = 53°`) and the
  timbre stays linear in `a`, which is what lets slide 35 integrate it
  out exactly; the
  time axis is then split into `n_chunk` equal blocks and the column set
  is repeated once per block, zeroed outside it, so each harmonic's
  loudness can differ per block. Hence `p = 2 · n_harm · n_chunk = 32`,
  and `a` has 32 entries: cosine and sine weight for each of 8 harmonics
  in each of 2 chunks. `Φ a` is the synthesized signal at every instant.
  The likelihood forms `Φ^T Φ`, `p × p`, never anything `m × m`. In a
  chain fit each note's `Φ` is built on its own slice plus a 0.2 s
  margin, multiplied by its fade windows, and the notes' blocks are
  concatenated side by side: `p` becomes the sum over notes and `m` the
  whole segment. That concatenation is where the leak of slide 50 lived.

**From `z` to `Φ(z)`, step by step** (`chunked_design_torch`).
1. `z` is the pitch deviation at every sample instant, `cents(t_i)` for
   `i = 1…m` (stored as `J` knot values and interpolated to the grid).
2. Deviation to frequency, per sample:
   `f_0(t_i) = f_pitch · 2^(cents(t_i)/1200)`.
3. Frequency to phase, by accumulation (the odometer):
   `φ(t_i) = 2π Σ_{j ≤ i} f_0(t_j) Δt`, a cumulative sum with
   `Δt = 1/16000`.
4. Harmonic `k` has `k` times the phase: `φ_k(t_i) = k · φ(t_i)`. Its two
   columns are `cos φ_k(t_i)` and `sin φ_k(t_i)` for `i = 1…m`.
5. Chunking: copy the `2K` columns once per time block, zeroed outside
   the block.
6. Stack all columns side by side: `Φ(z)`, `m × p`.
So `Φ` depends on `z` only through the phase, and changing `z` changes
the shape of every column; `a` never touches `z`. Toy example: one
sample per millisecond, `f_pitch = 100` Hz, `cents = 0` throughout, so
`f_0 = 100` Hz and the phase advances `2π · 100 · 0.001 = 0.628` rad
(36°) per sample. Harmonic 1's cosine column over four samples is
`[cos 36°, cos 72°, cos 108°, cos 144°] = [0.81, 0.31, −0.31, −0.81]`,
its sine column `[0.59, 0.95, 0.95, 0.59]`; harmonic 2 uses 72°, 144°,
216°, 288°. Raise the whole note one semitone (`cents = +100`): `f_0 =
105.9` Hz, the step becomes 38.1°, every column changes shape. Let the
cents wobble (vibrato): the step varies sample to sample, and the
columns become chirped sinusoids. That is the whole bridge from a pitch
curve to a matrix.

**Sizes of everything on slides 33 to 36, symbolically.**

| object | size | what it is |
|---|---|---|
| `m` | scalar | number of samples in the segment |
| `K` (`n_harm`) | scalar | number of harmonics |
| `C` (`n_chunk`) | scalar | time chunks per note |
| `p = 2KC` | scalar | basis columns for one note |
| `t`, `x`, `ε`, `f_0(t)`, `φ_k(t)` | `m × 1` | sample instants, samples, noise, frequency, phase of harmonic `k` |
| `z` | `J × 1` as a parameter (`J` knots per note), `m × 1` on the grid | the pitch curve |
| `a` | `p × 1` | amplitude weights |
| `Φ(z)` | `m × p` | design matrix: row = instant, column = basis waveform |
| `Σ_a` | `p × p` (`σ_a² I_p` in the studies) | amplitude prior covariance |
| `σ² I` | `m × m` | noise covariance |
| `Φ Σ_a Φ^T + σ² I` | `m × m` | covariance of `x` under a candidate curve; never formed |
| `G = Φ^T Φ + (σ²/σ_a²) I_p`, its Cholesky `L` | `p × p` | what the code forms instead |
| `Φ^T x` | `p × 1` | the data projected on the basis |

For a chain of `N` notes (slide 39): each note has its own `Φ_i`,
`m × p_i` with `p_i = 2KC`, on its own slice plus margin times its fade
windows; the chain design is `Φ = [Φ_1 … Φ_N]`, `m × P` with
`P = N · 2KC`; `a` is `P × 1`, `G` is `P × P`; the boundary offsets `db`
are `(N−1) × 1` with an `(N−1) × (N−1)` prior covariance (slide 47);
the per-note constants `c` are `N × 1` with an `N × N` prior covariance.
Cost `O(m p²)`: forming `Φ^T Φ` is `m` rows times `p × p` entries; the
Cholesky of `G` is `p³`, negligible since `p ≪ m`.

## Slide 34. From the pitch curve to the design matrix

**What is on the slide.** The chain, four labelled lines: pitch curve
`z = cents(t)`; frequency `f_0(t) = f_pitch · 2^(cents(t)/1200)`; phase
`φ_k(t) = 2π k ∫_0^t f_0`, `k = 1..K`; design
`Φ(z) = [cos φ_k(t), sin φ_k(t)]_{k=1..K} ∈ R^{m×p}`. One sentence of
intuition (frequency is the speed, phase the distance travelled) and a
small-print line on the two implementation details: the curve is held
at `J` knot values and interpolated; the columns are repeated per time
chunk, `p = 2KC`. The step-by-step reading with the toy example is in
the slide 33 notes above ("From `z` to `Φ(z)`").

**If asked why the cosine and sine pair:** the two columns with two
weights produce a sinusoid of any amplitude and any starting phase
(`A cos(φ − θ) = A cos θ · cos φ + A sin θ · sin φ`), so the unknown
starting phase of each harmonic is absorbed into `a` and the timbre
stays linear. That linearity is what the next slide needs.

## Slide 35. Scoring a curve: the timbre integrated out

**What is on the slide.** One implication: `a ~ N(0, σ_a² I_p)` implies
`p(x | z) = N(x; 0, σ_a² Φ(z) Φ(z)^T + σ² I_m)`. Then three bullets:
one number per candidate curve, timbre-free, every sample votes, no
tracker; it is computed through the `p × p` matrix
`G = Φ^T Φ + (σ²/σ_a²) I_p`, never the `m × m` covariance, at cost
`O(m p²)`; and `σ²` is the residual variance at the best amplitudes,
fitted per note.

**Symbols.** `σ_a²` is the amplitude prior variance (`amp_var` in
code); `I_p` and `I_m` are identities of the two sizes; `G` is what the
code forms, and the log-likelihood is evaluated through `G` and its
determinant by the Woodbury identity (the full expansion is in the
thesis; it is not needed to follow the flow).

**The problem.** We want `z` but do not know `a`. **The trick.** `a`
enters linearly, so with a Gaussian prior on the knobs we can average
over every knob setting exactly. The result is a score for a candidate
curve that no longer mentions the timbre:

- `p(x | z) = N(x; 0, Φ(z) Σ_a Φ(z)^T + σ² I)` (the slide wrote "curve"
  in words until 2026-10-21; `z` IS the curve, slide 33). Under this
  curve the audio should look like Gaussian noise with a covariance
  built from the harmonic waves. In the thesis `z` names every
  nonlinear position the synthesizer needs, so in a chain fit it also
  holds the boundary offsets; on slides 33 to 36, one note with fixed
  boundaries, `z` is just the pitch curve. `Σ_a` is the prior covariance of the knobs, `σ²`
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

## Slide 36. Finding the curve, with its uncertainty

**What is on the slide.** Three labelled lines: the prior on the curve
`u ~ N(0, K_knots)`; the best curve `ẑ = argmax_z [log p(x|z) + log
p(z)]`; its band `Σ_z ≈ [−∇²_z (log p(x|z) + log p(z))]⁻¹ at ẑ`. Then
three bullets: the score anchors `f_pitch` and `K_knots` is the
two-component GP (vibrato band plus drift), so candidates are curves a
player would produce; `Φ(z)` is differentiable in `z`, so gradient
descent through the synthesizer, seconds per note; the band is the
Laplace approximation, a sharply curved peak gives a narrow band.

**Symbols.** `u` are the knot values of the pitch curve and `K_knots`
their prior covariance, built from the two-component Gaussian process
of the strings-and-winds chapter and fitted by evidence. `∇²` is the
matrix of second derivatives of the objective at the optimum, the
curvature; its inverse is the posterior covariance of the curve
parameters under the Laplace approximation, and the 90 percent band on
later slides is 1.645 times the square root of its diagonal.

## Slide 37. One note: the model against the raw audio

40 ms of a violin note with strong vibrato. Vermilion: the recorded
audio `x`. Grey dashed: the best single constant pitch. Blue: the
inferred pitch curve. The constant drifts out of alignment within 40 ms
because vibrato changes the odometer's speed; the curve stays locked.
This is phase accumulation made visible, and why the method is
sensitive to small pitch deviations.

## Slide 38. The curve from the waveform: three notes

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

## Slide 39. Note boundaries from the waveform

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

## Slide 40. Where this chapter ends: the joint loop

The architecture of the complete intended system. Top path, blue: the
score graph from the model chapter with its spectral shaping, giving
the graph covariance `K_G`, and the coupling matrix `B` across
channels. Bottom path, amber: score features and the music model's
embeddings as linear kernels. These sum into the prior over every
note's expression `z`, the same prior as in the earlier chapters.
Right, orange: the audio likelihood of slides 33 to 35 replaces the
per-note observations; it is not Gaussian in `z`, so the posterior is
found by Laplace or variational inference, with the amplitudes `a`
recovered in closed form afterwards.

Bullets: one posterior for a whole recording; with no per-note targets
the evaluation becomes held-out pieces and resynthesis from the
posterior; solid blocks are measured (curves, boundaries), dashed ones
(transient model; loudness and articulation under the prior) are not
built.

## Slide 41. The end goal on one passage (development)

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

## Slide 42. Summary

Three sentences, say them exactly. Piano: beats the two-stage baseline
on accuracy and calibration, confirmed once on untouched pieces.
Strings and winds: the graph improves recovery and calibration on the
estimated channels, confirmed once. Audio: the method is in place and
running at development level, no claims.

## Slide 43. Plan

October, done: curve studies at scale, the curve as a bundle channel,
joint boundaries. November: the registration-pool decision, the
one-shot timing confirmation, the scalars-or-curve-level channel
decision, a transient model at the boundary. December: whole-phrase
joint inference, chapters consolidated as results land. This row does
not mention the three spans of 2026-10-15..20; if asked, go to the
backup frames.

## Slide 44. Divider: Backup

Shown only if asked.

## Slide 45. Backup: the curve against the tracker, at scale

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

## Slide 46. Backup: boundary accuracy from score-only starting points

One dot per boundary, 128. Horizontal: the error of the score-only
tempo-line guess. Vertical: the error after the joint fit. Below the
diagonal improved: 73 percent. Median 40 to 20 ms. Repeated-pitch
boundaries excluded: identical harmonic stacks on both sides carry no
boundary information. Dots far above the diagonal: the systematic
acoustic-versus-written remainder.

## Slide 47. Backup: the score prior inside the waveform inference, the design

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

## Slide 48. Backup: one chain end to end

Panel A: a five-note violin chain, the middle note's samples hatched
(masked) and its columns dropped. Panel B: for each note the posterior
of `c` under the three priors with 90 percent bands, against the
quasi-truth (black bar). The observed notes are pinned by the audio to
the same value under every prior. The hidden note gets the prior's
conditional, what the prior expects given the neighbours. On this chain
all three land near the chain's mean, 16 to 19 cents, while the truth is
29.5: this note is sharper than its context. That is what "prior
conditional" looks like.

## Slide 49. Backup: the three spans, paired

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

## Slide 50. Backup: why the first intonation run was withdrawn

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

## Slide 51. Backup: calibration of the hidden element's band

A z-score is the error divided by the claimed uncertainty. Honest bands
put 90 percent of z-scores inside ±1.645 (the shaded band); "cov" above
each strip is the fraction that did. Intonation at chain and phrase
scale: 0.89 to 0.96, honest. Hidden joints in the boundary study: 1.00,
bands too wide, because the prior's spread was set from the track's
whole timing variability. Observed joints on clean audio: 0.05, 27 of
40 beyond ±8; the model claims 0.7 ms precision because it pins the
acoustic transition exactly, while the annotation marks the musical
onset about 20 ms away. The estimand gap of slide 38, at the boundary.

---

## If stopped on one slide

The two most likely probes are 35 (how the timbre is removed) and 49
(what the three spans say). One-line answers: "the amplitudes are
linear, so we integrate them out exactly," and "the score prior gives a
missing note its context's offset with an honest band, beats a
time-only prior at phrase scale, and so far nothing more."
