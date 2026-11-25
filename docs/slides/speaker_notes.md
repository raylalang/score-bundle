# Speaker's notes for `deck.tex` (the one living thesis deck)

Standing rule: **every change to `deck.tex` is followed by a change to
this file, in the same commit.** Frame numbers are PDF page numbers
(page 1 is the title page). The note for each slide states what the
slide says, defines every symbol on it, and gives an example. Written at
the simplest level that is still complete. Terms and symbols follow the
thesis (`docs/thesis/style_guide.md`, "Canonical terms"): NLL, never
"log score", hidden notes inside a fit, held-out pieces and tracks for
the test sets, the temporal chain, the time-only graph, the common
offset `c_0`, boundaries between notes.

Coverage: every slide, 2 to 47.

---

## Slide 2. Expressive performance

**What is on the slide.** Three sentences and one figure. The score
fixes which pitches and which note values. The performer adds when
exactly, how long, how loud, and on strings and winds how the pitch
moves inside the note. Those deviations are the expression. The figure:
the opening of Schumann's Kreisleriana No. 2, the printed score on top,
then a piano roll (pitch against score time in beats, grey outlines the
written notes, blue bars the played ones), then three strips, one per
variable, with the written value at zero and the played value as a dot:
timing `τ_i` in seconds, articulation `log r_i`, velocity `v_i`.

**How to say it.** "Everything in the three strips is zero on paper.
The dots are what the pianist did. That is what we model."

**Example.** Around beat 5 and beat 13 the timing dots drop to −0.5 s,
where the pianist held back before a chord. Articulation sits mostly
below zero, so notes were played shorter than written, detached.
Velocity drifts below zero in the middle, a quieter passage.

## Slide 3. The thesis, and the three problems

**What is on the slide.** The thesis sentence, to be said as written:
one Gaussian process on the score graph turns a known score and a
performance into per-note expressive variables, each with an error bar
you can trust. Then the three observation problems. Piano: variables
read directly from aligned MIDI, are the error bars calibrated note by
note? Strings and winds: six variables, now with intonation and vibrato,
estimated from audio with noise, does the same prior still work? Audio
directly: no pitch tracker at all, can the waveform be the observation?

**The four layers.** Between the thesis sentence and the three problems
sits the vocabulary of the whole thesis: a recording is read as score,
expression, timbre and noise. The score is what was written and is
given. The expression is how each written note was played, a small
vector of numbers per note (timing, articulation, dynamics, later
intonation, vibrato, loudness), and it is what we infer. The timbre is
what the instrument adds once the expression is fixed (the harmonic
amplitudes). In the waveform setting it is integrated out exactly
rather than estimated. The noise is the rest. Say it once here. Every
later slide is one of these layers being observed or removed.

**Symbols implied.** A Gaussian process (GP) here means that the vector
of all unknown per-note variables gets a joint Gaussian distribution
whose covariance is built from the score. "On the score graph" means
the covariance comes from a graph whose nodes are the written notes.
"Calibrated" means a 90 percent error bar contains the truth 90 percent
of the time, checked on hidden notes.

## Slide 4. One model of expression, three depths of measurement

**What is on the slide.** The chain figure. The performer's choices
(expression) produce the per-note variables (pitch curve, timing,
loudness), which produce the sound. The three settings read that chain
at three depths. Piano reads the variables directly from MIDI, strings
and winds measure them through a pitch tracker, and audio asks the
waveform itself. All three feed one GP on the score graph.

**How to say it.** "Same model, same prior. What changes is how far
from the performer's hands we are allowed to look."

## Slide 5. Related work

**What is on the slide.** Two columns. Left, expressive performance
modelling: the KTH rule system (Friberg 2006), linear basis-function
models (Grachten and Widmer 2012), deep rendering models such as
VirtuosoNet (Jeong 2019). All of them output one rendering or corpus
statistics, none a calibrated per-note posterior. Right, the tools:
Gaussian processes over graphs (Venkitaraman 2018), graph Matérn
kernels (Borovitskiy 2021), multi-output coupling by the intrinsic
coregionalization model (Bonilla 2008), differentiable synthesis
(Engel 2020). The closing line: this work combines them into a graph GP
over score positions with per-note side information, for calibrated
inference, down to the waveform.

**If asked about MIDI-DDSP** (Wu 2022, the closest relative). It
generates audio from notes, takes pitch from a tracker and boundaries
from annotations, and reads expression off with fixed formulas, with no
uncertainty. This thesis runs the other way, from recording to
expression, with calibrated bands.

## Slide 6. Divider: The model

One sentence: "The model is one multi-output Gaussian process on the
score graph. Three slides build it."

## Slide 7. The model (1): the score graph and its kernel

**What is on the slide.** The nodes are the `N` written notes, each
with pitch `p_i` (MIDI number), beat onset `b_i` (score time in beats)
and beat duration `d_i`. The edge weight between notes `i` and `j`:

`W_ij = exp(−(b_i − b_j)² / (2 ℓ_b²) − (p_i − p_j)² / (2 ℓ_p²))`,

and the graph Laplacian `L_G = D − W` with `D_ii = Σ_j W_ij`. `W`, `D`
and `L_G` are all `N × N`.

**Symbols.** `ℓ_b` is the time length scale, 2 beats, and `ℓ_p` the
pitch length scale, 4 semitones. Both are fixed. An edge weight is near
1 for two notes at the same time and pitch and decays as they separate
in either. `D` is diagonal, each entry the total weight at that node.
The Laplacian measures how much a function on the nodes differs from
its neighbours.

**Example.** Two notes one beat apart at the same pitch:
`exp(−1/8) ≈ 0.88`. Two notes at the same time an octave apart (12
semitones): `exp(−144/32) ≈ 0.01`, almost no edge. So neighbours are
close in time and in pitch at once.

**The figure.** Four beats of real score as nodes (pitch against score
time). Blue edges: the time and pitch weights, darker is stronger.
Green: same-chord edges. Vermilion: voice-leading steps. Those two
extra edge families form the chord graph, tested on the development set
(thesis, piano development appendix: redundant once the music model is
in the kernel). The model uses the blue edges only.

**The kernel.** Eigendecompose the Laplacian, `L_G = U diag(ν) Uᵀ`, with
`U` the `N × N` eigenvectors and `ν` the `N` eigenvalues, and shape the
spectrum with one parameter `s`: `K_G(s) = U g(ν; s) Uᵀ`,
`g(ν; s) = 1 / (1 + s ν)`. This is the graph Matérn kernel with `α = 1`.

**How to read it.** Each eigenvector of the Laplacian is a pattern over
the notes. A small eigenvalue means smooth over the graph (neighbours
alike), a large one means rough. `g` keeps the variance of smooth
patterns near 1 and shrinks rough ones by `1/(1 + s ν)`. Larger `s`
means a smoother prior. Example: with `s = 1`, a pattern with `ν = 0.1`
keeps 91 percent of its variance, one with `ν = 10` keeps 9 percent.

## Slide 8. The model (2): channels coupled, side information as kernels

**What is on the slide.** The channels (timing, articulation, velocity)
are correlated, so they share the graph through a learned `3 × 3`
coupling matrix `B`: `Cov[y_{i,c}, y_{j,c'}] = B_{cc'} [K_G(s)]_{ij}`,
in one line `Cov[y] = B ⊗ K_G(s)`.

**How to read it.** `⊗` is the Kronecker product: the covariance over
all channels of all notes is a `3N × 3N` matrix built by multiplying
each entry of `B` with the whole of `K_G`. Example: if `B` says timing
and articulation are positively correlated, then observing a late onset
at note 5 raises the posterior of a long articulation at note 6. That is
the sentence on the slide: an observed channel at one note informs the
other channels at neighbouring notes.

**The side information.** Two feature sets, each one row per note:
score features `X_feat` (`N × 26`: register, duration, metre, position
in bar, chord context, density) and music-model embeddings `X_emb`
(`N × 512`, rows `h_i`) from a small music model trained on MAESTRO. A
linear mean with unknown weights, integrated out, becomes a kernel
term: `y = X_f β + f_G`, `β ~ N(0, c_f I)` implies
`Cov[y] = c_f X_f X_fᵀ + Cov[f_G]`.

**Symbols.** `X_f` is either feature matrix, `f_G` the graph GP of
slide 7, `β` the regression weights, `c_f` the prior variance of those
weights, one number per channel, learned per piece. The weights are not
fitted once on the corpus. They are re-inferred for every piece
together with everything else, and their uncertainty enters the
posterior. This per-piece adaptation is the main source of the accuracy
gain (slide 13).

**How to say it.** "Instead of fitting a regression and plugging in its
prediction, we leave the regression weights unknown and let each piece
decide them. Mathematically that is just one more kernel term."

## Slide 9. The model (3): the covariance of one performance

**What is on the slide.** Stack all channels of all notes into one
vector `y ∈ R^{3N}`. Its covariance `K` is three coloured terms: blue
`B ⊗ K_G(s)` (the coupled graph GP), amber `Σ_f diag(c_f) ⊗ X_f X_fᵀ`
(side information, `f` over the two feature sets, `c_f ∈ R³` the
per-channel scales `c_feat` and `c_emb`), vermilion `diag(ς²) ⊗ I`
(noise, `ς² ∈ R³` the per-channel noise variances). All parameters
`θ = (s, B, {c_f}, ς²)`, 16 numbers on piano, are fitted by one exact
evidence per piece. Prediction is the conjugate Gaussian formula:
`m = C_{·o} K_oo⁻¹ ỹ_o`, `Σ_y = C − C_{·o} K_oo⁻¹ C_{o·}`.

**Why 16.** `s` (1) + `B` symmetric `3 × 3` (6) + `c_f` for two feature
sets times three channels (6) + `ς²` (3) = 16.

**Symbols in the prediction.** `o` indexes the observed entries of `y`,
`ỹ_o` their observed values, `K_oo` the observed block of `K` including
noise. `C` is `K` without the noise term, the covariance of the
noise-free field, and `C_{·o}` its columns at the observed entries. `m`
is the posterior mean for every entry, `Σ_y` the posterior covariance.
The evidence is the marginal likelihood of the observed values under
`K`, maximized over `θ`. Conjugate means a Gaussian prior plus Gaussian
noise gives a Gaussian posterior in closed form, with no approximation.

**The last sentence.** The baselines are exact special cases. A diagonal
`B` switches the coupling off, `c_f = 0` removes the features, and a
fixed cross-piece mean instead of per-piece `β` is the two-stage
pipeline.

## Slide 10. Divider: Piano

One sentence: "First setting: the variables are read exactly from
aligned MIDI, so this isolates the prior."

## Slide 11. Setup: aligned performance MIDI

**What is on the slide.** Data: ASAP, score-aligned Disklavier
performances. ASAP reuses MAESTRO recordings and the music model
pretrains on MAESTRO, so every performance seen in pretraining is
removed, 1036 performances down to 653. Development set 30 pieces,
held-out test set 20 pieces, 400 notes per piece (12,000 and 8,000
notes). The task: hide a fraction of the notes, predict them from the
rest, score the hidden ones. The main evaluation hides 40 percent, and a
sweep runs from 50 percent down to leave-one-out. The variables:
`y_i = [τ_i, log r_i, v_i]`. Metrics per channel and per level: RMSE
for accuracy, NLL and coverage for the error bars, all comparisons
paired per piece.

**The variables, exactly.** `τ_i` = performed onset minus the
tempo-implied onset, in seconds. The tempo-implied onset comes from a
beat-grid warp fitted to the whole performance, so `τ_i` is "early or
late relative to the local tempo". `log r_i` = log of performed duration
over the warp-implied duration. Zero means held exactly as written,
negative means detached. `v_i` = MIDI velocity, centred.

**The metrics, exactly.** RMSE: root mean squared error of the
posterior mean against the truth at hidden notes. NLL: the negative log
predictive density of the truth under the Gaussian predictive
distribution, averaged over hidden notes, lower is better. It punishes
both a wrong mean and a wrong band. Coverage at 0.90: the fraction of
hidden truths inside the 90 percent band. Calibrated bands give 0.90.

**Example.** `τ_i = −0.05`: the note came 50 ms early. `log r_i = −0.7`:
held for half its written length. `v_i = 0.1`: slightly louder than the
piece's average.

## Slide 12. The same hidden notes, with and without the graph

**What is on the slide.** The velocity channel of one piece, 40 hidden
notes. Orange circles: truth. Green: the model without the graph term
(RMSE 0.057). Blue: with the graph (RMSE 0.055), each with its band.
Both see the same observed notes. The per-piece margins are small and
consistent. The aggregate difference is the table (slide 13) and the
sweep (slide 14).

**How to say it.** "You will not see the graph's value on one piece by
eye. It shows up as a consistent small gain across all pieces, which is
why every comparison is paired per piece."

## Slide 13. Results (piano, ASAP): the held-out test set

**What is on the slide.** Evaluated once on 20 held-out pieces at 40
percent hidden, against the two-stage pipeline built from the same
ingredients, a regression mean first and then a per-channel graph
residual. RMSE pooled over the three channels: 0.393 for the pipeline,
0.376 for the proposed model, paired difference −0.014
[−0.0246, −0.0040]. Pooled NLL: −0.311 against −0.300, paired +0.011
[−0.129, +0.245], no difference. So the NLL half of the hypothesis is
not supported, and one steady piece with a few timing outliers dominates
it. Coverage at nominal 0.90: 0.922 against 0.925. Inside the model the
graph improves NLL by 0.074 [−0.0918, −0.0574]. The adoption rule fixed
in advance required the RMSE advantage, an NLL not significantly worse,
and the graph's contribution, and all three hold. A further 30 pieces,
not pre-specified, reproduce the graph (−0.0187 RMSE) and music-model
(−0.0102) contributions. Attribution: the features carry most of the
mean, the graph adds to recovery and most clearly to the NLL, and the
graph and music-model components are close to uncorrelated on average.

**If asked "so the hypothesis failed?"** Half of it. The RMSE half
holds, the NLL half shows no difference, and the adoption rule written
before the evaluation allowed exactly that case.

**What "evaluated once" means.** The hypotheses and the decision rule
were written down before the 20 pieces were used, the evaluation was run
once, and every outcome is reported.

**What "pooled RMSE" means.** The three channels are concatenated in
their own units, seconds for timing, a log ratio for articulation and
normalized velocity, so the pooled number is dominated by articulation,
the channel with the largest spread. The thesis reports each channel
separately.

## Slide 14. Across masking levels: 50 percent to leave-one-out (development)

**What is on the slide.** Development set. Left: RMSE at hidden notes
against the fraction of notes hidden, for the proposed model (blue), no
graph (amber dashed), features only (green dotted). Open markers at the
right are the leave-one-out limit, where every note is predicted from
all the others with hyperparameters fitted on the full piece. Right:
what each ingredient adds per level, as a paired difference in RMSE with
95 percent intervals, blue the graph, vermilion the music-model
embeddings. The full model is lowest at every level. The order of the
two ablations changes with the masking level.

**How to read the left panel.** Everything improves as fewer notes are
hidden, because there is more context. Features only flattens out below
30 percent: a cross-piece regression cannot use extra context, the
graph can.

**The component table, in the thesis.** For each piece the full model
and the model with one component removed are run on the same masks, and
`Δ = metric(full) − metric(removed)` is averaged over 30 pieces, so
negative means the component helps. Every entry is significant except
the graph at 10 percent (both metrics) and at 30 percent (NLL only), and
the music model at 50 percent and on timing. The graph's clearest
effect is on the NLL. The music model's value grows with observed
context. Fixing a cross-piece mean loses almost nothing, so the accuracy
gain is per-piece re-weighting of the features.

**Example.** Graph, NLL, 40 percent: −0.067. Removing the graph makes
the predictive density of the truth worse by 0.067 nats per note on
average.

## Slide 15. Divider: Strings and winds

One sentence: "Second setting: same prior, but the variables now come
from audio through a pitch tracker, with noise and a variance per note,
and six channels instead of three."

## Slide 16. Setup: audio instead of MIDI

**What is on the slide.** Data: URMP, 44 chamber pieces, duets to
quintets, 13 instrument types, each instrument recorded on its own
monophonic track with note annotations and the score. Split by
composition, because arrangements share recordings: development set 31
pieces (78 tracks, about 17,000 notes), held-out test set 13 pieces.
Pitch now carries expression, intonation and vibrato within every note.
A pitch tracker, pYIN, turns each track into frames: time, `f_0`,
voicing confidence. The variables must be estimated from these frames,
with noise and a variance per note. Two measurements of every note
exist: our tracked curve, and the corpus's own annotation of the same
track. The model is scored against the first. The second gives the
reference, which checks the ordering of systems and is the target of
the waveform studies.

**Why "split by composition".** URMP records each composition once and
offers it in several arrangements, so the same audio appears under
different piece names. Splitting by piece name would leak test audio
into development.

**Why score against our own estimates.** No intonation ground truth
exists. Recovery therefore means agreement with the estimator's targets
at hidden notes, a weaker kind of evidence than on piano, and the
thesis says so wherever the result appears.

## Slide 17. From the tracked frames to the pitch-curve channels

**What is on the slide.** The tracker's frames `(t_j, f_{0,j}, π_j)`:
time, fundamental frequency, voicing confidence. Frame rule: note `i`
keeps `J_i = { j : voiced, π_j ≥ q_0.2, t_j ∈ [t_i, t_i + δ_i) }`, at
least 4 frames, where `t_i, δ_i` are the note's annotated onset and
duration and `q_0.2` is the track's lowest confidence quintile. Cents
curve: each kept frame becomes
`c_i(t_j) = 1200 log₂(f_{0,j} / 440) − 100 (p_i − 69)`, the deviation in
cents from equal temperament at A4 = 440 Hz, `p_i` the MIDI pitch.

**Example.** A4 is MIDI 69. A frame at 442.5 Hz on an A4:
`1200 log₂(442.5/440) ≈ 9.8` cents, minus `100 × 0`, so 9.8 cents
sharp. On an A3 (MIDI 57) at 221 Hz: `1200 log₂(221/440) ≈ −1193`,
minus `100 × (57 − 69) = −1200`, giving +7 cents.

**The channels.** The annotated note in the figure: within one note, a
sustained level plus one oscillation. The level is the intonation
`c_i` (10.1 cents in the figure), the oscillation's half-height is the
vibrato extent `γ_i` (15.0 cents), and its period is `1/f_i^vib` with
`f_i^vib` the vibrato rate (6.1 Hz). The model receives `c_i`,
`log γ_i`, `log f_i^vib`. Extent and rate are positive, so they are
modelled on the log scale.

**How to say it.** "Three numbers describe how the pitch behaved inside
the note: where it sat, how wide it wobbled, how fast."

## Slide 18. Estimating the channels: one sinusoid fit per note

**What is on the slide.** The note's cents curve is written as a
constant centre plus a delayed sinusoid:
`c_i(t) = c_i + γ_i sin(2π f_i^vib (t − t_i − d_i^vib)) + ε_i(t)`, with
the sinusoid present only after the onset delay. The fit is a nonlinear
least-squares (NLLS) fit per note on the kept frames, and each
parameter's variance comes from linearizing the fit at the optimum.
Those variances enter the model as given, so the estimator's own
uncertainty becomes the noise row of each entry.

**Symbols.** `c_i`: the centre in cents, the intonation. `γ_i`: the
extent in cents. `f_i^vib`: the rate in Hz. `d_i^vib`: the onset delay
in seconds, how long after the note starts the vibrato begins. `t_i`:
the note's performed onset. `ε_i(t)`: the residual, tracker error plus
within-note movement the sinusoid does not describe (for example a slow
drift of the centre).

**Why the fitted constant and not the mean.** The mean of a sinusoid
over a window equals its centre only over a whole number of cycles. On
a short note with one and a half cycles the plain mean is pulled toward
the extra half cycle. The fitted constant is not.

**The identifiability rule.** Vibrato entries need at least 8 voiced
frames and 1.5 cycles at the fitted rate, otherwise they are missing for
the model. Intonation needs 4 frames and `|c_i| ≤ 150` cents, which
removes octave errors of the tracker. So missingness is per entry, not
per note: a short note can have intonation but no vibrato.

**Example.** The note of slide 17: the fit returns `c_i = 10.1` cents,
`γ_i = 15.0` cents, `f_i^vib = 6.1` Hz, each with a standard error. A
0.3-second note at 6 Hz has fewer than two cycles, so its vibrato
entries are missing and only its intonation, loudness and timing enter
the model.

**If asked whether a GP estimator was tried.** Yes, on the development
set: a two-component Gaussian-process fit of the curve (a vibrato band
plus a drift). It lost on extent and rate, so the sinusoid fit stayed.
The same two-component kernel reappears as the prior over the pitch
curve in the waveform setting (slide 24).

## Slide 19. Results (strings and winds, URMP): the held-out test set

**What is on the slide.** The same prior, unchanged. Six per-note
channels estimated from audio, each with its own variance, used as
given: `y_i = [c_i, log γ_i, log f_i^vib, ℓ_i, τ_i, d_i^vib]`,
intonation, vibrato extent, vibrato rate, loudness, timing, vibrato
onset delay. Pre-specified and evaluated once on 13 held-out pieces
(40 tracks). `Δ` = full model minus the same model without the graph
term, paired per track and seed, negative favours the graph. Intonation
recovery −0.877 cents, significant. Vibrato extent calibration −2.990
in NLL, significant. Vibrato rate calibration −0.564, significant.
Timing calibration −0.030, not significant, the secondary hypothesis,
not supported. Coverage 0.88 to 0.91 on all six channels at nominal
0.90.

**The three new channels.** `ℓ_i` is loudness, the mean log RMS of the
note's audio over four chunks, relative to the track. `τ_i` is timing,
the onset relative to a tempo line fitted through the neighbouring
annotated onsets with that note left out. `d_i^vib` is the vibrato
onset delay.

**If asked about timing.** "The timing hypothesis was not supported,
and we report it. Its diagnosis was the noise model, since alignment
error is correlated along the score. Correlated timing noise was then
tested on a disclosed second use of the same pieces and supported: the
next slide."

## Slide 20. Two follow-up hypotheses on the same test pieces

**What is on the slide.** Two hypotheses developed on the development
tracks after the first evaluation, each with its measures fixed in
advance, evaluated once on the same 13 held-out pieces as a disclosed
second use. The targets, masks and seeds are those of the first
evaluation. Two changes to the model, one at a time, each against the
model of slide 19.

**The timing-noise line.** `Σ_τ(ρ)` is the noise covariance of the
timing channel across the notes of one track in score order.
`σ_{e,iτ}²` is the variance the timing estimator gives note `i` (the
tempo-line fit with that note left out), used as given. `ρ` is the
correlation between the alignment errors of neighbouring notes: entry
`(i, j)` is the product of the two standard deviations times `ρ` to the
power of the note distance, so the correlation decays by a factor `ρ`
per note. `ρ` is chosen per track and seed from the grid 0, 0.15, up to
0.9 by the marginal likelihood of the observed notes alone. Hidden notes
are then predicted through the full covariance, prior plus noise, so an
observed neighbour's error informs the hidden note's. `ρ = 0` is the
diagonal model of slide 19. Example: on a track where the annotated
onsets drift late over a bar, with `ρ = 0.45` (the median found), a
hidden note between two late neighbours is predicted late too, and its
band narrows.

**The tonal-metric line.** `W^tonal_ij` replaces the semitone term of
the score graph. `d_5(i, j)` is the circle-of-fifths distance between
the two pitch classes (C to G is one step, C to C sharp is seven), with
length scale `ℓ_5`. The octave term, the pitch difference in octaves
with length scale `ℓ_oct`, keeps register locality, because `d_5` is
blind to it. The hypothesis: mistuning travels with harmonic
relatedness more than with semitone proximity. Example: a cellist's
slightly flat G and the D a fifth above it are neighbours under this
metric, the A flat a semitone up is far.

**The table.** Each row is one hypothesis: the paired difference, new
model minus the model of slide 19, over 80 track-seed pairs (40 tracks,
two mask seeds), with the 95 percent bootstrap interval. Negative
favours the new model. Timing calibration (NLL at hidden timing
entries): −0.121 [−0.156, −0.089], supported (development estimate
−0.111). Timing recovery: −0.011 seconds, 0.058 to 0.047, a 19 percent
cut, supported. Intonation calibration under the tonal metric: −0.023
[−0.049, +0.001], the mean on the predicted side but the interval
reaching zero, not supported (development estimate −0.050). Intonation
recovery: −0.312 cents [−0.599, −0.081], supported.

**The last line.** Coverage of the 90 percent bands stays at nominal
under correlated noise (0.908 against 0.911), and `ρ > 0` on 79 of 80
pairs. The tonal metric stays exploratory.

**If asked.** "Why use the test pieces again?" Because no other aligned
strings-and-winds corpus was available. The re-use is disclosed, and
evidence from a re-used test set is weaker than from a first use.
"Is the correlated noise model now the default?" No. Making it the
default would change the reported tables and needs its own evaluation.

## Slide 21. Divider: The waveform as the observation

One sentence: "Third setting: no tracker. The waveform itself is the
observation. Development studies, and one held-out test at phrase
scale."

## Slide 22. The method, the data and the signal model

**Where the method comes from.** DDSP (Engel 2020): gradients through a
differentiable synthesizer. This thesis takes that idea and keeps its
own probabilistic model: the exact likelihood with the amplitudes
integrated out, and GP priors, with no neural encoder and no spectral
surrogate loss. A learned many-component spectral prior was tried and
added nothing over the hand-structured two-component kernel.

**How to say it.** "We borrowed one engineering idea, gradients through
synthesis, and kept our own probabilistic model. The next slides are
that model."

**Data.** The same URMP development tracks as the strings-and-winds
chapter, 7 tracks, one instrument each. Each note is cut out of the
recording at its annotated start and end and resampled to 16,000
samples per second. 376 notes in the pitch-curve studies, 47 chains of
3 to 5 consecutive notes (154 boundaries) in the boundary study.

**The idea.** A sustained note is a stack of `H` harmonics: a wave at
the fundamental frequency, another at twice it, another at three
times, and so on. Example: a violin A at 440 Hz also contains waves at
880, 1320, 1760 Hz. `H = 8` in the studies.

**The equation and its symbols.** `x = Φ(z) a + ε`.
- `x`: the whole waveform of the note, one entry per audio sample, `M`
  entries. At 16,000 samples per second a 1.2-second note has
  `M = 19,200`.
- `z`: the pitch curve, the unknown.
- `Φ(z)`: an `M × n_a` matrix, one column per harmonic wave (a cosine
  and a sine per harmonic). The curve fixes the shape of every column.
- `a`: the `n_a` amplitudes, the timbre. They only scale the columns,
  like volume knobs.
- `ε`: the part of the recording the harmonic stack does not reproduce,
  modelled as independent Gaussian noise with variance `σ_x²` per
  sample.

**Sizes.** `n_a = 2HC`: a cosine and a sine per harmonic, repeated for
each of `C` time chunks so that loudness can change within the note.
With `H = 8` and `C = 2`, `n_a = 32`. Scoring one candidate curve costs
about `M n_a²` operations (slide 23), here `19,200 × 32² ≈ 20 million`,
milliseconds. `M` is large and `n_a` small, so thousands of samples pin
down a few dozen amplitudes precisely, and the pitch curve is what
remains to be found.

**If asked about the noise assumption.** The residual is not white: at
the harmonics it is strongly correlated across samples. The likelihood
therefore counts correlated samples as independent votes and is
overconfident, which is one reason the waveform measurement needs a
calibrated floor before it enters the bundle (slide 26).

## Slide 23. From the pitch curve to the likelihood: the timbre integrated out

**The first block.** From curve to design: the pitch curve
`z = cents(t)`, the frequency `f_0(t) = f_pitch · 2^(cents(t)/1200)`,
the phase `φ_k(t) = 2π k ∫_0^t f_0` for harmonics `k = 1..H`, and the
design `Φ(z) = [cos φ_k(t), sin φ_k(t)]`, an `M × n_a` matrix.

**Symbols.** `f_pitch`: the frequency the score says the note should
have, 440 Hz for A4. `cents(t)`: how far the played pitch deviates from
it at time `t`, 100 cents to a semitone. Vibrato is `cents(t)` wobbling.
`φ_k(t)`: the phase of harmonic `k`. Think of an odometer: frequency is
the speed, phase is the distance travelled. Phase accumulates, so a
small pitch error grows into a large phase error over time, and that is
what the method exploits.

**From `z` to `Φ(z)`, a toy example.** One sample per millisecond,
`f_pitch = 100` Hz, `cents = 0` throughout, so the phase advances
`2π · 100 · 0.001 = 0.628` rad (36°) per sample. Harmonic 1's cosine
column over four samples is `[0.81, 0.31, −0.31, −0.81]`, its sine
column `[0.59, 0.95, 0.95, 0.59]`. Raise the note one semitone
(`cents = +100`): `f_0 = 105.9` Hz, the step becomes 38.1°, and every
column changes shape. Let the cents wobble (vibrato): the step varies
from sample to sample, and the columns become chirped sinusoids.

**Why a cosine and a sine.** Two columns with two weights produce a
sinusoid of any amplitude and starting phase,
`A cos(φ − θ) = A cos θ · cos φ + A sin θ · sin φ`, so the unknown
starting phase of each harmonic is absorbed into `a` and the timbre
stays linear. That linearity is what the second block needs.

**The second block.** `a ~ N(0, σ_a² I)` implies
`p(x | z) = N(x; 0, σ_a² Φ(z) Φ(z)ᵀ + σ_x² I_M)`. We want `z` but do not
know `a`. Because `a` enters linearly, a Gaussian prior on it can be
integrated out exactly, and the result scores a candidate curve without
mentioning the timbre.

**The bullets.** One number per candidate curve, every audio sample
contributes, no pitch tracker. It is computed through the
`n_a × n_a` matrix `G = Φᵀ Φ + (σ_x²/σ_a²) I`, never the `M × M`
covariance, at cost `O(M n_a²)`. `σ_x²` is the residual variance at the
best amplitudes, fitted per note. The curve is held at knot values `u`
and interpolated to the sample grid.

## Slide 24. Finding the curve, with its uncertainty

**What is on the slide.** Three labelled lines: the prior on the curve
`u ~ N(0, K_knots)`, the best curve
`ẑ = argmax_z [log p(x|z) + log p(z)]`, its band
`Σ_z ≈ [−∇²_z (log p(x|z) + log p(z))]⁻¹` at `ẑ`. Then three bullets.

**The prior.** `u` are the knot values of the pitch curve and
`K_knots` their prior covariance, a two-component kernel on the knot
times: `k(Δt) = w_1 e^{−2π² λ_1 Δt²} cos(2π f̄ Δt) + w_2 e^{−2π² λ_2 Δt²}`.
The first component is a vibrato band around the rate `f̄` (Hz), the
second a slow drift around zero frequency. `w_1, w_2` are the powers
(cents²), `λ_1, λ_2` the bandwidths (Hz²). They are fitted by the
evidence and shared across notes. The prior turns "any curve" into
"curves a player would produce".

**Example.** `w_1 = 100` cents² and `f̄ = 6` Hz mean the prior expects
a vibrato of about 10 cents standard deviation at 6 Hz. `λ_1 = 0.5` Hz²
lets the rate wander by roughly 0.7 Hz.

**The other two bullets.** `Φ(z)` is differentiable in `z`, so the best
curve is found by gradient descent through the synthesizer, seconds per
note. The band is the Laplace approximation: `∇²` is the matrix of
second derivatives of the objective at the optimum, its inverse the
posterior covariance of the curve, and the 90 percent band on later
slides is 1.645 times the square root of its diagonal. A sharply curved
peak gives a narrow band.

## Slide 25. One note: the model against the raw audio

40 ms of a violin note with strong vibrato. Vermilion: the recorded
audio `x`. Grey dashed: the best single constant pitch. Blue: the
inferred pitch curve. The constant drifts out of phase within 40 ms
because vibrato changes the odometer's speed, while the curve stays in
phase. This is phase accumulation made visible, and why the method is
sensitive to small pitch deviations.

## Slide 26. The curve against the tracker, at scale (development)

376 notes, 7 tracks. Dumbbell plot: per instrument family, blue is the
waveform curve's median error against the annotation frames, green the
tracker's. All notes 3.16 against 4.09 cents. Winds 2.26 against 5.15,
a clear gain. Brass 3.15 against 3.51. Strings 4.90 against 3.09, where
the tracker is better.

**The limits.** Strings favour the tracker, and the waveform fit's
worst cases are worse (90th percentile 10.6 against 8.6 cents).

**The second bullet.** The inferred curve added to the strings-and-winds
bundle as a seventh channel improves intonation at hidden notes on 14
of 14 track-and-seed pairs, by 1.36 cents, significant, and the
calibrated floor halves, from 3.1 to 1.8 cents. The floor is the
per-note discrepancy between the waveform's constant and the
estimator's, the estimand gap: the two measure slightly different
quantities.

## Slide 27. Note boundaries: the chain model

**What is on the slide.** A chain of `n` consecutive notes with
boundaries `b_1 … b_{n−1}` and one pitch curve `z_k` per note. Five
labelled lines:
1. `w_k(t) = sig((t − b_{k−1})/t_w) · sig((b_k − t)/t_w)`, with
   `sig(u) = 1/(1 + e^{−u})` the logistic function and `t_w = 10 ms` the
   softness: note `k`'s window, a crossfade that rises after the
   boundary before it and falls at the boundary after it. The first note
   has no fade-in, the last no fade-out.
2. `Φ(z, b) = [w_1 ⊙ Φ_1(z_1) | … | w_n ⊙ Φ_n(z_n)]`, an
   `M × n n_a` matrix: each note's own design (slide 23), every column
   scaled sample by sample by that note's window (`⊙`), then all notes'
   blocks placed side by side.
3. `p(x | z, b) = N(x; 0, σ_a² Φ Φᵀ + σ_x² I_M)`: the same likelihood as
   slide 23, now a function of the boundaries too, because they shape
   the windows.
4. `b^(0)`: the starting boundaries come from the score through the
   local tempo line, not from the annotated onset of that boundary. The
   chains themselves are selected and cropped at annotated notes, so
   this is boundary refinement given a local alignment.
5. `(ẑ, b̂) = argmax [log p(x | z, b) + log p(z)]`: curves and
   boundaries found together by gradient descent, because the window is
   differentiable in `b`. Bands come from the Laplace approximation over
   `(z, b)`.

**How to say it.** "Each note is switched on and off by a soft window.
Where one note's harmonics stop explaining the samples and the next
note's start, the likelihood is highest, so the boundary settles there.
Curves and boundaries are fitted at once, from the score and the audio."

**Example.** Two notes, C then E. The tempo line puts the boundary at
0.60 s, but the player started the E at 0.63 s. At `b = 0.60` the C's
harmonics are asked to explain 30 ms of E, which they cannot, so the
score is poor. Sliding `b` to 0.63 lets each note explain its own
samples and the score rises, so the gradient moves `b` there.

**If asked.** A free crossfade width was tried. The data drove it to
the minimum (1 ms), which beat a fixed 10 ms width, so transitions are
effectively sharp to a harmonic likelihood. When two consecutive notes
have the same pitch the harmonic stacks are identical on both sides and
carry no boundary information, so those boundaries are excluded from
the numbers (slide 28).

## Slide 28. Note boundaries from the waveform: one chain, and the numbers

Panel A: a five-note violin passage with three kinds of vertical line.
Black: where the annotation says a note starts. Grey dashed: where the
score alone, through the local tempo line, predicts it (`b^(0)`). Blue:
where the fit put it (`b̂`). Panel B: the per-note pitch curves from the
same fit. The 90 percent band widens at the boundaries, where the audio
carries the least information about each note.

Numbers: starting from the tempo line, the joint fit improves 73
percent of boundaries and the median boundary error halves, 40 to
20 ms, over 128 boundaries. The remaining 20 ms is systematic: the
audio shows where the sound changes, the annotation marks where the
musician intended the note. A model of the attack transient at the
boundary is the next step (Future Work).

## Slide 29. The end goal on one passage (development)

A real 13-note, 8-second violin passage from the score and the
waveform, with chains cut at annotated notes. Panel A: the recording
segmented into notes by the inferred boundaries, annotated onsets drawn
only for reference. Panel B: each note's pitch expression with its
band, against annotation frames the fit never saw. Panel C: the timbre
strand, each note's harmonic envelope read off the amplitude posterior,
alternating colours for adjacent notes. The gap in the middle: notes
outside the chain-eligibility rule, left unfitted. Chains are fitted
independently and envelopes are read out after the fit, not inferred
under the prior. One posterior for a whole recording under the
score-graph prior is the joint loop, Future Work.

## Slide 30. The score prior inside the waveform inference: the setup

**What is on the slide.** The score graph of slide 7 brought inside the
waveform inference of slides 22 to 27, and tested at three spans. This
slide is the design, slide 31 the results.

**Symbols.** `θ`, an `n × 1` vector, one latent per element of a chain:
for the chain and phrase spans the per-note intonation constants `c`
(cents above the written pitch), for the boundary span the boundary
offsets `b` (milliseconds from the tempo line). `c_0`: the common
offset, one number shared by the whole chain (a tuning offset in cents,
or a shared timing offset in ms), with prior variance `V_0` (50² cents²
for intonation). `θ̃`: the per-element part on top of the common offset.
`V_ref`: the track's own variance of the reference centres (for the
boundary span, of its timing residuals), a population scale, not tuned,
the same in every arm. `K_arm`: the `n × n` correlation matrix of one
arm, the identity for the independent prior, and for the two structured
priors the inverse of `I + L_arm` rescaled so that its diagonal is all
ones. That rescaling is what "matched marginal variance" means: every
element has the same prior variance under every arm, only the
correlations differ. `L_arm`: the Laplacian of the temporal chain
(consecutive notes, unit weight) or of the score graph (the time and
pitch weights of slide 7). `K`: the full prior covariance of `θ`, the
arm's part plus the rank-one common-offset term. `i`: the hidden
element, `−i` all the others.

**How one case runs.** Take a chain. Remove the hidden element from the
audio and from the design matrix: its window columns go, not only its
samples, because its columns would otherwise pick up acoustic evidence
from the neighbouring samples. Fit the chain under the likelihood of
slide 23 with the prior above. The observed elements are pinned by the
audio, nearly identically under every arm. The hidden element has no
data, so given its neighbours it follows the prior conditional on the
slide: mean `K_{i,−i} K_{−i,−i}⁻¹ θ_{−i}`, a weighted average of the
neighbours with weights from the graph, and a variance reduced from the
prior by however much the neighbours tell. The neighbours are uncertain
themselves, so the band actually scored comes from the joint Laplace
approximation and is slightly wider. Score: the distance from the
posterior mean to the reference. Pair the three arms on the same case
and bootstrap the mean difference.

**Example.** A four-note violin chain, hidden third note, reference
centre +9 cents, chain mean +4 cents. Independent prior: the hidden note
gets the chain mean through `c_0`, +4. Temporal chain: its two
neighbours, say +3 and +5, so about +4 again. Score graph: the same two
neighbours and, if the first note has the same pitch, that one at higher
weight, say +8, so +5. All three fall short of +9 by a similar amount,
which is what the first row of slide 31 says.

## Slide 31. The score prior inside the waveform inference: three spans

**The table.** First row, 47 chains of three to five notes from four
development tracks: graph minus independent −0.02 cents, graph minus
temporal −0.19, both intervals across zero. Second row, 40 boundaries:
the same two contrasts in milliseconds, no difference. Third row, 48
phrases of six to ten notes from 30 of the 78 development tracks, the
hidden pitch recurring at least two notes away: graph minus temporal
−1.63 cents, interval excluding zero, graph minus independent −0.07, no
difference. Fourth row, 48 phrases on held-out tracks: graph minus
temporal −0.82 [−1.59, −0.05], interval excluding zero, graph minus
independent −0.13, no difference.

**The held-out row.** The phrase comparison was pre-specified and
evaluated once on the 13 strings-and-winds test pieces, their third
disclosed use and the first at the waveform level. The unchanged
selection rule gave 48 phrases from 20 of the 40 tracks, 14 on strings,
25 on winds, 9 on brass. Because the development effect came from
strings and this mix is winds-heavy, the expected difference was stated
in advance as about −0.9, with a power of only 0.3 to 0.5. The result,
−0.82, is better on 62 percent of phrases. Temporal minus independent is
+0.68, the same sign as development (+1.56) but with an interval
including zero, so on held-out tracks the graph's advantage is read
against the temporal prior alone. Clean rows are zero, coverage 0.94 to
0.98. By family: strings −1.55, winds −0.54, brass −0.44, none
significant alone.

**The sentence.** "Every prior gives a missing element its context's
common offset, worth about 7 cents and 10 to 14 ms. The score graph is
never significantly worse than the temporal chain and better at phrase
scale, on development and on held-out tracks. Beyond the common offset,
nothing measurable."

**If asked why the temporal chain hurts at phrase scale.** Adjacency
pulls the hidden note toward its immediate neighbours, which at phrase
scale resemble it less than the chain mean does. The pitch edge pulls
it back toward the chain mean. So the graph's advantage is the temporal
prior's harm undone, and the weak form holds while the strong form does
not.

## Slide 32. What in the graph does the work: the time window

**What is on the slide.** The same question at whole-track scale on
both corpora, and its answer: which part of the score graph carries its
value.

**Symbols.** `W_ij`, the edge weight of slide 7: `b_i, b_j` the onsets,
`p_i, p_j` the MIDI pitches, `ℓ_b = 2` and `ℓ_p = 4` semitones. On piano
the onsets are score beats, so the window is two beats. On strings and
winds the corpus annotates onsets in seconds and has no beat grid, so
the same `ℓ_b = 2` is two seconds of performance time. The time-only
graph: the same weight with `ℓ_p → ∞`, so only the time window remains.
The temporal chain: weight 1 between consecutive notes, 0 elsewhere.
Everything else in the GP is identical across the arms, including the
per-piece intercept column in the feature kernel, so every contrast is
beyond the common offset that dominated slide 31.

**The table.** Strings and winds, the development evaluation of slide
19 with 30 percent hidden, 153 track-and-seed pairs (148 for vibrato).
Score graph minus temporal chain on intonation: −0.50 cents, interval
excluding zero, so the graph beats the naive temporal prior. Score graph
minus time-only graph: −0.02, interval across zero, so removing the
pitch term changes nothing on intonation. On the two vibrato channels
removing it helps slightly (+0.04, +0.03), on loudness the pitch term
helps slightly (−0.007), the one place it shows. Piano: the proposed
model of slide 13 with the pitch term removed from its graph, paired
per piece on the 30 development pieces, 4 seeds. The score graph is
worse than the time-only graph by 0.008 in pooled RMSE and 0.019 in
NLL, both intervals excluding zero. A positive number in the last
column means the time-only graph is better.

**Example.** Two notes a beat apart, a C and a G (7 semitones). Score
graph: time factor exp(−1/8) ≈ 0.88, pitch factor exp(−49/32) ≈ 0.22,
weight 0.19. Time-only graph: weight 0.88. The time-only graph ties
these two notes more strongly, and the data say that is at least as
good.

**The sentence.** "On both corpora, what helps is smoothing over a
short time window, two beats on piano, two seconds on strings and winds.
Linking notes by pitch adds nothing on intonation." Then the qualifier,
always: "The held-out results stand, because they compare the full model
with its alternatives. What narrows is the attribution of the graph
term." If asked whether the thesis should switch to the time-only graph:
that would change a reported protocol and need its own evaluation.

## Slide 33. Divider: Motif structure

Say: "The attribution leaves the graph a modest role, a short time
window. This last section asks whether a structure the window cannot
see, the repetition of a motif, carries anything of its own.
Development studies first, then one pre-specified test."

## Slide 34. Motif edges: the family

**What is on the slide.** The one new edge family and its two forms.
The effect of a motif edge on a hidden note is the next slide.

**Symbols.** `O_k`: the `k`-th annotated occurrence, a set of notes
with a span in beats from `b_k^start` to `b_k^end` and a type
`type(k)`. Two occurrences of one type are returns of one motif. The
annotations are BPS-motif, covering all 32 Beethoven sonata first
movements. `u_{i,k}`: the relative position of note `i` inside
occurrence `k`, 0 at the start and 1 at the end, so returns of different
length share one coordinate (a note in two occurrences has a position in
each). `(W_motif)_ij`: a Gaussian in the difference of relative
positions, maximized over every pair of different occurrences of one
type that contain `i` and `j`, so a note keeps its strongest link.
`ℓ_u = 0.1`: a tenth of the span, fixed. Every pair of returns gets a
soft block of edges, strongest at equal relative position. No
self-edges, no alignment, nothing learned inside the family.

**Two forms.** Shared: the time-only graph `W_time` gains the family
with one weight `w_m`, learned on the log scale by the per-piece
evidence and shared by the three channels through `B`. Per channel: the
family is its own kernel term, `K_m` the shape-normalized kernel of the
motif Laplacian on the motif notes (zero elsewhere), with one scale
`c_m` per channel, so a channel can switch the family off on its own.
Because `K_m = X_m X_mᵀ`, it is a feature kernel of the model and
nothing in the inference changes.

**Example.** A four-note motif stated three times, once stretched to
twice the length. Its second note sits at the same relative position in
every statement, so the second notes of all three statements are joined
at full weight, and to the neighbouring positions more weakly.

## Slide 35. Motif edges: what they do to a hidden note

**The conditional.** In the single-channel GMRF with precision
`I + s L`, the mean of a hidden note `i` is a weighted average of its
neighbours and its variance is `1/(1 + s D_ii)`, with `D_ii` its
weighted degree. Split the degree by family and the mean splits into the
time-neighbour average and the motif-neighbour average, each weighted by
its share of the degree.

**Example.** A hidden note with time degree 2 and motif degree 1, at
`w_m = 1` and `s = 1`, puts weight one half on the mean of the notes
around it and one quarter on the mean of the corresponding notes in the
other returns. The remaining quarter goes to the prior mean of zero.
Its variance falls from `1/3` to `1/4` with the extra edge.

**Two readings.** If performers shape a motif alike at each return, the
motif average is closer to the truth than the time average, the
evidence raises `w_m`, and the error falls. At a fixed scale the
conditional variance falls with the extra degree (the band actually
scored also carries the neighbours' uncertainty and the noise). If not,
the motif average is noise, the evidence drives `w_m` toward zero, and
the arm approaches the time-only graph. So the study can come back
empty.

## Slide 36. Motif edges, development: whole occurrences hidden, 21 movements, 4 seeds

**Setup.** 21 Beethoven first movements from ASAP matched to the
annotations (25 annotated movements have an ASAP score, two are piano
test pieces and are excluded, two fall below the matching threshold),
400-note excerpts, score features without the music model so the
contrast is one score structure against another on identical features,
a per-piece intercept in every arm. The occurrence scheme hides whole
occurrences while at least one occurrence of each type stays out of the
draw. A hidden note is scored only when another occurrence of its type
has an observed note. Four seeds, 672 fits over both masking schemes,
the table is the occurrence scheme. Paired per movement, 95 percent
percentile bootstrap, negative favours the motif arm.

**The table.** Shared form: timing NLL −0.071 with the interval reaching
+0.097, timing error −0.0027 s with the interval excluding zero,
articulation RMSE −0.035 with the interval excluding zero, velocity NLL
+0.033 with the interval reaching zero and coverage 0.835, below
nominal: one weight for three channels ties the velocities of returns
more tightly than performers play them. Per-channel form: timing
nothing (−0.005), articulation −0.019 with the interval excluding zero,
velocity −0.086 with the interval excluding zero and coverage 0.944. So
the per-channel form removes the velocity penalty.

**The three bullets.** First: on two seeds alone the timing NLL
difference is −0.117 with an interval excluding zero, on the other two
−0.026 with a wide interval, so the timing effect does not hold on four
seeds. Second: the learned per-channel scales show where the family
matters. On timing the scale is below 0.01 on 19 of 21 movements,
switched off, because the time-only graph is already well calibrated
there. On articulation it exceeds 0.1 on 10 movements, on velocity
0.01 on 7. Third: the one effect that holds on every pair of seeds and
in both forms is articulation accuracy, which is why the pre-specified
test takes it as its primary measure. Under random masking the time
window already interpolates inside an occurrence and the family has
little to act on.

## Slide 37. Motif edges from a discovery model's predictions (development)

**What is on the slide.** The same evaluation with the annotations
replaced, as the edge source only, by the occurrences a motif-discovery
model predicts on each movement held out from its training fold.

**The model, in one pass.** Notes are vertices of a typed graph
(simultaneity, temporal proximity, overlap). A pretrained encoder gives
each note a vector `h_i`, which relational graph convolutions update.
Two heads: `m_i`, the probability that note `i` belongs to any motif,
and `â_ij`, an affinity on candidate pairs. Training is label
propagation: the annotated occurrences seed a membership matrix
`M^seed`, memberships `M` diffuse along the learned affinities with
restart weight `α`, and the losses act on `M`. Decoding gates the
affinity by the memberships, thresholds it at the quantile 0.86, takes
connected components as occurrences and clusters them into types.
Lalang and Yoshii, ISMIR 2026. (Here `h_i`, `M`, `α` and `s_self` are
the discovery model's own symbols, not the thesis model's.)

**What changes.** Predicted occurrences are mapped to the excerpt by
onset and pitch exactly as the annotations were. The masks and the
scored notes stay those of the annotated occurrences, identical to
slide 36, so the only change is where the edges come from. Membership
against the annotation: precision 0.12 to 1.00, recall 0.44 to 0.99, F1
median 0.78. On two movements the model predicts several times the
annotated number of motif notes.

**The table.** Shared form, predicted edges: timing NLL −0.034, 0.48 of
the annotated −0.071, interval reaching +0.003. Articulation RMSE
−0.0135, four tenths of −0.035, interval reaching +0.003. Velocity
coverage 0.865. The two edge sources act on related movements:
per-movement effects correlate at 0.59 on timing and 0.84 on
articulation, and 12 of 21 movements are below zero for both. The
prediction stated in advance named the per-channel form, half the
annotated timing effect with an interval clear of zero. In that form
the predicted edges move the timing NLL by +0.036 [−0.010, +0.095], the
wrong direction, and cost 0.0014 s of timing error, so the prediction
is not supported. The sentence: in the shared form a model's
occurrences give point estimates of about half the timing effect and
four tenths of the articulation effect, with intervals that include
zero. In the per-channel form they give nothing.

## Slide 38. The pre-specified test: held-out performances of the same movements

**What is on the slide.** The design of the one motif test, written
down before it was run, and the power behind it.

**Held-out material.** For each of the 21 development movements, the
other aligned ASAP performances, at most four in sorted order, none used
in development. 18 movements have at least one (three have none), 58
performances in all. Scores, annotations, the masking rule with its four
seeds, and the scoring rule are the development study's. Each
performance has its own aligned excerpt, so its edges and masks are
rebuilt by the same rules, and it supplies new targets. So the test asks
whether the development finding holds across performers of the same
pieces, not across pieces, and the thesis says so wherever it appears.
The movement is the unit of analysis, with performances averaged inside
it before pairing, so no movement counts four times.

**Hypotheses.** Primary: articulation RMSE on the hidden motif notes,
shared form minus the time-only graph, negative with the 95 percent
interval excluding zero. Secondary: the timing NLL, same rule.
Descriptive: velocity NLL and coverage, timing RMSE, and the
per-channel form beside the shared one. The shared form was chosen
because its power is the higher of the two (0.95 against 0.93) and its
articulation effect the larger and more even across movements.

**The power table.** Draw 18 of the 21 development movements'
differences with replacement many times and ask how often the interval
excludes zero: articulation RMSE 0.95 at the development effect and 0.51
at two thirds of it, timing NLL 0.33 at the development effect and below
0.2 at two thirds. So articulation is the well-powered hypothesis, and
timing is included because it was the first study's hypothesis, with its
weak power stated.

## Slide 39. The pre-specified test: the result

**The result.** 696 fits. Primary, supported: the articulation error of
the hidden returns falls by 0.029 with interval [−0.049, −0.011], below
zero on 15 of the 18 movements, against 0.035 on development. The
per-movement effects of the two studies correlate at 0.88, and 12
movements are below zero in both. The interval is for the mean over
movements, no single movement is tested on its own. Secondary, not
supported: timing NLL −0.018 [−0.191, +0.225], as the power table said
was likely. Descriptive: the shared form's timing error falls by 3.7 ms
with the interval excluding zero, its velocity NLL shows no penalty
(−0.016, interval covering zero), though velocity coverage (0.847) is
still below nominal, and the per-channel form reproduces its
development picture (articulation NLL −0.051 and error −0.023, velocity
NLL −0.085 with coverage 0.934, nothing on timing). The learned weights
agree: the shared weight exceeds 0.1 on 37 of 58 performances, the
per-channel scale exceeds 0.1 on articulation on 26 and on timing or
velocity on none.

**How to say it, in musical terms.** "If a pianist plays a motif three
times and we hide one statement, knowing how they played the other two
predicts how long they hold each note of the hidden one, about 4 percent
better than knowing only the surrounding notes (error 0.622 to 0.598).
That holds for performers the model was not developed on, of the same
movements. It does not show for timing."

**If asked what 0.029 means in units.** Articulation is the log of
played over written duration, so an RMSE of 0.6 means a typical error of
a factor of about 1.8 in held duration. The motif edges shrink that
error by about 4 percent on the hidden returns.

## Slide 40. The motif result on one hidden statement

**What is on the slide.** One real case from the held-out test:
Beethoven's Sonata No. 17, first movement, a held-out performance, one
statement of a six-note motif hidden from the model. The horizontal axis
is the position inside the statement, 0 at its first note. The vertical
axis is how long each note was held, as a fraction of its written
length (1 = as written, 0.5 = half, log scale). Grey lines: the same
pianist's other statements of this motif in the same performance, which
the model sees. Black diamonds: what the pianist actually played in the
hidden statement. Amber squares: the prediction from the time window
alone. Blue circles: the prediction with motif edges. Bars are 90
percent predictive intervals.

**How to read it.** The grey lines share a zig-zag: long, short, long,
short across the statement. That is the pianist's way of articulating
this motif. The hidden statement (black) follows the same zig-zag. The
time window alone only sees the notes around the hidden statement, so
its predictions are flatter and its bars wide. The motif edges let each
hidden note borrow from the notes at the same position in the other
statements, so the blue predictions lean toward the zig-zag and their
bars are narrower. Here the error falls from 0.35 to 0.31.

**How to say it.** "We hide one statement of a motif and ask the model
how long each note was held. Knowing how the pianist played the other
statements helps: the prediction follows the pattern, and it is more
confident."

**If asked whether this example is cherry-picked.** It is chosen by a
stated rule: of 1001 fully hidden statements in the held-out test (every
performance and seed, at least five notes), the one at the upper
quartile of the improvement. So one statement in four improves at least
this much, 67 percent improve at all, and the rest do not. The average
over movements is the next slide. The figure is drawn from the stored
records of the test, with no new fit (`scripts/make_motif_demo_figs.py`).

## Slide 41. The motif result per movement

**What is on the slide.** Panel A: one row per movement (sonata number,
first movement), the articulation error of the hidden returns on the
held-out performances, amber with the time window only, blue with motif
edges, rows sorted by the change. In 15 of 18 movements the blue dot is
to the left (lower error). In the other three the error rises by at
most 0.03.
The mean change is −0.029 [−0.049, −0.011], about 4 percent. Panel B:
one dot per movement, the change on the development performance
(horizontal) against the change on the held-out performances (vertical).
The dots lie along the diagonal (r = 0.88), and 12 movements are better
in both.

**How to say it.** "This is not one lucky example. Most movements gain,
and the movements that gained when we developed the method are the same
ones that gain on new performances. The effect belongs to the music,
not to one recording."

**The limit, said every time.** The held-out performances are of the
same movements, so this shows the effect holds across performers, not
yet across new pieces. And it is articulation only. Timing shows
nothing.

## Slide 42. Summary

Five sentences, say them as written. Piano: more accurate than the
two-stage pipeline, and the graph improves calibration inside the
model, on 20 held-out pieces. Strings and winds: the graph improves
recovery and calibration on the estimated channels, on 13 held-out
pieces. Audio: curves, boundaries and timbre from the waveform at
development level, and the score graph beats the temporal chain at
phrase scale, also on held-out tracks, with the joint loop ahead. What
the graph does: smoothing over a short time window, two beats on piano
and two seconds on strings and winds, with pitch proximity adding
nothing on intonation (development). Motif structure: one edge family
outside the time window, which lowers the articulation error of hidden
returns in both forms, leaves the timing NLL unchanged, keeps four
tenths of the articulation gain with predicted occurrences (interval
including zero), and holds on further performances (−0.029).

## Slide 43. Plan

Three rows, read top down. October: the waveform studies (curves, the
bundle channel, boundaries, the score prior at three spans) and the
attribution of the graph's value to its time window. Two follow-up
hypotheses on the strings-and-winds test pieces as a disclosed second
use: correlated timing noise supported (−0.121 calibration, −0.011 s
recovery), the tonal metric not supported on its primary measure
(−0.023) while keeping its recovery gain (−0.312 cents). November: the
motif family's two development studies and the predicted-occurrence
study, then two pre-specified held-out tests, the phrase-scale waveform
comparison (−0.82 cents [−1.59, −0.05]) and the motif test on further
performances (−0.029 [−0.049, −0.011]), both supported on their primary
measure. Every result of the thesis is in, and the thesis was audited
and cut from 92 to 66 pages. Next: the lab seminar on 3 December,
December for writing only with no new experiments, the progress talk on
7 January from this deck, submission on 2 February. The transient model
at the boundaries and the whole-performance joint loop stay Future
Work.

## Slide 44. Divider: Backup

Shown only if asked.

## Slide 45. Backup: the curve from the waveform, three notes

Three development notes: violin, cello, clarinet. Blue: the curve from
the waveform. Orange dots: the corpus's annotation frames, never seen by
the fit. Green dots: the tracker's frames (pYIN), also scored against
the annotation. Legend numbers are the median distance from the
annotation in cents. Example: clarinet, 1.0 cents for the waveform
curve against 4.5 for the tracker.

The caption's point: the 90 percent band around the blue curve is about
half a cent wide, hidden under the line. It is a band for the curve
itself. The annotation frames come from their own measurement chain,
offset by 2 to 4 cents, so the band does not cover them. That gap, the
estimand gap, is carried separately as a calibrated floor wherever the
curve is used downstream.

## Slide 46. Backup: the three spans, paired

Three panels, one per span. Each row: the mean paired difference in the
hidden element's error between two priors, with a 95 percent bootstrap
interval. Negative: the first-named prior is better.
- Left, intonation on 47 chains (cents): score graph minus independent
  −0.02, no difference.
- Middle, boundaries on 40 chains (ms): −2.45, no difference.
- Right, intonation on 48 phrases where the hidden pitch recurs two or
  more notes away: score graph minus temporal −1.63, significant,
  temporal minus independent +1.56, significant (the temporal prior
  hurts), score graph minus independent −0.07, no difference.
Third bullet: the common offset alone, which every prior gets through
`c_0`, is worth about 7 cents on intonation and 10 to 14 ms on
boundaries against the references without any transfer.

## Slide 47. Backup: the three priors at track scale (strings and winds)

**What is on the slide.** The track-scale version of the three-prior
question on the tracked six-channel bundle: 78 development tracks, 153
track-and-seed pairs (148 for vibrato), 30 percent of notes hidden per
track. Every arm, including the no-graph ablation, carries the
per-piece intercept (a column of ones in the feature kernel), so every
contrast is already beyond the common offset that dominated the
chain-scale waveform tests. Three columns of paired differences,
negative favouring the first-named arm.

**The three arms.** Score graph: notes linked by closeness in time
(2-second window, onsets annotated in seconds) and in pitch (4-semitone
window). Temporal chain: consecutive notes linked with unit weight,
nothing else. Time-only graph: the score graph with the pitch term
removed, the same 2-second window, the score graph minus exactly the
thing that makes it a score graph.

**How to read the table.** First column: the score graph beats the
temporal chain, significantly, on intonation, vibrato extent, vibrato
rate and loudness. Second column: the score graph equals the time-only
graph on intonation (−0.02 cents, interval across zero) and is slightly
worse on the two vibrato channels. Third column: the time-only graph
against the chain reproduces the first column. So the whole advantage
over the chain is the time window, not pitch proximity.

**The piano bullet.** The proposed model of slide 13 with the pitch term
removed from its graph (the 2-beat window kept, which on piano links
the notes of a chord at full weight), paired per piece on the 30
development pieces, 4 seeds. Pooled RMSE −0.008 [−0.016, −0.001], NLL
−0.019 [−0.030, −0.008]: removing the pitch term helps slightly, mostly
in articulation. The graph's whole value on that set is +0.017 RMSE, so
the time-only graph is worth about half again as much as the score
graph.

**What it does not change.** The held-out piano result (slide 13) and
the held-out strings-and-winds results (slide 19) compare the full
model with its alternatives, and they stand. What is narrower is the
attribution: the graph term works through temporal smoothing, not
through pitch coupling.

---

## If stopped on one slide

The four most likely questions are on slides 20 (why use the test
pieces again), 23 (how the timbre is removed), 31 (what the three spans
say) and 32 (what the graph is for). One-line answers: "a disclosed
second use, weaker evidence than a first use, one hypothesis supported
and one not, both reported," "the amplitudes are linear, so we integrate
them out exactly," "the score prior gives a missing note its context's
common offset, beats the temporal chain at phrase scale, and nothing
more so far," and "the graph's value is its short time window, two
beats on piano and two seconds on strings and winds, and the pitch term
adds nothing on intonation."
