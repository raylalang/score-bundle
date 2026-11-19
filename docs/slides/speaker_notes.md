# Speaker's notes for `deck.tex` (the one living thesis deck)

Standing rule (Ray, 2026-10-21): **every change to `deck.tex` is followed
by a change to this file.** Frame numbers are PDF page numbers (page 1
is the title page). The note for each slide states what the slide says,
defines every symbol on it, and gives an example. Written at the
simplest level that is still complete.

Coverage: every slide, 2 to 45 (the motif section, slides 33 to 39, was added
on 2026-11-18; the deck was cut from 56 to 37 pages on 2026-10-28, Ray's direction: findings only; the removed frames and
their notes are in git history, commit 9fe8778). (`docs/zemi_prep.md` holds the older
notes for the 28-slide zemi deck; its frame numbers do not match this
deck.)

---

## Slide 2. Expressive performance


**What is on the slide.** Three sentences and one figure. The score
fixes which pitches and which note values; the performer adds when
exactly, how long, how loud, and on strings and winds how the pitch
moves inside the note. Those deviations are the expression. The figure:
the opening of Schumann's Kreisleriana No. 2, the printed score on top,
then a piano roll (pitch against score time in beats; grey outlines are
the written notes, blue bars the played ones), then three strips, one
per variable, with the written value at zero and the played value as a
dot: timing `τ_i` in seconds, articulation `log r_i`, velocity `v_i`.

**How to say it.** "Everything in the three strips is zero on paper.
The dots are what the pianist did. That is what we model."

**Example.** Around beat 5 and beat 13 the timing dots drop to −0.5 s:
the pianist held back before a chord. Articulation sits mostly below
zero: notes played shorter than written, detached. Velocity drifts
below zero in the middle: a quieter passage.

## Slide 3. The thesis, and the three problems


**What is on the slide.** The thesis sentence, to be said verbatim:
one Gaussian process on the score graph turns a known score and a
performance into per-note expressive variables, each with an error bar
you can trust. Then the three observation problems: piano (variables
read directly from aligned MIDI; are the error bars calibrated note by
note?), strings and winds (six variables, now intonation and vibrato
too, estimated from audio, noisily; does the same prior still work?),
audio directly (no pitch tracker at all; can the waveform be the
observation?).

**The four layers.** Between the thesis sentence and the three problems
sits the vocabulary of the whole thesis, one line: a recording is read
as score, expression, timbre and noise. The score is what was written
and is given. The expression is how each written note was played, a
small vector of numbers per note (timing, articulation, dynamics; later
intonation, vibrato, loudness); it is the thing we infer. The timbre is
what the instrument adds to a note once its expression is fixed (the
harmonic amplitudes); in the waveform setting it is integrated out
exactly rather than estimated. The noise is the rest. Say it once here;
every later slide is one of these layers being observed or removed.

**Symbols implied.** A Gaussian process (GP) here means: the vector of
all the unknown per-note variables is given a joint Gaussian
distribution whose covariance is built from the score. "On the score
graph" means the covariance comes from a graph whose nodes are the
written notes. "Calibrated" means a 90 percent error bar contains the
truth 90 percent of the time, checked on held-out notes.

## Slide 4. One model of expression, three depths of measurement


**What is on the slide.** The chain figure: the performer's choices
(expression) produce the per-note variables (pitch curve, timing,
loudness), which produce the sound. The three settings read that chain
at three depths: piano reads the variables directly from MIDI; strings
and winds measure them through a pitch tracker; audio asks the
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

**If asked about MIDI-DDSP** (Wu 2022, the closest cousin, on slide 22): it generates audio from notes, takes pitch from a tracker and
boundaries from annotations, and reads expression off with fixed
formulas, with no uncertainty; this thesis runs the other way, from
recording to expression, with calibrated bands.

## Slide 6. Divider: The model


One sentence: "The model is one multi-output Gaussian process on the
score graph. Three slides build it."

## Slide 7. The model (1): the score graph and its kernel


**What is on the slide.** The nodes are the `N` written notes, each
with pitch `p_i` (MIDI number), beat onset `b_i` (score time in beats)
and beat duration `d_i`. The edge weight between notes `i` and `j`:

`W_ij = exp(−(b_i − b_j)² / (2 ℓ_b²) − (p_i − p_j)² / (2 ℓ_p²))`,

and the graph Laplacian `L_G = D − W` with `D_ii = Σ_j W_ij`. `W`,
`D`, `L_G` are all `N × N`.

**Symbols.** `ℓ_b` is the time length-scale, 2 beats; `ℓ_p` the pitch
length-scale, 4 semitones. Both fixed, global. An edge weight is near
1 for two notes at the same time and pitch and decays as they separate
in either. `D` is diagonal, each entry the total weight at that node;
the Laplacian measures how much a function on the nodes differs from
its neighbours.

**Example.** Two notes one beat apart at the same pitch:
`exp(−1/8) ≈ 0.88`. Two notes at the same time an octave apart (12
semitones): `exp(−144/32) ≈ 0.01`, almost no edge. So "neighbours" are
close in time and in pitch at once.

**The figure.** Four beats of real score as nodes (pitch against score
time). Blue edges: the time and pitch proximity weights, darker is
stronger. Green: same-chord edges; vermilion: voice-leading steps.
Those two extra edge families are the chord-graph variant, tested in
development (thesis, piano development appendix: redundant once the
music model is in the kernel); the base graph uses the blue edges only.

**The kernel.** 
Eigendecompose the
Laplacian, `L_G = U diag(ν) Uᵀ`, with `U` the `N × N` eigenvectors and
`ν` the `N` eigenvalues, and shape the spectrum with one parameter `s`:
`K_G(s) = U g(ν; s) Uᵀ`, `g(ν; s) = 1 / (1 + s ν)`. This is the graph
Matérn kernel with `α = 1`.

**How to read the first equation.** Each eigenvector of the Laplacian
is a pattern over the notes; small eigenvalue means smooth over the
graph (neighbours alike), large means rough. `g` keeps the variance of
smooth patterns near 1 and shrinks rough ones by `1/(1 + s ν)`. Larger
`s` means a smoother prior. Example: with `s = 1`, a pattern with
`ν = 0.1` keeps 91 percent of its variance, one with `ν = 10` keeps
9 percent.

## Slide 8. The model (2): channels coupled, side information as kernels

**What is on the slide.** The channels (timing, articulation,
velocity) are correlated, so they share the graph through a learned
`C × C` coupling matrix `B`: `Cov[y_{i,c}, y_{j,c'}] = B_{cc'} [K_G(s)]_{ij}`,
in one line `Cov[y] = B ⊗ K_G(s)`.

**How to read the second.** `⊗` is the Kronecker product: the full
covariance over all channels of all notes is a `CN × CN` matrix built
by multiplying each entry of `B` with the whole of `K_G`. Example: if
`B` says timing and articulation are positively correlated, then
observing a late onset at note 5 raises the posterior of a long
articulation at note 6. That is the sentence on the slide: an observed
channel at one note informs the other channels at neighbouring notes.


**The side information.** Two feature sets, each one row per note:
score descriptors `X_feat` (`N × 26`: register, duration, metre,
position in bar, chord context, density) and `H` (`N × 512`), per-note
features from a small music model trained on MAESTRO. A linear mean
with unknown weights, integrated out, becomes a kernel term:
`y = X β + f_G`, `β ~ N(0, c_X I)` implies
`Cov[y] = c_X X Xᵀ + Cov[f_G]`.

**Symbols.** `X` is either feature matrix; `f_G` is the graph GP of
slide 7; `β` the regression weights, `d` per feature set; `c_X` the
prior variance of those weights, one number per channel, learned per
piece. The two bullets: the weights are not fitted once on the corpus,
they are re-inferred for every piece together with everything else,
and their uncertainty enters the posterior. This per-piece adaptation
is the main source of the accuracy gain (slide 13).

**How to say it.** "Instead of fitting a regression and plugging in its
prediction, we leave the regression weights unknown and let each piece
decide them. Mathematically that is just one more kernel term."

## Slide 9. The model (3): the covariance of one performance


**What is on the slide.** Stack all channels of all notes into one
vector `y ∈ R^{CN}`. Its covariance `K` is three coloured terms:
blue `B ⊗ K_G(s)` (the coupled graph GP), amber
`Σ_f diag(c_f) ⊗ X_f X_fᵀ` (side information, `f` over the two feature
sets, `c_f ∈ R^C` the per-channel scales), vermilion `diag(ς²) ⊗ I`
(noise, `ς² ∈ R^C` the per-channel noise variances). All parameters
`θ = (s, B, {c_f}, ς²)`, 16 numbers on piano, are fitted by one exact
evidence per piece. Prediction is the conjugate Gaussian formula:
`m = C_{·o} K_oo⁻¹ ỹ_o`, `Σ_y = C − C_{·o} K_oo⁻¹ C_{o·}`.

**Why 16.** `s` (1) + `B` symmetric `3 × 3` (6) + `c_f` for two feature
sets times three channels (6) + `ς²` (3) = 16.

**Symbols in the prediction.** `o` indexes the observed entries of
`y`; `ỹ_o` their observed values; `K_oo` the observed block of `K`,
including noise; `C` is `K` without the noise term, the covariance of
the noise-free field; `C_{·o}` its columns at the observed entries.
`m` is the posterior mean for every entry, `Σ_y` the posterior
covariance. "Evidence" is the marginal likelihood of the observed
values under `K`, maximized over `θ`. "Conjugate" means Gaussian prior
plus Gaussian noise gives a Gaussian posterior in closed form, no
approximation.

**The last sentence.** The baselines are exact special cases: diagonal
`B` switches the coupling off; `c_f = 0` removes the features; a fixed
cross-piece mean instead of per-piece `β` is the two-stage pipeline.

## Slide 10. Divider: Piano


One sentence: "First setting: the variables are read exactly from
aligned MIDI, so this isolates the prior."

## Slide 11. Setup: aligned performance MIDI


**What is on the slide.** Data: ASAP, score-aligned Disklavier
performances. ASAP reuses MAESTRO recordings and the music model
pretrains on MAESTRO, so every performance seen in pretraining is
removed, 1036 performances down to 653. Development set 30 pieces,
one-shot test set 20 untouched pieces, 400 notes per piece (12,000 and
8,000 notes). The task: hide a fraction of the notes, predict them from
the rest, score the hidden ones. Headline operating point 40 percent
hidden; the sweep runs from 50 percent down to leave-one-out. The
variables: `y_i = [τ_i, log r_i, v_i]`. Metrics per channel and per
level: RMSE for accuracy, log score and coverage for the error bars,
all comparisons paired per piece.

**The variables, exactly.** `τ_i` = performed onset minus the
tempo-implied onset, in seconds; the tempo-implied onset comes from a
beat-grid warp fitted to the whole performance, so `τ_i` is "early or
late relative to the local tempo". `log r_i` = log of performed
duration over the warp-implied duration; zero means held exactly as
written, negative means detached. `v_i` = MIDI velocity, centred.

**The metrics, exactly.** RMSE: root mean squared error of the
posterior mean against the truth on hidden notes. Log score: the
negative log predictive density of the truth under the Gaussian
posterior, averaged over hidden notes, lower is better; it punishes
both a wrong mean and a wrong band. Coverage at 0.90: the fraction of
hidden truths inside the 90 percent band; honest bands give 0.90.

**Example.** `τ_i = −0.05`: the note came 50 ms early. `log r_i = −0.7`:
held for half its written length. `v_i = 0.1`: slightly louder than the
piece's average.

## Slide 12. The same hidden notes, with and without the graph


**What is on the slide.** The velocity channel of one piece, 40 hidden
notes. Orange circles: truth. Green: the model without the graph term
(RMSE 0.057). Blue: with the graph (RMSE 0.055), each with its band.
Both see the same observed notes. The per-piece margins are small and
consistent; the aggregate difference is the table (slide 13) and the
sweep (slide 14).

**How to say it.** "You will not see the graph's value on one piece by
eye. It shows up as a consistent small gain across all pieces, which is
why every comparison is paired per piece."

## Slide 13. Results (piano, ASAP): confirmed once on untouched pieces


**What is on the slide.** Preregistered, scored once on 20 untouched
pieces at 40 percent hidden. Baseline: the two-stage pipeline built
from the same ingredients, a regression mean first and then a
per-channel graph residual. RMSE pooled over the three channels: 0.393
baseline, 0.376 this model; coverage at nominal 0.90: 0.925. The RMSE
difference is paired per piece and significant; removing the graph
term costs log score, significantly. Attribution, measured: the
per-piece feature weighting carries the mean, the graph carries the
calibration, and the two are near-orthogonal complements.

**What "preregistered, once" means.** The claims and the test rule were
written down before the 20 pieces were touched; the test was run once;
whatever came out was reported. The pool is spent and is never rerun.

**What "pooled RMSE" means.** The three channels are standardized so
the numbers can be averaged; 0.376 is in those standardized units, not
seconds.

## Slide 14. Across masking levels: 50 percent to leave-one-out (development)


**What is on the slide.** Development set. Left: held-out RMSE against
the fraction of notes hidden, for the proposed model (blue), no graph
(amber dashed), features only (green dotted); open markers at the
right are the leave-one-out limit, where every note is predicted from
all the others with hyperparameters fitted on the full piece. Right:
what each ingredient adds per level as a paired difference in RMSE
with 95 percent intervals, blue the graph, vermilion the music-model
features. The ordering is stable at every level.

**How to read the left panel.** Everything improves as fewer notes are
hidden because there is more context. The proposed model is lowest at
every level. Features only flattens out below 30 percent: a cross-piece
regression cannot use extra context; the graph can.

**The component table, now in the text.** 
The ablation table, in the thesis: For each piece, the full
model and the model with one component removed are run on the same
masks, and `Δ = metric(full) − metric(removed)` is averaged over 30
pieces; negative means the component helps. Rows: graph in RMSE,
music model in RMSE, graph in log score; columns 50 percent down to
leave-one-out. Everything shown is significant except the graph at 10
percent (both metrics) and at 30 percent (log score only), and the
music model at 50 percent and in timing. The graph carries calibration;
the music model's value grows with observed context. Fixing a
cross-piece mean loses almost nothing: the accuracy gain is per-piece
re-weighting of the features.

**Example row.** Graph, log score, 40 percent: −0.067. Removing the
graph makes the predictive density of the truth worse by 0.067 nats per
note on average.

## Slide 15. Divider: Strings and winds


One sentence: "Second setting: same prior, but the variables now come
from audio through a pitch tracker, noisily, with a variance per note,
and six channels instead of three."

## Slide 16. Setup: audio instead of MIDI


**What is on the slide.** Data: URMP, 44 chamber pieces, duets to
quintets, 13 instrument types, each instrument recorded on its own
monophonic track with note annotations and the score. Split by
composition because arrangements share recordings: development 31
pieces (78 tracks, about 17,000 notes), one-shot test 13 pieces. Pitch
now carries expression: intonation and vibrato within every note. A
pitch tracker, pYIN, turns each track into frames: time, `f_0`, voicing
confidence. The variables must be estimated from these frames,
noisily, with a variance per note. Two measurements of every note
exist: our tracked curve, and the corpus's own annotation of the same
track, cleaner but still a measurement; evaluation uses the second.

**Why "split by composition".** URMP records each composition once and
offers it in several arrangements; the same audio appears under
different piece names. Splitting by piece name would leak the test
audio into development.

## Slide 17. From the tracked frames to the pitch-curve channels


**What is on the slide.** The tracker's frames `(t_j, f_{0,j}, π_j)`: time,
fundamental frequency, voicing confidence. Frame rule: note `i` keeps
`J_i = { j : voiced, π_j ≥ q_0.2, t_j ∈ [o_i, o_i + u_i) }`, at least
4 frames, where `o_i, u_i` are the note's annotated onset and duration
and `q_0.2` is the track's lowest confidence quintile. Cents curve:
each kept frame becomes `x_j = 1200 log₂(f_{0,j} / 440) − 100 (p_i − 69)`,
the deviation in cents from equal temperament at A4 = 440 Hz, `p_i` the
MIDI pitch.

**Example.** A4 is MIDI 69. A frame at 442.5 Hz on an A4:
`1200 log₂(442.5/440) ≈ 9.8` cents, minus `100 × 0` = 9.8 cents sharp.
On an A3 (MIDI 57) at 221 Hz: `1200 log₂(221/440) ≈ −1193`, minus
`100 × (57 − 69) = −1200`, giving +7 cents.

**The channels.** 
The annotated note in the figure: within one note, a
sustained level plus one oscillation. The level is the intonation
`c` (10.1 cents in the figure), the oscillation's half-height is the
vibrato extent `γ` (15.0 cents), its period is `1/f` with `f` the
vibrato rate (6.1 Hz). The model receives `c`, `log γ`, `log f`; `γ`
and `f` are positive, so they are modelled on the log scale.

**How to say it.** "Three numbers describe how the pitch behaved inside
the note: where it sat, how wide it wobbled, how fast."

## Slide 18. Estimating the channels with a Gaussian process


**What is on the slide.** The note's cents curve is given a Gaussian
process prior with a two-component kernel, a special case of the
spectral mixture kernel (Wilson and Adams 2013):
`k(Δt) = w_1 exp(−2π² v_1 Δt²) cos(2π μ_1 Δt) + w_2 exp(−2π² v_2 Δt²)`,
vibrato component plus drift component.

**Symbols.** `Δt` is the time lag between two frames in seconds (so
`τ` stays the timing channel); `μ_1` the vibrato rate in Hz; `w_1, w_2`
the powers in cents²; `v_1, v_2` the bandwidths in Hz². The prior says
only: the curve's energy sits in a band of width about `√v_1` around
the rate `μ_1`, plus a low-frequency drift band around zero. Fitted per
note by exact marginal likelihood, with variances from the curvature
of the evidence. Channels from the posterior: `c` is the realized
average of the curve, `γ` comes from the vibrato component's amplitude,
`f = μ_1`.

**Example.** `w_1 = 100` cents² and `μ_1 = 6` Hz means the prior expects
a vibrato of about 10 cents standard deviation at 6 Hz. `v_1 = 0.5`
Hz² lets the rate wander by roughly 0.7 Hz.

## Slide 19. Results (strings and winds, URMP): confirmed once


**What is on the slide.** The same prior, unchanged. Six per-note
channels estimated from audio, each with its own variance, used as
given: `y_i = [c_i, log γ_i, log f_i, ℓ_i, τ_i, d_i^vib]`, intonation,
extent, rate, loudness, timing, vibrato onset delay. Preregistered,
scored once on 13 held-out pieces; `Δ` = full model minus the same
model without the graph term, paired per piece, negative favours the
graph. Intonation recovery −0.877 cents, significant. Vibrato extent
calibration −2.990 in log score, significant. Vibrato rate calibration
−0.564, significant. Timing calibration −0.030, not significant, the
one failed claim, reported as such. Coverage 0.88 to 0.91 on all six
channels at nominal 0.90.

**The three new channels.** `ℓ_i` is loudness, a per-note level from
the audio; `τ_i` is timing, the onset relative to a tempo line fitted
through the annotated onsets with that note left out; `d_i^vib` is the
vibrato onset delay, how long after the note starts the vibrato begins.

**Honest line, if asked.** "The timing claim failed. We report it. Its
named remedy, correlated timing noise across notes, was then registered
and run once on the same pool, re-used with disclosure, and passed:
the next slide."

## Slide 20. Two registered follow-ups on the same pool, run once (27 October)


**What is on the slide.** Two claims developed on the development
tracks after the August confirmation, each registered with its decision
rule frozen on the morning of 27 October and run once that afternoon on
the same 13 confirmed pieces, re-used with full disclosure: the pool had
been spent on the four August claims, its plain-model fits were known
when these claims were chosen, and the claim-specific quantities had
never been computed on it. The inputs are the archived August targets,
checksum-verified; the masks and seeds are August's. Two changes to the
model, one at a time, each against the published model on the same
cells. Verdicts: timing calibration passed (the primary), timing
recovery passed; intonation calibration missed (the primary), intonation
recovery passed. One of two claims confirmed.

**The timing-noise line.** `Σ_τ(ρ)` is the noise covariance of the
timing channel, `τ`, across the notes of one track in score order.
`d_i` is the variance the timing estimator gives note `i` (the tempo-line
fit with that note left out), used as given, as everywhere in this
setting. `ρ` is the correlation between the alignment errors of
neighbouring notes: entry `(i, j)` is the geometric mean of the two
variances times `ρ` to the power of the note distance, so the
correlation decays by a factor `ρ` per note. `ρ` is picked per track
and seed from the grid 0, 0.15, up to 0.9 by the marginal likelihood of
the observed notes alone; the hidden notes are then predicted through
the full covariance, prior plus noise, so an observed neighbour's error
informs the hidden note's. `ρ = 0` is exactly the published diagonal
model, which is the paired baseline. Example: a 100-note track where
the annotated onsets drift late over a bar: with `ρ = 0.45` (the
median found), a hidden note between two late-aligned neighbours is
predicted late too, and its band narrows; with the diagonal model the
same note is predicted from the prior alone with a band too wide where
the neighbours agree, which is what the August timing-calibration miss
measured.

**The tonal-metric line.** `W^tonal_ij` is the edge weight between
notes `i` and `j` with the semitone term of the score graph replaced:
`b_i` the beat onset, `ℓ_b` two beats as before; `d_5(i, j)` the
circle-of-fifths distance between the two pitch classes (C to G is one
step, C to C sharp is seven), with its own length scale `ℓ_5`; and an
octave term, the pitch difference in octaves with length scale
`ℓ_oct`, which keeps register locality because `d_5` is blind to it.
The hypothesis, committed in development: mistuning travels with
harmonic relatedness (a fifth shares its tuning error) more than with
semitone proximity. Example: a cellist's slightly flat G and the D a
fifth above it, in the same bar, are neighbours under this metric and
share a posterior offset; the A flat a semitone up is far.

**The table.** Each row is one claim: the paired difference, new arm
minus published arm, per track and seed across the 80 cells (40 unique
tracks, two mask seeds), with the 95 percent bootstrap interval
(cluster bootstrap over unique tracks for the timing rows; the plain
percentile bootstrap of the August tables for the intonation rows, as
registered). Negative favours the new arm; "pass" means the interval
excludes zero on the predicted side, "miss" means it does not. The
timing calibration row is the log score (negative log predictive
density) on held-out timing: `−0.121` with interval `[−0.156, −0.089]`,
the development basis was `−0.111`. Timing recovery, RMSE in seconds,
`−0.011`, which is 0.058 to 0.047 seconds, a 19 percent cut. Intonation
calibration under the tonal metric: `−0.023` with interval
`[−0.049, +0.001]`, the mean on the predicted side but the interval
reaching zero, so a miss by the frozen rule; the development basis was
`−0.050` with interval `[−0.072, −0.029]`. Intonation recovery,
`−0.312` cents with interval `[−0.599, −0.081]`, registered as
underpowered and starred anyway.

**The last line.** Coverage of the 90 percent bands stayed at nominal
under the correlated noise (0.908 against the diagonal's 0.911);
`ρ > 0` was selected on 79 of 80 cells. The three arms that did not
change (plain graph, learned-scale graph, no graph) reproduced the
August table digit for digit, which is the pre-stated identity check:
this run altered nothing confirmed earlier. The predicted timing
penalty under the tonal metric did not appear in the registered
variant (`−0.002`, not significant).

**Honest lines, if asked.** "Why re-use a spent pool?" Because the
only other audio corpus on disk is synthetic; the re-use is disclosed
in the registration and in the thesis, and the evidence is labelled
replication-grade, weaker than the August kind. "Did the tonal metric
fail?" Its calibration claim missed by the frozen rule and its recovery
claim held; the metric stays exploratory, nothing is adopted, and the
thesis reports both numbers. "Is the timing model now the default?"
No: adopting it would re-run the published tables and needs its own
registration; the claim is confirmed once, as a repair of the August
miss.

## Slide 21. Divider: The waveform as the observation


One sentence: "Third setting: no tracker. The waveform itself is the
observation. Methodology and development studies, and one held-out
test at phrase scale."

## Slide 22. The method, the data and the signal model


**Where the method comes from.** Reference works: DDSP (Engel 2020),
gradients through a differentiable synthesizer, with MIDI-DDSP (Wu
2022) the closest cousin for performance control; GPs for music audio
(Alvarado and Stowell 2016), per-note windows over a shared time axis;
spectral mixture kernels (Wilson and Adams 2013), of which the curve
prior's kernel is a special case. Right, what this thesis does with
them: takes the implementation idea, gradient descent through the
synthesizer; keeps the exact amplitude-marginalized likelihood and the
GP priors, no neural encoder and no spectral surrogate loss as the
objective; measured that a learned many-component spectrum adds
nothing over the hand-structured two-component kernel on these curves.

**How to say it.** "We borrowed one engineering idea, gradients through
synthesis, and kept our own probabilistic model. The next three slides
are that model."


**What is on the slide.** The method and data lines; the one equation
`x = Φ(z) a + ε` with its sizes (`x ∈ R^m`, `Φ(z) ∈ R^{m×p}`, `a ∈ R^p`,
`ε ~ N(0, σ² I_m)`); and a five-row table naming each symbol. The
message: the curve fixes the SHAPE of every column, the amplitudes only
SCALE them. The two slides that follow unpack how `z`
becomes `Φ(z)` and how a curve is scored without knowing `a`, then how
the best curve and its band are found.

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
  is what makes slide 23's likelihood a Gaussian with covariance
  `Φ Σ_a Φ^T + σ² I` (spread from the unknown amplitudes, plus the noise
  floor on every sample). `σ²` is not chosen by hand: per note, the
  code solves the best amplitudes by least squares on the flat-curve
  design and takes the residual's variance as `σ²`, refitting it once
  halfway through the curve fit. In a real recording `ε` contains bow
  and breath noise, room reverberation, harmonics above the eighth,
  slight inharmonicity, and the mismatch between piecewise-constant
  amplitudes and the true envelope. Caveat, said on slide 43: it is not
  white; the residual at the harmonics is strongly correlated across
  samples (0.944 measured), so the likelihood counts correlated samples
  as independent votes and is overconfident, which is why the
  calibrated floor exists.
- `x`: the whole waveform of the note as a vector, one entry per audio
  sample, NOT a single number. A recording is a sequence of pressure
  readings at a fixed rate; at 16,000 per second a 1.2-second note gives
  `m = 19,200` entries, each a value like 0.012 or −0.028. The vermilion
  line on slide 25 is `x` plotted against time; each point on it is one
  entry.
- `x = Φ(z) a + ε`: a matrix equation. `x` has `m` rows; `Φ(z)` has `m`
  rows and `p` columns, each column a waveform of length `m`; `a` has
  `p` entries; `ε` has `m` entries. Sample by sample: the recorded
  reading equals the sum of the harmonic waves at that instant, each
  scaled by its knob, plus noise. "Every audio sample votes" on slide 23
  means every one of the `m` entries contributes to a candidate's score.
- `K`: the number of harmonics, 8 in the studies.
- `m` and `p`, the two sizes of the matrix. `m` is the number of audio
  samples in the note, the length of `x` and the number of rows of
  `Φ(z)`: a 0.5-second note has `m = 8,000`, a 1.2-second note
  `m = 19,200`. `p` is the number of columns of `Φ(z)`, which is the
  number of knobs in `a`: cosine and sine per harmonic gives `2K = 16`,
  and the note is split into 2 time chunks so loudness can change
  within it, so `p = 2 × 8 × 2 = 32`. Scoring one candidate curve costs
  about `m · p²` operations (slide 23), here `19,200 × 32² ≈ 20 million`,
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
  timbre stays linear in `a`, which is what lets slide 23 integrate it
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
  whole segment. That concatenation is where the leak of the withdrawn first run lived
(slide 31's note).

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

**Sizes of everything on slides 22 to 24, symbolically.**

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

For a chain of `N` notes (slide 27): each note has its own `Φ_i`,
`m × p_i` with `p_i = 2KC`, on its own slice plus margin times its fade
windows; the chain design is `Φ = [Φ_1 … Φ_N]`, `m × P` with
`P = N · 2KC`; `a` is `P × 1`, `G` is `P × P`; the boundary offsets `db`
are `(N−1) × 1` with an `(N−1) × (N−1)` prior covariance (slide 31);
the per-note constants `c` are `N × 1` with an `N × N` prior covariance.
Cost `O(m p²)`: forming `Φ^T Φ` is `m` rows times `p × p` entries; the
Cholesky of `G` is `p³`, negligible since `p ≪ m`.

## Slide 23. From the pitch curve to the likelihood: the timbre integrated out


**The first block.** The chain from curve to design: pitch curve
`z = cents(t)`; frequency `f_0(t) = f_pitch · 2^(cents(t)/1200)`; phase
`φ_k(t) = 2π k ∫_0^t f_0`, `k = 1..K`; design
`Φ(z) = [cos φ_k(t), sin φ_k(t)]_{k=1..K} ∈ R^{m×p}`. One sentence of
intuition (frequency is the speed, phase the distance travelled) and a
small-print line on the two implementation details: the curve is held
at `J` knot values and interpolated; the columns are repeated per time
chunk, `p = 2KC`. The step-by-step reading with the toy example is in
the slide 22 notes above ("From `z` to `Φ(z)`").

**If asked why the cosine and sine pair:** the two columns with two
weights produce a sinusoid of any amplitude and any starting phase
(`A cos(φ − θ) = A cos θ · cos φ + A sin θ · sin φ`), so the unknown
starting phase of each harmonic is absorbed into `a` and the timbre
stays linear. That linearity is what the next slide needs.


**The second block.** One implication: `a ~ N(0, σ_a² I_p)` implies
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
  in words until 2026-10-21; `z` IS the curve, slide 22). Under this
  curve the audio should look like Gaussian noise with a covariance
  built from the harmonic waves. In the thesis `z` names every
  nonlinear position the synthesizer needs, so in a chain fit it also
  holds the boundary offsets; on slides 22 to 24, one note with fixed
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

## Slide 24. Finding the curve, with its uncertainty


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

## Slide 25. One note: the model against the raw audio


40 ms of a violin note with strong vibrato. Vermilion: the recorded
audio `x`. Grey dashed: the best single constant pitch. Blue: the
inferred pitch curve. The constant drifts out of alignment within 40 ms
because vibrato changes the odometer's speed; the curve stays locked.
This is phase accumulation made visible, and why the method is
sensitive to small pitch deviations.

## Slide 26. The curve against the tracker, at scale (development)


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

## Slide 27. Note boundaries: the chain model


**What is on the slide.** A chain of `N` consecutive notes with
boundaries `b_1 … b_{N−1}` and one pitch curve `z_k` per note. Five
labelled lines plus the bands:
1. `w_k(t) = σ((t − b_{k−1})/s) · σ((b_k − t)/s)`, with
   `σ(u) = 1/(1 + e^{−u})` the logistic function and `s = 10 ms` the
   softness: note `k`'s window, a crossfade that rises after the
   boundary before it and falls at the boundary after it. The first
   note has no fade-in, the last no fade-out.
2. `Φ(z, b) = [w_1 ⊙ Φ_1(z_1) | … | w_N ⊙ Φ_N(z_N)] ∈ R^{m × Np}`: each
   note's own design matrix (slide 23), every column scaled sample by
   sample by that note's window (`⊙`), then all notes' blocks placed
   side by side. Each note's block is built on its own time slice plus
   a 0.2 s margin, so its columns are zero far from the note.
3. `p(x | z, b) = N(x; 0, σ_a² Φ Φᵀ + σ² I_m)`: the same collapsed
   likelihood as slide 23, now a function of the boundaries too,
   because they shape the windows.
4. `b^(0)`: the starting boundaries come from the score through the
   local tempo line, with no annotation access. The tempo line is a
   straight-line fit of score time to clock time around the chain, so
   `b^(0)` is "where the score says the note starts at this tempo".
5. `(ẑ, b̂) = argmax [log p(x | z, b) + log p(z)]`: curves and
   boundaries found together by gradient descent, because the window
   is differentiable in `b`. The curve prior `p(z)` is the knot prior
   of slide 24, per note.
6. `Σ_{z,b}` from the Laplace over `(z, b)`: the bands come from the
   curvature at the optimum, as on slide 24, now including the
   boundaries.

**How to say it.** Each note is switched on and off by a soft window.
Where one note's harmonics stop explaining the samples and the next
note's start, the likelihood is highest, so the boundary settles there.
The curves and the boundaries are fitted at once with the same
gradients; nothing but the score and the audio is used.

**Example.** Two notes, C then E, with the score's tempo line putting
the boundary at 0.60 s but the player having started the E at 0.63 s.
At `b = 0.60` the C's harmonics are asked to explain 30 ms of E, which
they cannot, so the score is poor; sliding `b` to 0.63 lets each note
explain its own samples and the score rises. The gradient points that
way and the optimizer moves `b` there.

**Two details, if asked.** The chain study tried letting the crossfade
width be free; the data drove it to the minimum (1 ms) and that beat a
fixed 10 ms width, so the transition is effectively sharp and the
studies use a fixed sharp edge. And when two consecutive notes have the
same pitch the harmonic stacks are identical on both sides, so the
likelihood carries no boundary information; those joints are reported
separately (slide 28).

## Slide 28. Note boundaries from the waveform: one chain, and the numbers


Panel A: a five-note violin passage with three kinds of vertical line.
Black: where the annotation says a note starts. Grey dashed: where the
score alone, through the local tempo line, predicts it (`b^(0)`). Blue:
where the fit put it (`b̂`). Panel B: the per-note pitch curves from the
same fit; the 90 percent band widens exactly at the joints, where two
notes overlap and the audio carries no boundary information.

Numbers: starting from the tempo line, the median boundary error
halves, 40 to 20 ms, over 128 boundaries. The remaining 20 ms is
systematic: the audio shows where the sound changes, the annotation
marks where the musician intended the note. A model of the attack
transient at the joint is the named next step.

**The scatter behind the numbers, if asked.** 
One dot per boundary, 128, in the thesis figure. Horizontal: the error of the score-only
tempo-line guess. Vertical: the error after the joint fit. Below the
diagonal improved: 73 percent. Median 40 to 20 ms. Repeated-pitch
boundaries excluded: identical harmonic stacks on both sides carry no
boundary information. Dots far above the diagonal: the systematic
acoustic-versus-written remainder.

## Slide 29. The end goal on one passage (development)


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

## Slide 30. The score prior inside the waveform inference: the setup


**What is on the slide.** The week of 2026-10-15 to 20 on one slide: the
score graph of slide 7 brought inside the waveform inference of slides 22 to 27, and tested at three spans. This slide is the design; the
table on slide 31 is the verdict.

**Symbols.** `θ`, an `n × 1` vector, one latent per element of a chain:
for the first and third span the per-note intonation constants `c`
(cents above the written pitch), for the second span the per-joint
boundary offsets `b` (milliseconds from the tempo line). `t`, a scalar,
the common-mode term: the whole chain's shared offset (a tuning offset
in cents, or a shared timing offset in ms), with prior variance
`σ_t²` (50 cents or 50 ms squared). `θ̃`, the per-element part on top of
the shared offset. `σ_ref²`, the track's own variance of the reference
centres (finite, within 150 cents of zero, the hidden note's included;
for the boundary span, the track's timing residuals in ms²), a
population scale, not tuned, the same in every arm. The hidden note's
reference centre also decides whether a phrase is eligible. `K_arm`, the `n × n` correlation
matrix of one arm: the identity for the independent prior; for the two
structured priors, the inverse of `I + L` scaled so its diagonal is all
ones, which is what "matched marginal variance" means (every element
has the same prior variance under every arm; only the correlations
differ). `L_arm`, the Laplacian of the arm's graph: for the temporal
chain, consecutive notes linked with unit weight; for the score graph,
the time-and-pitch edge weights of slide 7. `K`, the full prior
covariance of `θ`, the arm's part plus the rank-one common-mode term.
`h`, the index of the hidden element; `−h`, all the others.

**How one cell runs.** Take a chain. Remove the hidden element from the
audio and from the design matrix (its window columns go, not just its
samples; the withdrawn first run, below, explains why that matters). Fit the chain under the
likelihood of slide 23 with the prior above. The observed elements are
pinned by the audio, nearly identically under every arm. The hidden
element has no data, so given its neighbours' values it follows the
prior conditional, the Gaussian conditioning formula on the slide:
mean `K_{h,−h} K_{−h,−h}^{−1} θ_{−h}`, a weighted average of the
neighbours where the weights come from the graph, and variance
`K_hh − K_{h,−h} K_{−h,−h}^{−1} K_{−h,h}`, reduced from the prior by
however much the neighbours tell. The neighbours are themselves
uncertain, so the band actually scored comes from the joint Laplace
approximation over all elements and is a little wider than the
formula. Score: the distance from that
posterior mean to the reference centre (the same estimator run on the
corpus's annotated pitch curve). Pair the three arms on the same cell,
bootstrap the mean difference, 95 percent interval.

**Example.** A four-note violin chain, hidden third note, reference
centre +9 cents, chain mean +4 cents. Independent prior: the hidden
note gets the chain mean through `t`, +4. Temporal chain: its two
neighbours, say +3 and +5, so about +4 again. Score graph: the same two
neighbours and, if the first note has the same pitch, that one at
higher weight, say +8, so +5. All three are short of +9 by a similar
amount, which is what the first row of slide 31 says.

## Slide 31. The score prior inside the waveform inference: three spans


**What is on the slide.** The three-span table and the reading
paragraph, moved to their own slide on 18 November so the table no
longer runs off the right edge; the setup is the slide before.

**The table.** First row, 47 chains of three to five notes from four
development tracks: graph minus independent −0.02 cents, graph minus
temporal −0.19, both intervals across zero. Second row, 40 joints: the
same two contrasts in milliseconds, both ties. Third row, 48 phrases of
six to ten notes from 30 of the 78 development tracks searched, hidden pitch recurring
at least two notes away: graph minus temporal −1.63 cents, interval
excludes zero, the pre-stated primary; graph minus independent −0.07, a
tie. Not on the slide but in the bottom line: temporal minus independent
+1.56, interval excludes zero, so the temporal chain hurts at phrase
scale and the graph repairs that harm.

**The sentence.** "Every prior gives a missing element its context's
offset, worth about 7 cents and 10 to 14 ms. The score graph is never
significantly worse than a time-only chain and better at phrase scale. Beyond the
offset, nothing we can measure yet." If asked why the first run is not
here: it was withdrawn, next paragraph.

**The withdrawn first run, if asked.** In the first protocol the objective as the hidden note's `c` sweeps −100 to +100
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

**The fourth row, added 18 November.** The phrase-scale claim was
registered that morning (claim, rule and cell rule frozen and tagged
before the run) and run once on the strings-and-winds confirmation
tracks: 48 phrases from 20 of the 40 tracks by the unchanged
development rule, 14 strings, 25 winds, 9 brass. This is the pool's
third use, disclosed, and the first at the waveform level: no waveform
script had ever read those tracks. The primary, graph minus temporal on
the hidden note, is `−0.82` cents with the interval `[−1.59, −0.05]`,
excluding zero, better on 62 percent of phrases. The interval resamples
the 48 phrases and treats phrases from one track as separate
observations. The expectation stated
before the run for this winds-heavy mix was about `−0.9` (development
`−1.63` was carried by strings), and the power was only 0.3 to 0.5, so
the star was not expected with confidence. Everything else repeats
development: graph minus independent `−0.13` with the interval covering
zero (the strong form gets no new support), temporal minus independent
`+0.68`, same sign as development (`+1.56`) but with the interval
including zero, so on held-out tracks the temporal prior's harm is not
itself established and the graph's advantage is read against the
temporal prior alone; clean rows zero, coverage 0.94 to 0.98, no
dose-response. By family, strings `−1.55`, winds `−0.54`, brass
`−0.44`, none starred alone; strings are largest in both studies, winds
and brass change order. Say it in one breath: "the weak form
holds on tracks the waveform studies had never read; the strong form
does not, there either."


## Slide 32. What in the graph does the work: the time window


**What is on the slide.** The same question at whole-track scale with
the power that chains lacked, on both corpora, and its answer: which
part of the score graph carries its value.

**Symbols.** `W_ij`, the edge weight between notes `i` and `j` of slide 7: `b_i`, `b_j` their onsets; `p_i`, `p_j` their MIDI pitches;
`ℓ_b = 2` and `ℓ_p = 4` semitones the two length-scales. On piano the
onsets are score beats, so the window is two beats. On strings and
winds the corpus annotates onsets in seconds and has no beat grid, so
the same `ℓ_b = 2` is two seconds of performance time. Time Gaussian:
the same weight with `ℓ_p → ∞`, so the pitch factor is 1 for every pair
and only the time window remains. Time chain:
weight 1 between consecutive notes, 0 elsewhere. Everything else in the
GP is identical across the three arms, including the per-piece
intercept column in the feature kernel, so every contrast is beyond the
common offset that dominated slide 31.

**The table.** Strings and winds, the development evaluation of slide
19 with 30 percent of the notes hidden per track, 78 tracks, 153
track-and-seed pairs for intonation and loudness and 148 for vibrato,
paired per track and seed. Score graph minus time chain on
intonation −0.50 cents, interval excludes zero: the graph beats the
naive time prior. Score graph minus time Gaussian −0.02, interval across
zero: removing the pitch term changes nothing on intonation; on the two
vibrato channels removing it even helps a little (+0.04, +0.03); on
loudness the pitch term helps a little (−0.007, interval excluding
zero), the one place it shows. Piano,
the headline model of slide 13 with the pitch term removed from its
graph, paired per piece on the 30 development pieces, 4 seeds: the
score graph is worse than the pitch-less graph by 0.008 in pooled RMSE
and 0.019 in log score, both intervals excluding zero. Note the sign
convention: negative favours the first-named arm, so a positive number
in the last column means the pitch-less graph is better.

**Example.** Two notes a beat apart, one a C and one a G (7 semitones).
Score graph: time factor exp(−1/8) ≈ 0.88, pitch factor exp(−49/32) ≈
0.22, weight 0.19. Time Gaussian: weight 0.88. The pitch-less graph ties
these two notes more strongly, and the data say that is at least as
good.

**The sentence.** "On both corpora, at track scale, what
helps is smoothing over a short time window, two beats on piano, two
seconds on strings and winds. Linking notes by pitch adds nothing on
intonation, a little on loudness, and costs a little on vibrato." Then the
qualifier, always: "The confirmed results stand, because they compare
the full model with its alternatives. What narrows is the attribution
of the graph term." If asked whether the thesis should switch to the
pitch-less graph: that is a change to a reported protocol and would need
its own registration; it is the decision on the plan slide.

## Slide 33. Divider: Motif structure


Say: "The attribution left the graph with a modest role, a short
time window. This last section asks whether a structure the window
cannot see, the repetition of a motif, carries anything of its own.
Development studies first, then one pre-specified test."

## Slide 34. Motif edges: the family


**What is on the slide.** The one new edge family and its two forms;
the closed-form effect of a motif edge on a hidden note is the next
slide.

**Symbols.** `O_k`: the k-th annotated occurrence, a set of notes with
a span in beats from `b_k^start` to `b_k^end` and a type `q(k)` (two
occurrences of one type are returns of one motif). The annotations are
BPS-motif, which covers all 32 Beethoven sonata first movements. `u_{i,k}`:
the relative position of note `i` inside occurrence `k` (a note in two
occurrences has one position in each), 0 at the start
and 1 at the end, so returns of different length share one coordinate.
`(W_motif)_ij`: a Gaussian in the difference of relative positions,
taken over every pair of occurrences of the same type that contain `i`
and `j` (the max over pairs, so a note that belongs to several
occurrences keeps its strongest link). `ℓ_u = 0.1`: a tenth of the
span, fixed. Every pair of returns gets a soft block of edges,
strongest at equal relative position, whether or not the returns have
equal length; no self-edges, no alignment and nothing learned inside the
family.

**Two forms.** Shared: the thesis adjacency gains the family with one
weight `w_m`, learned on the log scale by the per-piece evidence like
every other hyperparameter, and shared by the three channels through
the coregionalization matrix `B`. Per channel: the family is its own
additive kernel term, `K_m` the shape-normalized kernel of the motif
Laplacian on the motif notes (zero elsewhere), with one scale `c_m` per
channel, so a channel can switch the family off on its own. Because
`K_m` factorizes as `X_m X_m^T`, it is a feature kernel of the model
and nothing in the inference changes.

## Slide 35. Motif edges: what they do to a hidden note


**What is on the slide.** The conditional of a hidden note in the
single-channel GMRF, split by edge family, and the two readings that
follow from it; split from the family slide on 18 November.

**The conditional.** In the single-channel GMRF with precision `I + sL`
the mean of a hidden note is a weighted average of its neighbours and
its variance is `1/(1 + s D_hh)`, with `D_hh` the weighted degree, the
diagonal of the degree matrix. Split
the degree by family and the mean splits into the time-neighbour
average and the motif-neighbour average, each weighted by its share of
the degree. Example: a hidden note with a time degree of 2 and a motif
degree of 1 at `w_m = 1`, `s = 1`, puts weight one half on the mean of
the notes around it and one quarter on the mean of the returns; the
remaining quarter goes to the prior mean of zero. Its variance falls
from `1/3` to `1/4` for the extra edge.

**Two readings.** If performers shape a motif alike at each return,
the motif average is closer to the truth than the time average, the
evidence raises `w_m`, the error falls, and at fixed scale the
conditional variance falls with the extra degree (the band actually
scored also carries the neighbours' uncertainty and the noise). If not,
the motif average is noise, the evidence drives `w_m` toward zero and
the arm approaches the time-only graph.
That second route is the falsification: the study can come back empty.


## Slide 36. Motif edges, development: whole occurrences hidden, 21 movements, 4 seeds


**What is on the slide.** The second development study's
occurrence-scheme results: the shared form and the per-channel form,
each against the time-only graph on the hidden motif notes. The first
study's two-seed timing number is in the first bullet.

**Setup.** 21 Beethoven first movements from ASAP aligned to the
annotations (25 annotated movements have an ASAP score; two are piano
test pieces and were never touched, two fall below the alignment
threshold), 400-note excerpts, score features without the
music model so that the contrast is one score structure against
another on identical features, a per-piece intercept in every arm. The
masking scheme draws whole occurrences for hiding while at least one
occurrence of the type is left out of the draw (overlapping occurrences
can still hide some of its notes; a hidden note is scored only when
another occurrence of its type has an observed note), four seeds; the 672 fits cover both masking
schemes, and the table is the occurrence scheme. Paired per
movement, 95 percent percentile bootstrap, negative favours the motif
arm.

**The table.** Shared form: timing log score `−0.071` with the interval
reaching `+0.097`, timing error `−0.0027` seconds with the interval
excluding zero, articulation RMSE `−0.035` with the interval excluding
zero, velocity log score `+0.033` with the interval reaching zero and
coverage of the 90 percent band `0.835`, below nominal: one weight for
three channels ties the velocities of returns together more tightly
than performers play them. Per-channel form: timing nothing (`−0.005`),
articulation `−0.019` with the interval excluding zero, velocity
`−0.086` with the interval excluding zero and coverage `0.944`. So the
per-channel form removes the velocity penalty and gives up the timing
log-score gain.

**The three bullets.** First: the first study, on two seeds, had a
timing star of `−0.117`; seeds 0 and 1 reproduce it bit for bit, seeds
2 and 3 give `−0.026` with a wide interval, so that star rested on two
seeds and three movements. Second: the learned per-channel scales say
where the family lives. On timing it is below `0.01` on 19 of 21
movements, switched off, because the time-only graph is already well
calibrated there. On articulation it exceeds `0.1` on 10 movements, on
velocity it exceeds `0.01` on 7. Third: the one effect that holds on every
pair of seeds and in both forms is articulation accuracy, and that is
what the pre-specified test takes as its primary. Under random masking
the time window already interpolates inside an observed occurrence and
the family has little to act on.

## Slide 37. Motif edges from a discovery model's predictions (development)


**What is on the slide.** The same evaluation with the annotations
replaced, as the edge source only, by the occurrences a motif-discovery
model predicts on each movement held out from its training fold.

**The model, in one pass.** Notes are vertices of a typed graph
(simultaneity, temporal proximity, overlap). A pretrained encoder gives
each note a vector `h_i`, relational graph convolutions update it. Two
heads: `m_i`, the probability that note `i` belongs to any motif, and
`â_ij`, an affinity on candidate pairs from the graph edges. Training
is label propagation: the annotated occurrences seed a membership
matrix `M^seed`, memberships `M` diffuse along the learned affinities with
restart weight `α`, and the losses act on `M` at the pair, occurrence
and type levels. Decoding gates the affinity by the memberships,
thresholds it (the quantile `0.86` is the model's headline operating
point), takes connected components as occurrences and clusters their
pooled states into types. Lalang and Yoshii, ISMIR 2026.

**What changed and what did not.** Predicted occurrences are mapped to
the excerpt by onset and pitch exactly as the annotations were, the
span is the first to last matched onset, the type is the pattern index.
The masks and the scored notes stay those of the annotated occurrences,
identical to slide 36, so the only change is where the edges come from.
Membership against the annotation: precision from `0.12` to `1.00`,
recall from `0.44` to `0.99`, F1 median `0.78`; on two movements the
model predicts several times the annotated number of motif notes.

**The table.** Shared form, predicted edges: timing log score `−0.034`,
`0.48` of the annotated `−0.071`, with the interval reaching `+0.003`;
articulation RMSE `−0.0135`, four tenths of `−0.035`, interval reaching
`+0.003`; velocity coverage `0.865`. The two edge sources act on
related movements: per-movement effects correlate at `0.59` on timing
and `0.84` on articulation, and 12 of 21 movements lie below zero for
both. The prediction fixed before the run named the per-channel form:
half the annotated timing effect with an interval clear of zero. In
that form the predicted edges move the timing log score by `+0.036`
`[−0.010, +0.095]`, the wrong way, and cost `0.0014` seconds of timing
error, so the prediction is not supported. The shared-form numbers in
the table were recorded without a prediction. The sentence: in the
shared form a model's occurrences give point estimates of about half
the timing effect and four tenths of the articulation effect, with
intervals that include zero; in the per-channel form they give
nothing.

## Slide 38. The pre-specified test: held-out performances of the same movements


**What is on the slide.** The registration of the one motif claim that
will be tested, before it is run, and the power behind it.

**The pool.** For each of the 21 development movements, the other
aligned ASAP performances, at most four per movement in sorted order,
none read by any run so far. 18 movements have at least one (three have
none and drop out), 58 performances in all. Scores, annotations, the
masking rule with its four seeds, and the scoring rule are the
development study's. Each performance has its own aligned excerpt, so
its edges and masks are rebuilt by the same rules, and it supplies new
targets. So the
test asks whether the development finding holds across performers of
the same pieces, not across pieces, and the thesis labels it that way
wherever it appears. The unit of analysis is the movement, with
performances averaged inside it before pairing, so no movement counts
four times.

**The claims.** M1, the primary and the only gate: articulation RMSE on
the hidden motif notes, shared form minus the time-only graph,
negative with the 95 percent interval excluding zero. M2, secondary and
non-gating: the timing log score by the same rule. M3, report only:
velocity log score and coverage, timing RMSE, and the per-channel form
running beside the shared one. The arm is the
shared form because its power is the higher of the two (`0.95` against
`0.93`) and its articulation effect the larger and more even across
movements.

**The power table.** By drawing 18 of the 21 development movements'
differences with replacement many times (per-movement differences from
seed-averaged RMSEs, slightly different from the test's root of the
seed-averaged squared error) and asking how often the interval clears zero: articulation
RMSE `0.95` at the measured effect and `0.51` at two thirds of it;
timing log score `0.33` at the measured effect and below `0.2` at two
thirds. So M1 is the well-powered claim and M2 is included because it
was the first study's hypothesis, with its weak power stated.

## Slide 39. The pre-specified test: the result, run once


**What is on the slide.** The result table of the registered test
and the one sentence to say; the pool, claims and power table are the
slide before.

**The result, run once on 18 November.** Frozen and tagged at midday,
two days ahead of the planned freeze with nothing in the protocol
changed, and run once that afternoon: 696 fits, 14 shards, under two
hours. M1 holds: the articulation error of the hidden returns falls by
`0.029` with interval `[−0.049, −0.011]`, with a point estimate below
zero on 15 of the 18 movements, against `0.035` on development. The
per-movement effects of the two studies correlate at `0.88`, and 12
movements are below zero in both. The interval is for the mean over
movements; no single movement is tested on its own. M2 does not hold: timing log score `−0.018` with the interval
`[−0.191, +0.225]`, as the power table said it would most likely not.
Report only: the shared form's timing error falls by `0.0037` seconds
with the interval clear of zero, its velocity log-score penalty from
the first study is gone (`−0.016`, interval covering zero), though the
coverage of `0.847` is still below nominal, and
the per-channel form reproduces its development picture (articulation
`−0.051` log score and `−0.023` error, velocity `−0.085` with coverage
`0.934`, nothing on timing). The learned weights say the same: the
shared weight above `0.1` on 37 of 58 performances, the per-channel
scale above `0.1` on articulation on 26 and on timing or velocity on
none. Say the one sentence at the bottom: articulation of a hidden
return holds across performers of the same movements; the timing log
score shows no effect, though the shared form's timing error falls by
3.7 ms with the interval clear of zero. The test is across performers, not
repertoire, and stays labelled so.


## Slide 40. Summary


Four sentences, say them exactly. Piano: beats the two-stage baseline
on accuracy, and the graph improves calibration inside the model,
confirmed once on untouched pieces.
Strings and winds: the graph improves recovery and calibration on the
estimated channels, confirmed once. Audio: the method is in place and
running at development level, the score prior measured inside it at
three spans, and one registered claim, the weak form at phrase scale,
confirmed once on the re-used pool. What the graph does: its value is
smoothing over a short time window, two beats on piano and two seconds
on strings and winds; pitch proximity adds nothing on intonation, a
development finding. Fifth, new: motif structure,
one edge family outside the time window. On development it improves
articulation accuracy on hidden returns in both of its forms, leaves the
timing log score unchanged, and a discovery model's predicted occurrences
give, in the shared form, about four tenths of the articulation gain,
with an interval that includes zero. Its one pre-specified test on
held-out performances of the same movements was run once on 18
November: articulation holds, `−0.029` with the interval excluding zero,
timing does not.

## Slide 41. Plan


Five rows, read top down. October, done: the curve studies, the bundle
channel, joint boundaries, the three spans of slide 31, the attribution
of slide 32; decided on 23 October with the professor hands-off: the
attribution is stated as a development finding, the headline models
keep their score graph (a pitch-less default would change a reported
protocol and need its own registration), writing freeze. Run once on
27 October: the two one-shots of slide 20 on the disclosed re-used
strings-and-winds pool, correlated timing noise passed (`−0.121`
calibration, `−0.011` seconds recovery, both intervals excluding zero),
the tonal metric missed its calibration primary (`−0.023`, interval
reaching zero) and kept its recovery gain (`−0.312` cents). November,
done: the motif family's two development studies and the
predicted-occurrence study, slides 36 and 37, and the thesis chapter
with its framing. Registered on 18 November: the phrase-scale waveform
claim, the weak form of slide 31 (graph beats temporal on the hidden
note), as one shot on the same re-used pool at the waveform level, its
third use, disclosed; power only 0.3 to 0.5, run anyway so the chapter
has a held-out answer; run the same day and passed, `−0.82` cents with
the interval excluding zero, the fourth row of slide 31. The same
afternoon, registered at midday and run once: the motif test of slide
39, articulation passed (`−0.029`, interval `[−0.049, −0.011]`), timing
missed as its power table said. The two sit in one row. With that, every
result line of the thesis is in.
Then: the lab seminar on 3 December with the results closed, December
for writing only, no new experiments, the progress talk on 7 January
from this deck, submission on 2 February. The transient model at the
joints and the whole-performance joint loop stay Future Work.

## Slide 42. Divider: Backup


Shown only if asked.

## Slide 43. Backup: the curve from the waveform, three notes


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

## Slide 44. Backup: the three spans, paired


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

## Slide 45. Backup: the three priors at track scale (strings and winds)


**What is on the slide.** The track-scale version of the three-prior
question, on the tracked six-channel bundle of the strings-and-winds
chapter: 78 development tracks, 153 track-and-seed pairs (148 for
vibrato), 30 percent of
notes hidden per track, the standard development evaluation. Every arm,
including the no-graph ablation, carries the per-piece intercept (a
column of ones in the feature kernel), so every contrast here is
already beyond the common offset that dominated the chain-scale
waveform tests. Three columns of paired differences with 95 percent
bootstrap intervals, negative favouring the first-named arm.

**The three arms.** Score graph: the thesis model's graph, notes linked
by closeness in time (2-second window, onsets annotated in seconds)
and pitch (4-semitone window).
Time chain: consecutive notes linked with unit weight, nothing else.
Time Gaussian: the score graph with the pitch term removed, the same
2-second window; it is the score graph minus exactly the thing that makes
it a score graph.

**How to read the table.** First column: the score graph beats the
time chain, significantly, on intonation, vibrato extent, vibrato rate
and loudness. Second column: the score graph equals the pitch-less
graph on intonation (−0.02 cents, interval across zero) and is slightly
worse on the two vibrato channels. Third column: the pitch-less graph
against the chain reproduces the first column. So the whole advantage
over the chain is the time window, not pitch proximity.

**The sentence.** "At track scale on both corpora, the structure that
helps is smoothing over a short time window. The waveform spans never
removed the pitch term, so they say nothing about it. Linking notes by pitch adds
nothing on intonation, a little on loudness, and costs a little on
vibrato." This reframes the confirmed
strings-and-winds result without retracting it: the graph term still
carries the recovery gain and the calibration; what we now know is which
structure inside it does the work.

**The piano bullet.** The same test on piano: the headline model of
slide 13 with the pitch term removed from its graph (the 2-beat window
kept, which on piano links the notes of a chord at full weight), paired
per piece against the headline on the 30 development pieces, 4 seeds.
Pooled RMSE −0.008 [−0.016, −0.001], log score −0.019 [−0.030, −0.008]:
removing the pitch term helps slightly, mostly in articulation. The
graph's whole value on that set is +0.017 RMSE, so the pitch-less graph
is worth about half again as much as the score graph.

**What it does not change, and say this if asked.** The confirmed piano
headline (slide 13) and the confirmed strings-and-winds results (slide 19) compare the full model with its alternatives; they stand. What is
narrower now is the attribution: the graph term works through temporal
smoothing, not through the pitch coupling. Changing the default graph
would be a change to a reported protocol and would need its own
registration. Record: `results/piano_timegauss_dev.md`.

---

## If stopped on one slide

The four most likely probes are 20 (why a spent pool again), 23 (how
the timbre is removed), 31 (what the three spans say) and 32 (what the
graph is for). One-line answers: "disclosed, replication-grade, one
pass and one miss, both verbatim," "the amplitudes are linear, so we
integrate them out exactly," "the score prior gives a missing note its
context's offset, beats a time-only prior at phrase scale, and so far
nothing more," and "the graph's value is its short time window, two
beats on piano, two seconds on strings and winds; the pitch term adds
nothing on intonation."
