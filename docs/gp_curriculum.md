# A guided GP curriculum — from zero to our model

For Ray, 2026-09-03. Purpose: a real grasp of Gaussian processes and of
every GP idea our thesis uses, built in six stages. Each stage: a short
reading, the idea in plain words, where it lives in OUR model, and
self-checks. Answers at the very bottom — try first, then check.
Pace: one stage per sitting is plenty. Stages 1–2 are the foundation;
if the meeting is soon, do those two and skim stage 5.

Readings are in `related_works/` and `docs/thesis/draft.tex` (printed
page numbers; your PDF viewer's counter runs 2 ahead).

---

## Stage 1 — GP regression from zero

**Read:** Alvarado & Stowell (`gp-music-audio-model.pdf`) §2–2.1 (their
pp. 2–4). It is a gentle, self-contained tutorial. Then our draft §3.6
up to the posterior equation (pp. 12–13).

**The idea in plain words.** Ordinary regression picks a formula
(a line, a sine) and fits its parameters. GP regression skips the
formula: you put a probability distribution directly over FUNCTIONS.
Before seeing data, "smooth functions with roughly this wiggliness are
likely" — that's the prior, and it is fully specified by one object,
the kernel k(x, x′): how similar the function's values at two inputs
should be. Seeing noisy data then just applies Bayes' rule, and because
everything is Gaussian, the answer is a formula, not an optimization:
a posterior mean (best guess curve) and a posterior variance (honest
error bars) at every input. The third object, the marginal likelihood
(evidence), scores how well a kernel explains the data with all
functions integrated out — it automatically punishes both too-rigid and
too-flexible kernels, which is why hyperparameters can be learned from
it without a validation set.

**Where it lives in our model.** Everything. The whole thesis is one GP
regression: inputs = notes (not time points!), outputs = expressive
values, kernel = built from the score, posterior = the transcription,
error bars = the calibration story, marginal likelihood = how every
hyperparameter is fit, per piece. When the draft says "exact per-piece
evidence", it means exactly the marginal likelihood you just read about.

**Self-checks.**
1. What plays the role of "the parameters" in GP regression, and what
   happened to them?
2. Two inputs have kernel value near zero. What does the posterior at
   one tell you about the other?
3. Why does maximizing the marginal likelihood not overfit the way
   maximizing data fit would?
4. In OUR model, what exactly is one "input point"? What is one output?

---

## Stage 2 — Kernels: what they encode, and the evidence as judge

**Read:** Wilson & Adams (`gp-kernel-pattern-discovery.pdf`) §1–2 and
the first half of §3 (up to eq. 11). Then our draft §3.3 (the graph
kernel, pp. 10–11).

**The idea in plain words.** The kernel is where all modelling
assumptions live. SE kernel = very smooth functions; Matérn = rougher;
periodic = repeating. Each has hyperparameters (lengthscale = how far
correlation reaches; scale = how big the wiggles are). The marginal
likelihood compares kernels and settles hyperparameters; components a
kernel doesn't need get switched off (their weights collapse — "ARD").
Wilson & Adams' complaint: hand-picked kernels can only interpolate
the patterns a human already saw; a kernel family rich enough to LEARN
the pattern can extrapolate it.

**Where it lives in our model.** Our kernel's "input distance" is not
time but position in the SCORE GRAPH: two notes are similar if they are
close in beat/pitch/voice. The spectral profile g(ν) plays the
lengthscale role — it says how quickly similarity decays across the
graph (draft: "additive" profile ≡ graph Matérn). And our per-piece
evidence does the ARD job: on some pieces it switches the embedding
kernel off entirely, on others it leans on it — that is the measured
"per-piece dominance" result.

**Self-checks.**
1. What goes wrong if a lengthscale is much too long? Much too short?
   How would each show up in the error bars?
2. In one sentence: what is our replacement for "distance between
   inputs", and why is that the thesis's central bet?
3. What does it mean, mechanically, that the evidence "switched a
   kernel off" for a piece?

---

## Stage 3 — The spectral view and the spectral mixture kernel

**Read:** Wilson & Adams §3 fully (eq. 12 and around) and §4.1 (the CO2
example). Optional skim: §4.2 (recovering standard kernels).

**The idea in plain words.** Bochner's theorem: every stationary kernel
is equivalent to a distribution over FREQUENCIES (its spectral density).
"Choose a kernel" = "choose which frequencies the function contains and
how coherent they stay." Wilson & Adams make the spectrum itself
learnable — a mixture of Q Gaussian bumps — giving the spectral mixture
(SM) kernel: a sum of damped cosines, each with a frequency μ_q, a
coherence decay v_q, and a weight w_q. Enough bumps approximate ANY
stationary kernel. The evidence prunes unused bumps.

**Where it lives in our model.** Twice. (i) Our graph kernel is already
"spectral": g(ν) reweights the graph Laplacian's eigenvalues exactly the
way a spectral density reweights frequencies — same mathematics, graph
spectrum instead of Fourier spectrum. (ii) On a within-note pitch curve,
vibrato is one SM component (frequency ≈ f_vib, finite coherence) and
drift is a second component near zero frequency — so "vibrato + drift"
is a two-component SM prior. Our parametric sine fit is the rigid
special case of that prior.

**Self-checks.**
1. Sketch (mentally) the spectral density of: pure noise; a slow trend;
   steady 5.5 Hz vibrato; vibrato whose phase wanders. Which SM
   parameters differ between the last two?
2. Why can the SM kernel extrapolate the CO2 curve where SE cannot?
3. What is the "spectrum" that our graph kernel g(ν) operates on, and
   what replaces "frequency" there?

---

## Stage 4 — Non-stationarity: the generalized spectral mixture

**Read:** Remes et al. (`non-stationary-spectral-kernels.pdf`) §1, §2.2
(the GSM kernel), and §5.1 (their first experiment). Skim the rest.

**The idea in plain words.** Stationary = the same statistical behavior
everywhere. Real signals change: a vibrato speeds up, its extent grows.
GSM makes the SM parameters FUNCTIONS of the input — frequency μ_i(x),
weight w_i(x), coherence ℓ_i(x) — each itself given a GP prior. SM is
the special case where those functions are constants. Their first
experiment recovers an oscillation whose frequency changes over time —
exactly the shape of our problem. The cost: hyperparameters became
latent functions, so inference is a heavier MAP optimization, and on
short data a free frequency function can imitate a drift (and vice
versa) — flexibility buys an identifiability problem.

**Where it lives in our model.** This is the principled version of what
our drift study measured: within one note, rate and extent drift
(two-thirds of notes have significant slopes; two independent curves
agree on direction 97% of the time), and our gated sine fit is the
degenerate limit — one component, constant functions, hard onset gate.
Our hard identifiability rules (min samples, min cycles) are the manual
version of what GSM would need priors to do.

**Self-checks.**
1. State precisely which constants of the SM kernel the GSM promotes to
   functions, and what prior those functions get.
2. Our drift study found within-note slopes are ~uncorrelated across
   notes. Does that argue for or against putting a GSM prior ACROSS
   notes? Within a note?
3. What is the identifiability trap on a 0.3-second note, in one
   sentence?

---

## Stage 5 — The music-audio assembly (nearest neighbour to Phase 3)

**Read:** Alvarado & Stowell §2.2–2.2.4 (change-windows, exponentiated-
cosine kernels) and §3 (the two tasks). Then our draft §3.10 opening
(the waveform likelihood, pp. 26–27).

**The idea in plain words.** A recording = a sum of per-note GPs, each
gated by a smooth on/off window (sigmoid rise at onset, fall at
offset). Each note's kernel is an exponentiated cosine: its spectrum is
a genuine HARMONIC comb — fundamental plus partials — controlled by two
numbers (fundamental ω, richness z); multiplying by an SE term lets the
envelope move. Pitch estimation = slide ω and read the marginal
likelihood. They must hand-specify the note windows.

**Where it lives in our model — and where we deliberately differ.**
Their windows are our score: onsets/durations come free in the
score-informed setting. Their marglik-over-ω is procedurally our
Phase-3 grid-over-c on the collapsed likelihood — independent
validation of the inference pattern. The deliberate difference: they
put the structure IN the kernel and marginalize the note away; we keep
a deterministic harmonic basis with explicit, named variables (c, γ,
f_vib, ...) because those variables ARE the object of study. Both are
valid designs; ours is chosen, not naive — be ready to say why.

**Self-checks.**
1. Why does the exponentiated-cosine kernel have harmonics where the
   plain periodic (exp-sine-squared) kernel's samples repeat but the SM
   single component doesn't?
2. What information do they lack that our setting supplies, and what do
   we lose by keeping explicit variables instead of their kernel route?
3. In our Phase-3 study, what exactly was gridded, and what was
   marginalized in closed form?

---

## Stage 6 — Multi-output and graphs: our native ground

**Read:** Our draft §3.2–3.6 straight through (pp. 9–14), slowly. Then
the Borovitskiy graph-Matérn paper (`related_works/graph-matern-gp.pdf`)
§1–3 if appetite remains.

**The idea in plain words.** Two last ingredients. (i) Multi-output:
we predict THREE (Phase 1) or SIX (Phase 2) channels per note; the ICM
construction says "one shared graph structure, coupled across channels
by a small matrix B" — B's off-diagonals let timing information help
velocity, etc. Measured: coupling earns its keep on velocity only.
(ii) The graph: replacing "time" by the score graph makes the prior a
graph GP; the additive profile is the graph Matérn. Side information
(score features, embeddings) enters as LINEAR kernels — mathematically
identical to a Bayesian linear regression mean, marginalized.

**The one-sheet exercise (the real test).** Close everything. On one
sheet, redraw the whole model from memory: score → graph → Laplacian
spectrum → g(ν) → ⊗ B → + feature kernels → + noise → posterior +
evidence. Annotate each arrow with the stage (1–6) it came from. If a
box feels vague, that stage needs a second pass.

**Self-checks.**
1. Write the model's covariance in words: "(coupling) ⊗ (graph kernel)
   + Σ (feature kernels) + (noise)". What does each term contribute to
   a held-out note's posterior mean? To its variance?
2. Why must g(0) = 1 (shape normalization) for B to be interpretable?
3. Which single measured fact justifies the GRAPH term's existence if
   you could keep only one? (Hint: it is a calibration fact, not an
   accuracy fact.)

---

## Answers

**Stage 1.** (1) The function itself is the parameter — infinitely many
values — and it is never optimized, only integrated over; the posterior
is over functions directly. (2) Almost nothing: near-zero kernel value
= the prior treats the values as unrelated, so data at one barely moves
the posterior at the other. (3) The evidence integrates over ALL
functions the kernel allows: a too-flexible kernel spreads its
probability over many datasets and scores badly on any particular one
(automatic Occam); data fit alone always prefers more flexibility.
(4) One input point = one NOTE of the score (its graph position +
features); one output = that note's value on one expressive channel
(e.g., its timing residual τ_i).

**Stage 2.** (1) Too long: everything shrinks toward one smooth trend —
underfit, error bars too NARROW where the function actually varies
(overconfident). Too short: notes stop sharing information — the
posterior reverts to the prior between observations, error bars fat,
mean ragged. (2) Distance = proximity in the score graph (beat, pitch,
voice); the bet is that expressive behavior varies smoothly over the
score's own structure. (3) The evidence drove that kernel's scale
(weight) to ~zero, so its term contributes ~nothing to the covariance —
the fit says this piece's data doesn't support that similarity.

**Stage 3.** (1) Noise: flat spectrum. Trend: mass near zero frequency.
Steady vibrato: one sharp bump at 5.5 Hz (small v). Wandering phase:
same bump location but WIDER (larger v = shorter coherence). (2) The SM
kernel puts probability mass at the yearly frequency + trend, so those
patterns persist beyond the data; SE has all mass at frequency ≈ 0 —
nothing to continue. (3) The eigenvalues ν of the score-graph Laplacian;
"frequency" becomes "graph roughness" — how fast a pattern alternates
across neighboring notes.

**Stage 4.** (1) Frequencies μ_i, log-weights log w_i, log-lengthscales
log ℓ_i each become a function of the input with its own GP prior
(logit for μ to cap at Nyquist). (2) Across notes: AGAINST — the slopes
are ~white across notes, so a smooth cross-note prior has nothing to
model (that is why we did not add a drift channel). Within a note: FOR —
drift/rate/extent vary smoothly there; that is where GSM belongs.
(3) With ~1.5 vibrato cycles, a slowly rising frequency function and a
center drift produce nearly the same curve, so the data cannot separate
them — the prior (or a hard rule) must.

**Stage 5.** (1) exp of a cosine contains ALL powers of the cosine
(its series expansion), i.e. energy at every integer multiple of ω —
a comb; one SM component is a single bump (one frequency, no partials).
(2) They lack note segmentation (hand-specified windows); the score
gives it to us. We lose the nonparametric flexibility of letting the
kernel absorb everything — in exchange the posterior is OVER the named
quantities the thesis is about. (3) Gridded: the intonation centre c
(and rate, in the pilot); marginalized exactly: the per-chunk harmonic
amplitudes (and, in the last variant, smooth pitch-curve deviations).

**Stage 6.** (1) Coupling ⊗ graph: a held-out note's mean borrows from
its graph neighbors on ALL channels via B; its variance shrinks with
how well-connected and well-observed its neighborhood is. Feature
kernels: a learned linear prediction from score features/embeddings
(the "mean" work). Noise: per-channel/per-cell trust in each
observation; enters predictive variance. (2) Because cB ⊗ K/c is the
same covariance — without pinning the kernel's scale (g(0)=1), B's
entries are not identified and could not be read as channel
variances/correlations. (3) Coverage: with the graph, held-out coverage
sits at nominal (0.925 / 0.88–0.91 confirmed); without it, intervals
stop being honest — the graph's confirmed contribution is calibration.
