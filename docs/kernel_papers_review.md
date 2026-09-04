# Review: the two kernel pointers, three papers — and why the spectral mixture is the working choice

2026-08-28, revised 2026-09-04 to answer three questions directly: why
this equation, where exactly it enters our model, and what the other two
papers are waiting on. Papers in `related_works/`:
`gp-kernel-pattern-discovery.pdf` (Wilson & Adams, ICML 2013),
`non-stationary-spectral-kernels.pdf` (Remes, Heinonen & Kaski, NeurIPS
2017), `gp-music-audio-model.pdf` (Alvarado & Stowell, arXiv 1606.01039,
2016). Prompted by the supervisor's pointers "periodic kernel Gaussian
process" and "generalized spectral mixture", in the context of the
vibrato/pitch-curve model (draft eq:cents-curve/eq:vibrato) and the
Phase-3 likelihood.

## 0. The decision in brief

The working choice is the **spectral mixture (SM) kernel** as a prior on
a note's cents curve, tested against the current sine estimator on
development data (the kill-cheap test of `docs/gp_everywhere_memo.md`).
The other two papers are not dropped: the GSM paper is the designated
*second* rung, gated on the SM test passing, and the Alvarado & Stowell
paper is the reference for the *other* slot (the Phase-3 waveform prior),
not for the estimator. Sections 1–2 give the why and the where; sections
3–5 review the papers; section 6 answers "are the other two relevant
now" explicitly.

## 1. Why this equation

**The short answer: the SM kernel is the prior whose parameters are the
quantities we want to measure.** The task is to estimate a rate and an
extent from a noisy oscillating curve. In the SM kernel a frequency, its
power, and its coherence *are* the hyperparameters, so fitting the prior
by marginal likelihood *is* the estimation. No other standard choice has
this property: SE or Matérn kernels encode only smoothness (there is no
rate to read out), the periodic kernel fixes one phase-locked oscillation
forever (the same rigidity that makes the sine fit fail), and the current
sine fit commits to a shape and needs hand rules wherever the shape
breaks. SM sits exactly at our level of knowledge — "oscillates near some
rate, with some strength, imperfect phase, plus slow drift" — assuming
nothing more, leaving nothing we need out. The rest of this section
unpacks that.

The current estimator (draft eq:vibrato) *assumes the curve's shape*: a
constant centre plus one phase-locked sinusoid switched on after a delay,

$$c_i(t) \;=\; c_i + \gamma_i \sin\!\bigl(2\pi f_i^{\mathrm{vib}}(t - t_i - d_i^{\mathrm{vib}})\bigr) + \epsilon_i(t),$$

and least-squares fits five numbers. Everything that is not that shape
becomes residual, and hand rules (min frames, min cycles, octave
threshold) protect the fit. The drift study then measured that the shape
assumption is false in a specific way: the centre moves within the note
(two-thirds of notes, median 10.5 cents), and vibrato is not perfectly
phase-stable.

The SM kernel is the natural relaxation, for four reasons.

**(a) Vibrato is a spectral object.** Bochner's theorem: every
stationary kernel is the Fourier transform of a spectral density, so
specifying a prior over curves in the *frequency* domain is fully
general. What we know about a cents curve is spectral knowledge: power
concentrated near the vibrato rate (5–8 Hz) plus power near zero
frequency (slow drift). The SM kernel writes exactly that down — the
spectral density is a Gaussian mixture, and the kernel comes out in
closed form:

$$k(\tau) \;=\; \sum_{q} w_q\, \exp\!\bigl(-2\pi^2 \tau^2 v_q\bigr)\, \cos\!\bigl(2\pi \tau \mu_q\bigr).$$

**(b) Every parameter is a musical quantity.** Each component $q$ is a
damped cosine: $\mu_q$ is its frequency (the vibrato rate), $w_q$ its
power (the extent: a coherent sinusoid of amplitude $\gamma$ has
stationary variance $\gamma^2/2$, so $\gamma \approx \sqrt{2 w_1}$), and
$v_q$ the rate at which phase coherence decays (how phase-stable the
vibrato is — a knob the sine fit does not have). Nothing is a black box.

**(c) It contains our current model as a limit.** As $v_q \to 0$ one
component becomes the periodic kernel (the supervisor's first pointer),
and the fully degenerate case — one component, phase locked, hard onset
gate — *is* the current gated sine fit. So the move relaxes exactly the
assumptions the drift study falsified, and nothing else.

**(d) It keeps the thesis's inference story.** The prior is Gaussian, the
noise is Gaussian, so the per-note fit is the same exact marginal
likelihood used everywhere else in the thesis, and the posterior
uncertainty flows into the bundle's noise rows directly (as-given
discipline) instead of via delta-method formulas. The mixture family is
also dense in stationary kernels, and the evidence prunes unused
components — the same ARD behaviour our per-piece fits already show.

## 2. How exactly it enters our model — the flow, step by step

The change is confined to the **within-note** level. Nothing above it
moves. In one line, the estimator box changes and everything around it
stays:

$$\text{tracker frames} \;\xrightarrow{\ \text{estimator}\ }\; \text{per-note cells} \pm \sigma \;\xrightarrow{\ \text{noise rows}\ }\; \text{graph GP across notes}.$$

The steps below spell out what happens inside the new estimator box for
one note $i$.

**Step 1 — inputs.** The same inputs the sine fit gets today: the
confidence-kept tracked frames of note $i$ (draft eq:framerule) — times
$t_j$ and cents values $y_j = c_i(t_j)$ for the $|J_i|$ frames (typically
30 to a few hundred at the 10 ms hop).

**Step 2 — the model for one note.** The curve is a constant centre plus
a zero-mean GP:

$$c_i(t) \;=\; c_i + g_i(t), \qquad g_i \sim \mathcal{GP}\bigl(0,\; k_{\mathrm{SM}}\bigr), \qquad y_j \;=\; c_i(t_j) + \varepsilon_j,\ \ \varepsilon_j \sim \mathcal{N}(0, \sigma^2),$$

with the two-component kernel

$$k_{\mathrm{SM}}(\tau) \;=\; \underbrace{w_1\, e^{-2\pi^2\tau^2 v_1} \cos(2\pi\tau\mu_1)}_{\text{vibrato: rate }\mu_1,\ \text{power } w_1,\ \text{coherence } v_1} \;+\; \underbrace{w_2\, e^{-2\pi^2\tau^2 v_2}}_{\text{drift: } \mu_2 = 0}.$$

**Step 3 — the fit.** Build the $|J_i| \times |J_i|$ covariance matrix
$K$ with entries $k_{\mathrm{SM}}(t_j - t_{j'}) + \sigma^2\,
\delta_{jj'}$ (plus the constant-mean term, handled by marginalizing
$c_i$ under a flat prior, exactly like the bias column of the across-note
feature kernels), and maximize the exact log marginal likelihood

$$\log p(y \mid \theta_i) \;=\; \log \mathcal{N}\bigl(y;\; 0,\; K(\theta_i)\bigr), \qquad \theta_i = (w_1, \mu_1, v_1, w_2, v_2, \sigma^2),$$

over the six hyperparameters, in log space, per note. This is the same
evidence formula as the across-note fit (draft eq:marglik), applied to
frames within one note instead of notes within one piece; at these sizes
the $O(|J_i|^3)$ cost is negligible. The one genuine numerical care is
that the evidence is multimodal in $\mu_1$ (the known SM issue), so
$\mu_1$ gets a coarse grid initialization over 3–10 Hz before the local
optimizer — the GP analogue of the sine fit's frequency initialization.

**Step 4 — the outputs and where each comes from.** Three channels, three
read-outs:

$$\hat c_i \;=\; \mathbb{E}[c_i \mid y] \quad (\text{exact: the marginalized constant mean has a closed-form Gaussian posterior}),$$

$$\hat f_i^{\mathrm{vib}} \;=\; \mu_1, \qquad \hat\gamma_i \;\approx\; \sqrt{2 w_1} \quad (\text{a sinusoid of amplitude } \gamma \text{ has stationary variance } \gamma^2/2).$$

Their uncertainties: the centre's variance is exact (from its Gaussian
posterior). Rate and extent are hyperparameter estimates, so their
variances come from the curvature of the log evidence at the optimum
(a Laplace approximation) — the honest statement is that this is the
GP analogue of the delta-method covariance the sine fit uses today, not
something categorically better. What *is* categorically better sits one
level down: given the hyperparameters, the whole curve $c_i(t)$ has an
exact GP posterior, which is what the diagnostics plot and what degrades
gracefully — on a short or weak note the evidence is nearly flat in
$\mu_1$, the Laplace variances blow up, and the note reports itself as
uninformative instead of being refused by a hand rule.

**Step 5 — into the bundle.** The three (value, variance) pairs become
the note's $c$, $\log\gamma$, $\log f^{\mathrm{vib}}$ cells and their
noise rows, as-given, exactly where the sine fit's outputs go today. The
across-note graph GP, the cell-mask machinery, and the loudness and
timing channels are untouched.

**What changes about the channels: nothing structural, two wrinkles.**
The channel set is unchanged — this is an estimator swap behind the three
pitch-derived channels. Wrinkle one: the estimands shift slightly ($\gamma$
as gated-sinusoid amplitude and $\gamma$ as component power are different
functionals of the same curve — the same estimand-gap phenomenon Phase 3
measured for $c$), which is why the kill test scores against quasi-truth
rather than against the old estimator's targets, and why v2 numbers are
not comparable to the confirmed Phase-2 numbers. Wrinkle two: the onset
delay $d_i^{\mathrm{vib}}$ cannot come from this estimator — a stationary
kernel marginalizes phase, so "the oscillation starts after a straight
beginning" is inexpressible in plain SM. That is non-stationarity, the
GSM paper's contribution (section 4). The delay channel carries no
registered claim, so during the test it keeps the gated fit (mixed
estimator provenance, stated rather than hidden), until GSM is on the
table. One door opens rather than changes: the coherence $v_1$ is a
per-note quantity no current channel captures — a potential future
channel, not part of this test.

The second slot (Slot B, Phase 3): the collapsed waveform likelihood
already contains a marginalized pitch-curve deviation prior with a crude
bump basis (draft App F). $k_{\mathrm{SM}}$ is the principled
replacement — still Gaussian, so the $O(mp^2)$ Woodbury machinery
carries over unchanged. Same kernel, different level.

Boundaries, unchanged: these kernels model within-note structure only.
The drift study showed within-note slopes are white across notes, so the
across-note graph GP keeps its role untouched; the confirmed Phase-2
result stays frozen; any adoption is post-hoc to the spent confirmation
and needs its own registered evaluation.

## 3. Wilson & Adams 2013 — the spectral mixture (SM) kernel

**Status: in use now — this is the kernel of the committed test.**

**What it is.** Bochner's theorem plus a modelling move: instead of
hand-picking a kernel, model the spectral density itself as a
$Q$-component Gaussian mixture; the transform has the closed form of
section 1. Gaussian mixtures are dense in spectral densities, so the
family approximates any stationary kernel; inference stays exact GP
regression; the marginal likelihood prunes unused components (their
$Q = 10$ runs typically keep ~7).

**What they show.** Pattern discovery and long-range extrapolation (CO2,
airline, sinc, negative-covariance AR series) where SE, Matérn, rational
quadratic and periodic kernels interpolate but cannot extrapolate; the
learned spectral density is interpretable (each peak is a discovered
periodicity).

**Limitation for us.** Stationary: one fixed rate, extent and coherence
per note. That is what the drift study found insufficient — hence paper
2 as the designated next step, not as part of the first test.

## 4. Remes, Heinonen & Kaski 2017 — the generalized spectral mixture (GSM)

**Status: second rung — relevant, deliberately deferred until the SM
test passes.**

**What it is.** The non-stationary extension via the generalised Fourier
transform (a spectral surface $S(s, s')$ instead of a density). The
practical kernel makes the SM parameters input-dependent functions, each
itself a GP — $\log w_i(x)$, $\log \ell_i(x)$,
$\operatorname{logit} \mu_i(x) \sim \mathcal{GP}$ — giving

$$k_{\mathrm{GSM}}(x, x') \;=\; \sum_{i} w_i(x)\, w_i(x')\; k_{\mathrm{Gibbs},i}(x, x')\; \cos\!\bigl(2\pi\,(\mu_i(x)\,x - \mu_i(x')\,x')\bigr),$$

a product of three positive-semidefinite factors (input-dependent
amplitude, Gibbs lengthscale, input-dependent frequency). It reduces
exactly to SM under constant functions. Inference: MAP on the
marginalized posterior with whitened gradients; Kronecker structure for
gridded inputs.

**What they show.** Their first experiment is our problem in miniature: a
simulated oscillation whose frequency changes over time, recovered where
SM and SE cannot. Then texture extrapolation and climate fields.

**Why it is not in the first test.** Three reasons. It is what supplies
the two things SM cannot express — the onset delay (an amplitude
function $w_1(t)$ rising from zero *is* the delay) and rate or extent
drifting within the note — so it only earns its complexity if the
stationary version already beats the sine fit. Its cost is real: every
hyperparameter becomes a latent function, so the closed-form simplicity
goes, and on 30–150-frame notes a free frequency function can absorb
drift and vice versa — the identifiability confound our hard rules
currently manage would have to be handled by priors on the hyper-GPs.
And the memo's caution stands: richer within-note models improved
accuracy in the Phase-3 study but did not fix cross-estimand
calibration, so flexibility is not the binding constraint there.

## 5. Alvarado & Stowell 2016 — GPs for music audio

**Status: the Slot-B reference — relevant to Phase 3, not to the
estimator test.**

**What it is.** The two ideas above, assembled for music in the waveform
domain. A recording is

$$f(t) \;=\; \sum_{m} \phi_m(t)\, f_m(t),$$

per-note sigmoid change-windows $\phi_m$ (onset and offset parameters)
gating independent per-note GPs, so the kernel is
$k_f = \sum_m \phi_m k_m \phi_m$. Each per-note kernel is the
exponentiated-cosine family:

$$k_{\mathrm{EC}}(\tau) \;=\; \sigma^2 \exp\!\bigl(z \cos(\omega \tau)\bigr)$$

has a genuinely harmonic spectrum (fundamental $\omega$ plus integer
partials, envelope set by $z$ — one parameter for spectral richness);
$k_{\mathrm{ECQ}}$ multiplies in an SE term so the amplitude envelope can
vary within the note. Pitch estimation = maximize the marginal
likelihood over each note's $\omega$; also missing-data imputation on
real polyphonic audio (8 kHz).

**Relevance to us.**
- Nearest neighbour to Phase 3 in the literature, with an instructive
  contrast: they put the harmonic structure *in the kernel* (waveform =
  one GP, stationary per note, harmonic covariance); we put it in the
  deterministic basis $\Phi(z)$ with marginalized amplitudes and keep the
  nonlinearity in explicit position variables $z$. Their route
  marginalizes everything but loses named per-note variables; ours keeps
  the named expressive channels ($c$, $\gamma$, $f$, $\tau$, ...) that
  the thesis is about — a deliberate design difference, not an omission.
- Their pitch estimation (marginal likelihood over $\omega$ per note) is
  procedurally our Phase-3 grid over $c$ on the collapsed likelihood —
  independent validation of the inference pattern.
- Their change-windows are manually specified — they lack a score. We
  have the score: our onsets and durations supply $\phi_m$ for free. The
  score-informed setting removes their main practical burden.
- Their $k_{\mathrm{EC}}$ is also the natural waveform-domain
  quasi-periodic prior if we ever want the harmonic content itself to be
  a GP rather than explicit amplitudes.
- Caveat shared with our explicit-$\Phi$ route: the exp-cos family
  assumes exact harmonicity, and strings are mildly inharmonic.

## 6. So are the other two papers relevant for now?

Relevant, but neither belongs in the first step, and each has a named
trigger:

- **GSM (Remes et al.)** enters if and only if the SM test passes. It is
  the principled home for the onset delay and for within-note drift of
  rate and extent — the exact structure the drift study measured — at
  the price of function-valued hyperparameters and an identifiability
  problem that priors would have to police. Testing SM first isolates the
  cheap question (does a spectral prior beat the sine fit at all?) from
  the expensive one (is input-dependence worth its inference cost?).
- **Alvarado & Stowell** is the design reference for Slot B: when the
  Phase-3 curve prior is upgraded from the bump basis to a spectral
  kernel, and if the harmonic content ever moves kernel-side. Nothing in
  the estimator test touches it.

## 7. The test that decides

Defined in `docs/gp_everywhere_memo.md` (minimal honest test): SM-prior
GP regression on the cents curves of the ~5,000 vibrato-identifiable
development notes, against the NLLS sine estimator, on (i) agreement
with quasi-truth, (ii) calibration of its own posterior, (iii) behavior
on estimator-missing notes. Development only, no claims; if it loses on
(i), the idea dies cheaply. Adoption of anything downstream needs its
own registration (the Phase-2 pool is spent).
