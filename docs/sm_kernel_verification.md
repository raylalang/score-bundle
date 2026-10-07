# Is the estimator-v2 kernel "SM-GP"? Verification (2026-10-06)

Prompted by the supervisor's correction at the 2026-09 meeting ("the
method you tried was not SM-GP") and checked against the implementation
(`src/score_bundle/phase2/sm_estimator.py`), the design note
(`docs/sm_estimator_note.tex`), and Wilson & Adams (ICML 2013).

## What the code implements (verified line by line)

Kernel (`sm_kernel`):

    k(dt) = w1 exp(-2 pi^2 v1 dt^2) cos(2 pi mu1 dt)
          + w2 exp(-2 pi^2 v2 dt^2)

Evidence (`sm_log_evidence`), flat prior on the constant centre c:

    log ∫ N(y; c 1, K) dc = -(n-1)/2 log 2pi - 1/2 log|K|
                            - 1/2 r' K^-1 r - 1/2 log(1' K^-1 1)

with K = k + s2 I, r = y - c_hat 1, c_hat the GLS mean. Both match the
note's equations (2) and (6) exactly, and the diagonal is k(0) + s2 as
it should be. Read-outs and their variances are as the note's section 4
states. **No implementation or algebra error found.**

## The verdict

**The kernel IS a member of the Wilson–Adams spectral-mixture family**
(the Q = 2 special case with mu2 = 0: the spectral density is a Gaussian
at the vibrato rate plus a Gaussian at zero frequency). Every equation
we wrote under the SM citation is correct.

**The method is NOT "the SM method", and the supervisor's correction is
right.** Wilson & Adams' contribution is the *mixture as a flexible,
learned spectral density*: many components (Q ~ 10), means free over the
whole spectrum, initialized from the empirical spectrum, used for
pattern discovery and extrapolation — the mixture as a universal
approximator of stationary kernels. What we built uses none of that:

- Q = 2, with the roles fixed by hand (one vibrato peak, one drift);
- the mu2 = 0 component is simply an RBF (squared-exponential) kernel,
  which predates the SM paper by decades;
- the rate is confined to a hand-set band (2.5–9 Hz) and the
  coherence bound sqrt(v1) <= mu1/8 forbids exactly the broadband
  spectra the SM family exists to reach;
- nothing is discovered from the spectrum; the structure is imposed.

Structurally this is a **hand-designed two-component kernel** of the
"oscillation plus smooth drift" type (the design pattern of the
Rasmussen–Williams CO2 example), written in the SM component form.

## Naming decision (forward-facing materials; revised 2026-10-07)

**No coined names** (Ray's standing rule). The first proposal here was
"quasi-periodic GP (QP-GP)" -- retracted: "QP-GP" is not a literature
term, and "quasi-periodic kernel" in the GP literature usually means a
periodic kernel damped by an SE envelope (MacKay's ExpSine times SE;
Solin and Sarkka's state-space form), whereas ours is a pure cosine
times SE. The accurate, source-grounded phrasing, used everywhere:

- running text: **"the GP estimator"** (vs "the sine fit");
- at first use: *"a Gaussian process with a two-component kernel, a
  damped cosine at the vibrato rate plus a slow drift -- the Q = 2,
  mu2 = 0 special case of the spectral mixture kernel (Wilson and Adams
  2013)"*. "Spectral mixture (SM) kernel" is the paper's own name for
  the KERNEL (verified below), so the special-case attribution is exact.

## Verified against the PDFs (2026-10-07)

- Wilson & Adams 2013 (`related_works/gp-kernel-pattern-discovery.pdf`):
  their Eq. (11), one component, is exp(-2 pi^2 tau^2 sigma^2)
  cos(2 pi tau mu) and Eq. (12) the Q-component sum -- our kernel and
  the note's equations match symbol for symbol (their sigma^2 = our
  v_q). The paper states "Henceforth, we refer to the kernel in
  Eq. (12) as a spectral mixture (SM) kernel", i.e. the name attaches
  to the kernel; the method of the paper is many components (Q = 10 in
  their CO2 experiment), free means, learned spectral density,
  "discover patterns without encoding them a priori" -- confirming the
  method-level distinction above.
- Remes, Heinonen, Kaski 2017
  (`related_works/non-stationary-spectral-kernels.pdf`): title
  "Non-Stationary Spectral Kernels"; the generalised spectral density /
  GSM framing is theirs.
- Alvarado & Stowell 2016 (`related_works/gp-music-audio-model.pdf`):
  "Gaussian Processes for Music Audio Modelling and Content Analysis",
  Queen Mary University of London, 2016 -- author/year attributions
  correct.

Applied to: `docs/slides/deck_week.tex` (the living deck), the two
estimator figures (legends regenerated), `docs/sm_estimator_note.tex`,
and a naming banner in `docs/meeting_qa_prep.md`. Per the repo
convention, code identifiers and the dated results records
(`results/sm_estimator_dev.md` etc.) keep their historical names; the
module docstring gains a pointer to this memo.

## What does NOT change

The study's measurements and verdicts are about the model and read-outs,
not the name: the head-to-head result, the stratified addendum, the
curve-level evidence, and the interface analysis (lossless vs lossy
round trip; 5 vs 7 numbers per note) all stand as recorded.
