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

Structurally this is a **hand-designed quasi-periodic kernel** — the
"periodic-ish component plus smooth drift" construction of the
Rasmussen–Williams CO2 example and the quasi-periodic state-space
models of Solin & Särkkä — written in the SM component form.

## Naming decision (forward-facing materials)

One term per concept, from now on: **quasi-periodic GP (QP-GP)**, with
the one-line attribution at first use: *"its kernel is the Q = 2,
mu2 = 0 special case of the spectral-mixture family (Wilson and Adams
2013)"*. That keeps the citation (the component form and the Fact-1/
Fact-2 spectral reading remain correct and useful) without claiming the
method.

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
