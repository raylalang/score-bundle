# Phase-2 synthetic pilot — machinery built and validated (2026-07-16)

> **Status: the first Phase-2 result.** Everything here is synthetic
> (ground truth known, sampled from the model's own generative process) and
> deterministic; no real data, no confirmation contact. What it establishes:
> the two structural blockers the thesis named are now **implemented, exact,
> and validated end to end**, and one open design decision now has a measured
> answer. What it does not establish: anything about real audio — the tracker,
> alignment, and model misspecification questions all remain.

## What was built

1. **Per-(note, channel) cell masks** in `gp.MultiOutputGraphGP` — the
   thesis's declared blocker ("the conjugate posterior assumes per-note
   masks"). A 2-D boolean mask now selects observed cells; exactness is
   unit-pinned against brute-force Gaussian conditioning, the row-constant
   case reproduces the published 1-D path to 1e-8, and heteroscedastic
   per-cell noise (`noise_scale`, estimator variances) is honored exactly
   (`tests/test_gp_cellmask.py`). The published 1-D code path is untouched.
2. **The specified vibrato estimator** (`phase2.intonation.fit_vibrato_note`)
   — the joint per-note NLLS fit of draft eq:vibrato (grid + closed-form
   linear solve + parabolic rate refinement), with Gauss–Newton parameter
   covariances and the identifiability rule. Tests pin recovery, the
   variance calibration, the short-note rule, and the thesis's own point
   that the vibrato-free centre is *not* the curve mean
   (`tests/test_phase2_estimator.py`).

## The pilot

20 synthetic monophonic pieces × 120 notes; truth `y* = [c, log γ, log f]`
sampled from a graph GP (additive kernel, `s=3`, coupled `B`); cents curves
synthesized per eq:vibrato (σ = 6 cents, onset delays present but unmodelled);
the estimator produces targets + variances, its identifiability rule producing
**926 genuinely missing vibrato cells on visible notes**; 30% of notes fully
hidden. Fits: cell-masked heteroscedastic GP (learned noise scale vs
estimator-variances-as-given) and the no-graph ablation; scored per channel
against the known truth (RMSE / latent-interval coverage@90).
Runner: `scripts/eval_phase2_synthetic.py`; log: `logs/phase2_synthetic.log`.

## Results (RMSE / cov@90, per channel)

**Q1 — fully hidden notes (imputation):**

| system | c (cents) | log γ | log f |
|---|---|---|---|
| graph GP (as-given noise) | **3.51 / 0.89** | **0.094 / 0.86** | **0.026 / 0.90** |
| graph GP (learned scale) | 3.52 / 0.87 | 0.099 / 0.79 | 0.028 / 0.89 |
| no-graph ablation | 4.11 / 0.88 | 0.103 / 0.85 | 0.030 / 0.89 |
| estimator (oracle: sees the hidden curve) | 1.34 / 0.93 | 0.041 / 0.90 | 0.007 / 0.93 |

**Q2 — estimator-missing vibrato cells on visible notes** (the new
capability; the estimator has no value here by construction):

| system | log γ | log f |
|---|---|---|
| graph GP (as-given) | **0.085 / 0.87** | **0.025 / 0.93** |
| graph GP (learned scale) | 0.088 / 0.81 | 0.026 / 0.90 |
| no-graph ablation | 0.096 / 0.86 | 0.028 / 0.91 |

(prior marginal scales: log γ 0.35, log f 0.10 — the model recovers missing
vibrato parameters at ~4× better than the prior, from the graph + coupling.)

**Q3 — observed cells (denoising vs the raw estimator):**

| system | c (cents) | log γ | log f |
|---|---|---|---|
| graph GP (as-given) | **1.41 / 0.90** | **0.039 / 0.86** | 0.008 / 0.87 |
| graph GP (learned scale) | 1.53 / 0.93 | 0.056 / 0.76 | 0.008 / 0.93 |
| raw estimator | 1.63 / 0.91 | 0.042 / 0.90 | 0.008 / 0.88 |

## Findings

1. **The pipeline works end to end**: estimator → per-cell variances → cell
   masks → exact heteroscedastic posterior, with the graph beating the
   no-graph ablation on every channel of every question.
2. **The open noise decision now has a measured answer, and it is the
   opposite of the draft's guess.** The draft called the learned per-channel
   scale "the safer default"; on calibrated estimator variances it is the
   worse choice — it over-shrinks the γ channel (coverage 0.76–0.81 vs
   0.86–0.87 as-given) and loses RMSE across the board. As-given even beats
   the oracle estimator at γ denoising (0.039 vs 0.042). Correct scoping:
   *as given wins when the estimator's variances are honest; the learned
   scale remains insurance for a mis-calibrated real-world tracker* — that is
   now the documented default order, reversed from before.
3. **Missing-cell recovery is real**: vibrato parameters of notes too short
   to estimate are recovered at far better than prior scale, with ~0.87–0.93
   coverage — the capability per-note masking could not even express.
4. Residual honest gap: mild γ under-coverage (~0.86–0.87) persists even
   as-given — candidate causes are the delta-method log-transform of the
   variance at small extents and the unmodelled onset delay; both are listed
   for the real-data phase.

## What this changes in the thesis

The Phase-2 section's "must be resolved before Phase 2 is built" blocker is
resolved in code (draft updated); the noise-decision paragraph now cites this
pilot; Phase-2 status moves from "scoped" to "machinery built + synthetic
pilot passed; real data open." The Phase-2 *claims* still require real audio
and their own preregistered confirmation — nothing here touches that.
