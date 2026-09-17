# Exploration week 2026-09: consolidated ledger

2026-09-16. One page over the seven development studies of 2026-09-04..11
(commits `a426fc8..96409b9`). Everything here is **development-only and
exploratory**: no registered artifact touched, no confirmation pool
rescored, published tables bit-unchanged. Direction decided 2026-09-16
(Ray): **adoption path** — the inference-level winners become opt-in
capabilities of the pipeline (the `fit_guarded` precedent: default off,
recommended for new runs, published paths untouched); any change to a
*reported* protocol still needs its own registration.

## The verdict table

| study | question | verdict | record |
|---|---|---|---|
| Slot-A kill test (09-04, prologue) | SM-GP replaces the sine estimator? | **Dead** by its pre-committed rule: extent +1.69\*, rate +0.16\* against; c ties. Curve-level measure favoured the GP (its one signal). | `sm_estimator_dev.md` |
| A — deploy-time t predictive | rescue the measured Gaussian timing tail? | **Win**: blow-up cell NLL 105.16 → −1.69, coverage intact, pooled NLL better on healthy cells. **Floors measured dead** (tail-shape problem, not scale). | `tail_predictive_dev.md` |
| A-full — inference-level t (EM scale mixture) | does robust *inference* buy more than rescoring? | **Yes, two things**: articulation recovery −0.018\* through the coregionalization (observed τ outliers corrupt coupled channels — first measured cross-bundle robustness effect) and coverage to nominal (0.919); τ t-NLL −0.30\*. Cost: must score likelihood-consistently. | `t_noise_em_dev.md` |
| B — AR(1) τ noise row | the failed C4 claim's named follow-up | **Strong win, both axes**: NLL −0.111\*, RMSE −20%\* (≈99→80 ms), ρ>0 detected on 97% of cells (median 0.45). | `corrnoise_tau_dev.md` |
| B hardening (09-11) | does B survive stress? | **Yes, three independent runs**: joint-evidence refit (ρ>0 on 100%, median 0.60) and fresh seeds 2–3 reproduce digit-for-digit. Registration-grade. | same, hardening section |
| C — completion fallback | close the adaptation boundary? | **Closed by the combination**: low-rank Mahalanobis coverage test + 3σ disagreement guard → worst cell 480,266 → ≤0.7 everywhere, GP advantage kept, interpolation untouched. **Mahalanobis alone is half the answer**; blow-ups also occur in interpolation. | `completion_fallback_dev.md` |
| D — Slot B (SM prior in the Phase-3 rig) | the kill test's surviving direction | **No gain**: median tie (2.25 vs 2.29 cents), paired +0.112\* against, harm in winds (scaffold-pinned band); coverage stays at the estimand-gap floor under both priors. | `phase3_smprior_dev.md` |
| E — learned spectral filter | "we only used basic kernels" | **Closed**: the evidence switches a free bump off on 90% of cells; the freedom mildly hurts (+0.009\*). Leverage is in the graph, not the filter. | `spectral_bump_dev.md` |

## The shape of the week

The reviewed spectral kernels (the supervisor's two pointers) are now
measured at **both** of the slots the review named — Slot A dead, Slot B
no-gain, and even a learned filter on the graph spectrum is declined by
the evidence. Their contribution to this pipeline is **conceptual**
(curve-level scoring, the coherence bound, the estimand lessons), not
numerical. Meanwhile the thesis's **own documented limitations each
converted into a measured win**: the timing tail (A, both levels), the
failed timing-calibration claim (B, hardened), the adaptation boundary
(C).

## Three durable lessons (each hit more than once)

1. **Process parameters are not the channels' estimands.** Ensemble scale
   ≠ realized amplitude (χ²₂, ~17% median bias); process constant ≠
   realized centre; band centre ≠ realized rate. Read-outs must target
   realized quantities.
2. **Tail problems are tail-shape problems.** Floors and variance
   inflation do not touch a quadratic-tail failure; changing the
   predictive's tail does — and robustifying the *fit* additionally
   protects coupled channels.
3. **Written follow-ups are cheap to measure.** Both of the week's
   strongest results (B, C) were sentences already sitting in the thesis
   as future work.

## Two measured-wrong predictions (to be corrected in the draft)

- "Deploy-time predictive floors are the cheap engineering alternative"
  — floors are measured ineffective (`tail_predictive_dev.md`).
- "a Mahalanobis coverage test would do" — it half-does; the disagreement
  guard is the load-bearing half (`completion_fallback_dev.md`).

## Statistical hardening (2026-09, BH pass over the whole week)

`scripts/week_bh_recheck.py` (offline, per-piece deltas — URMP contrasts
clustered at composition level by construction — Benjamini–Hochberg
q = 0.05 over the 14-contrast family, plus BCa/Wilcoxon/sign per
contrast; log: `logs/week_bh_recheck.log`). Three refinements:

1. **Every main claim survives BH**: all six correlated-τ contrasts
   (three runs × both axes), both t-EM contrasts, the learned-filter
   RMSE harm, and the kill test's adverse extent/rate stars.
2. **The Slot-A kill hardens**: even the intonation contrast, a tie at
   cell level, is adverse-starred per piece (Wilcoxon p = 0.0003,
   6/31 pieces negative).
3. **The Slot-B harm downgrades**: the "+0.112\*" adverse mean does not
   survive piece-level clustering (5 composition clusters only;
   Wilcoxon p = 0.63; n.s. after BH). The Slot-B verdict is therefore
   **no gain** — not measured harm — which changes no decision.

## What follows (decided 2026-09-16)

Adoption engineering (opt-in core support for the correlated τ noise
block, the t-EM robust fit, the t predictive and completion-guard
helpers), the draft's Future-Work passages updated from "measured needs"
to measured answers, and the meeting asks: the corpus/pool decision for a
future preregistered correlated-τ claim, and blessing of the opt-in
adoption defaults.

## Figures (added 2026-09-17)

One per finding, house style, from the committed pickles only
(`scripts/make_week_figures.py`): `corrnoise_tau_dev.png`,
`tail_predictive_dev.png`, `t_noise_em_dev.png`,
`completion_fallback_dev.png`, `kernels_closed_dev.png` (all under
`docs/thesis/figures/`). The corrnoise figure is on the meeting deck;
the rest are backup frames.
