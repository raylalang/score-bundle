# The prior-value experiment, tuning-scalar follow-up: free per-chain tuning scalar (sd 50 cents) in all arms (DEV, exploratory, no claims)

Three cross-note priors over per-note intonation constants at matched marginal
variance, identical likelihood/data/optimizer, boundaries fixed at annotations.
Conditions: the middle eligible note's audio MASKED from the likelihood (maximal
ambiguity) vs clean. Committed hypotheses and falsification rule in the script
docstring (`scripts/eval_prior_value_dev.py`). Scored against the quasi-truth centre.

47 cells (chains), tracks: (1, 1)(vn), (1, 2)(vc), (3, 2)(cl), (7, 2)(tbn); 25 cells have a recurring pitch.

| condition | arm | median abs err (cents) | cov@90 | median sd |
|---|---|---|---|---|
| masked | indep | 28.40 | 0.49 | 12.42 |
| masked | temporal | 18.34 | 0.51 | 10.61 |
| masked | graph | 21.80 | 0.51 | 10.60 |
| clean | indep | 2.35 | 0.91 | 6.87 |
| clean | temporal | 2.24 | 0.91 | 6.45 |
| clean | graph | 2.22 | 0.89 | 6.48 |
| masked | (deadpan reference) | 14.42 | --- | --- |

## Paired contrasts, masked condition (negative favours the first arm)

- graph vs indep: -4.84 [-6.38, -3.33]* (better on 87%, n=47)
- temporal vs indep: -4.75 [-6.69, -2.98]* (better on 77%, n=47)
- graph vs temporal: -0.09 [-0.65, +0.50]  (better on 53%, n=47)
- graph vs temporal, recurring-pitch cells only: -0.57 [-1.06, -0.08]* (better on 60%, n=25)

## Paired contrasts, clean condition (H2: expect ties)

- graph vs indep: -0.00 [-0.01, +0.00]  (better on 62%, n=47)
- temporal vs indep: -0.01 [-0.02, +0.00]  (better on 60%, n=47)

## Cross-run, masked condition: this run minus the first run, per arm (negative = the scalar helped; exactly paired cells)

- indep: +0.42 [-1.58, +2.46]  (better on 51%, n=47); first-run cov@90 0.38
- temporal: +0.06 [-0.60, +0.72]  (better on 51%, n=47); first-run cov@90 0.45
- graph: +0.21 [-0.49, +0.95]  (better on 53%, n=47); first-run cov@90 0.49

First-run graph vs indep, masked, same cells: -4.63 [-6.43, -2.96]* (better on 85%, n=47)

wall per fit incl. Laplace: median 173 s

## Reading (2026-10-15)

Measured as committed; superseded on the same day by the leakage
diagnosis recorded as the erratum in `results/prior_value_dev.md`.

1. **H3 as literally stated "survives"** (graph vs indep −4.84*, better
   on 87%; recurring-pitch stratum −0.57*), **but the independent arm
   did not move to the chain mean**: its masked error rose to 28.4 cents
   median (first run 21.2; paired mean change +0.42, ns) and stayed far
   above the deadpan 14.4. Under a free common mode the independent arm
   must do at least as well as the chain mean; it did not, so something
   other than the prior was setting the hidden note's value.
2. **H4 fails**: masked coverage 0.49–0.51 for every arm, barely moved
   from 0.38–0.49. The "miscentred zero-mean prior" diagnosis was wrong.
3. Both facts are explained by the window-leakage finding: the hidden
   note's estimate is a leakage-driven likelihood optimum in every arm
   and in both runs (zero correlation with the truth, ±85-cent spread).
   The scalar changes the prior, and the prior was not what decided the
   estimate. H5 holds (clean rows tie, coverage at nominal), as it must
   since no mask is involved.
4. This run therefore answers the attribution question only after the
   protocol fix: see `results/prior_value_v2_tuning_dev.md`.
