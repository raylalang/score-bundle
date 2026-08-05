# Kernel variants across masking rates — dev results (run 2026-07-16, written up 2026-07-31)

> **Status: development-set study, two-stage regime.** Extends the kernel
> comparison (`docs/kernel_comparison_results.md`, 40% hidden) to the other
> four masking levels of the sweep protocol. Everything here is the
> **two-stage development form** (plug-in mean `μ_LM` + per-channel graph
> residual) — NOT the adopted GP-first model. Confirmation set untouched.
> Raw log: `logs/kernels_ms_report.log`; cells in `results/kernels_ms_obs*/`.

## Protocol

Same as the anchor kernel comparison, at four additional rates: strict
mask-aware `μ_LM`, 30 dev pieces × 4 seeds, per-rate leak-free masks/means
(`.cache/masksweep_inputs_obsX.pkl`), EB guard on, paired per-piece bootstrap
CIs vs the additive baseline. Kernels: additive (baseline), tonal
(pitch-metric replacement), harmonic (chord edges), harmonic_vl (chord +
voice-leading edges).

## Result — the anchor finding generalizes to every masking level

ΔRMSE / ΔNLL vs additive, paired 95% CI (`*` = CI excludes 0):

| hidden | kernel | ΔRMSE | ΔNLL |
|---|---|---|---|
| 50% | harmonic | −0.0089* | −0.0089* |
| 50% | harmonic_vl | −0.0077* | −0.0164* |
| 30% | harmonic | −0.0108* | −0.0149* |
| 30% | harmonic_vl | −0.0095* | −0.0502* |
| 20% | harmonic | −0.0119* | −0.0140* |
| 20% | harmonic_vl | −0.0108* | −0.0153* |
| 10% | harmonic | −0.0138* | −0.0187* |
| 10% | harmonic_vl | −0.0111* | −0.0185* |

Tonal replacement stays neutral-to-harmful at every rate (significantly worse
RMSE at 30/20/10%), replicating the anchor finding that *replacing* the pitch
metric with tonal distance hurts while *adding* harmonic edges helps.

The harmonic RMSE edge, if anything, grows as observation gets denser
(−0.0089 at 50% hidden → −0.0138 at 10% hidden).

## What this does and does not reopen

This is a **two-stage-regime** result at every rate. The GP-first adoption
record says harmonic edges are measured-redundant once the LM embeddings
enter the kernel as features (`c_harm_lm` 0.3561 ties `b_featlm` 0.3590 at
the 40%-hidden anchor) — but that redundancy check exists **only at the
anchor rate**. Two follow-ups, in order:

1. **Dev check:** run `c_harm_lm` vs `b_featlm` at the other rates (the
   per-rate masks and embedding dumps already exist). If the tie holds
   everywhere, the regime-scoped statement stands as-is and this doc is
   closed. The new posterior-decomposition machinery
   (`gp.posterior_components`, graph×LM cross-covariance) gives a per-note
   redundancy measure to accompany the aggregate A/B.
2. **Only if the dev check breaks the tie** at some rate would harmonic
   edges become an adoption question again — and any adoption-level change
   goes through a **second preregistered confirmation set**
   (`docs/graphgp_first_design.md`), not dev numbers.

### Outcome of the first dev check (obs 0.90, run 2026-07-31)

**The tie breaks at the densest level.** `c_harm_lm` vs `b_featlm` at 10%
hidden (30 dev pieces × 4 seeds, guard on;
`scripts/run_charmlm_obs090.sh`, `results/graphgp_charmlm_obs0.90/`,
`logs/charmlm_obs090_report.log`):

| | RMSE | NLL | cov@.9 |
|---|---|---|---|
| c_harm_lm | **0.3383** | −0.160 | 0.920 |
| b_featlm | 0.3472 | −0.418 | 0.925 |

Paired dRMSE **−0.0089 [−0.0158, −0.0027]\***. Paired dNLL +0.264
[−0.050, +0.864] ns — but 22/30 pieces are NLL-better (median −0.030); the
positive mean is one cell of the documented tail piece 28 (seed 2, cell NLL
+33.5). So: where observation is dense enough to determine the extra edge
weights, harmonic edges carry recovery information the embeddings do not —
consistent with the near-zero posterior graph×LM correlation
(`results/component_redundancy_dev.md`).

### Completed sweep (obs 0.50/0.70/0.80 run 2026-08-05) — a density gradient, not a uniform win

`scripts/run_charmlm_rates.sh`, report `scripts/report_charmlm_rates.py`
(`logs/charmlm_rates_report.log`). Paired c_harm_lm − b_featlm:

| hidden | dRMSE | dNLL (mean) | NLL better | median dNLL |
|---|---|---|---|---|
| 50% | −0.0037 ns (excl. collapse; see below) | +0.026 ns | 15/29 | −0.003 |
| 40% (anchor) | tie (adoption record) | tie | — | — |
| 30% | **−0.0096\*** | −0.016 ns | 19/30 | −0.014 |
| 20% | **−0.0105\*** | −0.004 ns | 21/30 | −0.022 |
| 10% | **−0.0089\*** | +0.264 ns (piece 28) | 22/30 | −0.030 |

**Collapse cell at 50% hidden:** piece 18 seed 2, RMSE 5×10⁴ (predictions to
8.6×10⁵). Shard log (`logs/charmlm_obs0.50.shard6.log`) shows the
graph-parameter optimizer at degenerate length scales (exp overflow,
divide-by-zero in the adjacency); the guard's calibration screen passed it —
the documented invisible-from-observed-notes class, now demonstrated for a
learned-graph config at sparse observation.

**Verdict:** harmonic edges + learned graph parameters add recovery value
where observation is dense enough to determine the extra edge weights
(significant at ≤30% hidden), tie at the 40% operating point, and offer
nothing at 50% hidden while adding a collapse risk. The plan's prereg
criterion (uniform win, no NLL harm) is NOT met → **no preregistration
package; the thesis model keeps the plain graph at its operating point**, and
the harmonic family stands as a measured dense-observation refinement.
Thesis passage updated accordingly (sec:kernels). Doc closed.

## Reproduce

```bash
# per rate OF in 0.50 0.70 0.80 0.90 (run logs: logs/ms_baseline_obs$OF.log):
OMP_NUM_THREADS=8 PYTHONPATH=src python scripts/eval_kernels.py --stage run \
    --kernels additive,tonal,harmonic,harmonic_vl \
    --inputs .cache/masksweep_inputs_obs$OF.pkl \
    --out-dir results/kernels_ms_obs$OF
# report over the four rates: logs/kernels_ms_report.log
```
