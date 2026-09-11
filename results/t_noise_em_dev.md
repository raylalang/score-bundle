# Inference-level Student-t timing noise via scale-mixture EM (development)

2026-09-11. The inference-level companion to the deploy-time study
(results/tail_predictive_dev.md): Student-t observation noise on τ,
implemented as its Gaussian scale mixture — per-note EM weights
w = (ν+1)/(ν+z²) over observed τ notes, all hyperparameters refit under
the weighted noise (two rounds, ν = 5), config b_featlm on the published
dev masks (120 cells), plain Gaussian fit recomputed in-script as the
paired baseline. **Development only; exploratory; no claims.**

## Verdict: robust inference buys two things rescoring cannot — and costs one

| metric | t-EM | Gaussian fit | paired Δ |
|---|---|---|---|
| τ RMSE | 0.1122 | 0.1109 | +0.0013 [+0.0001, +0.0026]\* (negligible vs the 0.11 scale) |
| τ NLL under the t predictive | **−1.702** | −1.404 | **−0.298 [−0.343, −0.253]\*** |
| τ NLL under the Gaussian predictive | 0.093 | −0.828 | +0.92\* (see below) |
| τ cov@90 | 0.919 | 0.948 | −0.029\* — **toward nominal** |
| log r RMSE | **0.5618** | 0.5800 | **−0.0182 [−0.0238, −0.0130]\*** |
| v RMSE | 0.0708 | 0.0708 | +0.0001 ns |

The EM's action is small and targeted: 2.8% of observed τ notes
down-weighted below w = 0.5 (max 6.2% per cell).

Three findings:

1. **The coupling spillover is the genuinely new effect.** Down-weighting
   τ outliers improves *articulation* recovery significantly (−0.018\*):
   an observed timing outlier corrupts the coupled channels through the
   coregionalization, and no deploy-time rescoring of τ can undo that.
   This is the first measured instance of robustness propagating across
   the bundle.
2. **Under its own scoring rule the t fit is clearly better** (τ t-NLL
   −0.30\*) with coverage moving to 0.919 ≈ nominal from the Gaussian
   fit's 0.948 over-coverage. The fit sharpens honestly: the outliers no
   longer inflate the τ noise floor for everyone else.
3. **The systems must be scored consistently.** Scoring the t fit with a
   Gaussian predictive is incoherent (+0.92: the sharpened variance meets
   the outliers it deliberately stopped absorbing). Any adoption is a
   likelihood change, predictive and all — not a fit tweak.

## Where this leaves the timing-tail question

Two measured options now exist, cleanly separated: the **deploy-time t
predictive** (zero fit change, kills the pooled-NLL tail, free) and the
**inference-level t noise** (adds the articulation spillover and nominal
coverage, costs EM machinery and a likelihood-consistent protocol).
The thesis's Future Work names exactly this fork; both branches are now
measured. Nothing is adopted; published tables unchanged.

- Reproduce: `PYTHONPATH=src:scripts python scripts/eval_t_noise_em.py
  run --shard K/4` (×4) then `report`.
