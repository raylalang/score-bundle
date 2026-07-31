# Calibrated deep baselines — results (2026-07-16)

> **Status: development-set study; closes the thesis's top flagged gap.** Until
> tonight, no deep baseline that produces *its own* per-note uncertainty had
> been run — the strongest structure-free comparator ("music-model mean, graph
> off") used a single shared predictive std. This study adds two properly
> calibrated deep rivals under the identical strict protocol.
> Confirmation set untouched.

## Systems

Both consume exactly the proposed model's information set — the 25 hand-built
score features plus the music-model embeddings (per-piece standardized; strict
mask-aware embeddings at evaluation, `emb_leakfree` at training, i.e. the same
convention as the published cross-piece heads) — trained on the 40 head pieces
(32 train / 8 validation, early stopping), evaluated on the 30 dev pieces × the
4 published anchor masks:

- **hetero-mlp** — MLP (537→256→128→6) with a heteroscedastic Gaussian output
  per channel (`μ_c(x), log σ_c²(x)`), Gaussian-NLL trained;
- **deep-ensemble** — five such heads (seeds 0–4), predictive = mixture moments.

Fairness: predictive variances receive the same 5%-of-observed-variance
per-piece floor the GP's noise floor provides (raw-sd numbers are stored too;
they are slightly worse). Cells use the standard schema; every comparison is
paired per piece. Runner: `scripts/eval_deep_baseline.py`;
raw cells: `results/deep_baseline/`.

## Results (dev, pooled; paired ΔRMSE/ΔNLL = GP − baseline, negative = GP better)

| System | RMSE | NLL | cov@.9 | GP − it: ΔRMSE [95% CI] | ΔNLL [95% CI] |
|---|---|---|---|---|---|
| **proposed model (GP)** | **0.3601** | **−0.404** | 0.927 | — | — |
| deep-ensemble (5×) | 0.4501 | −0.104 | 0.923 | **−0.0897 [−0.115, −0.067]\*** | **−0.301 [−0.402, −0.206]\*** |
| hetero-mlp | 0.4536 | −0.069 | 0.918 | **−0.0926 [−0.119, −0.069]\*** | **−0.335 [−0.435, −0.240]\*** |
| (music-model mean, graph off — prior record) | 0.446 | −0.164 | 0.90 | | |

## Reading

1. **The GP wins on both axes by large, significant margins** — ~0.09 RMSE and
   ~0.3 nats paired. Ensembling five heads buys almost nothing over one.
2. **The deep heads land exactly at the "cross-piece mean" level** (≈0.45,
   where the plug-in music-model mean without a graph already sat). This is
   the attribution story confirmed from the other side: what the deep head
   lacks is not capacity or calibration machinery — it is **per-piece Bayesian
   adaptation and the graph**, neither of which a cross-piece variance head
   can imitate. Their coverage is respectable (0.92, thanks to the variance
   head) but their NLL is far behind: the *ranking* of which notes are
   uncertain is what they cannot do.
3. **Honest scope**: these are frozen-embedding heads, the natural deep
   comparator at matched information. A fully fine-tuned end-to-end deep model
   (or one with piece-conditioning) remains untested; the measured claim is
   that at the same information set and training data, calibrated deep
   read-outs do not approach the GP — not that no deep model ever could.

## Addendum — LOO limit (run 2026-07-16, written up 2026-07-31)

Both baselines re-run under the LOO protocol of the masking sweep
(`logs/baselines_loo_nll.log`, `results/baselines_loo.pkl`; mean+smoothing =
feat+LM mean with chord+VL graph smoothing, the strongest two-stage
configuration). Per-channel RMSE / cov@.9 / NLL, dev pieces:

| system | τ | log r | v |
|---|---|---|---|
| proposed (`b_featlm`, GP LOO) | **0.149** / 0.96 / **−1.116** | **0.546** / 0.92 / **+0.730** | **0.066** / 0.91 / **−1.334** |
| mean+smoothing | 0.159 / 0.95 / −0.952 | 0.607 / 0.92 / +0.862 | 0.073 / 0.91 / −1.210 |
| MLP 5-ensemble | 0.165 / 0.97 / −0.504 | 0.755 / 0.88 / +1.104 | 0.084 / 0.95 / −0.976 |

The anchor-rate verdict extends to the LOO limit: the GP wins every channel
on both axes; the ensemble's NLL gap stays large (its per-note uncertainty
ranking does not improve with denser observation), and even the strongest
two-stage system trails the GP everywhere. (GP rows computed from
`results/graphgp_masksweep/loo.pkl`; baseline rows from the 07-16 run log.)

## Reproduce

```bash
PYTHONPATH=src:scripts python scripts/eval_deep_baseline.py
# LOO addendum (logs/baselines_loo_nll.log):
PYTHONPATH=src:scripts python scripts/eval_baselines_loo.py
```
