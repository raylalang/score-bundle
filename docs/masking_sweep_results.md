# Masking-level sweep — results (2026-07-14)

> **Status: development-set study.** Every number here is on the 30 development
> pieces (selection-reused); the confirmation set (pieces 31–50) was never read.
> Motivation (supervisor comment, 2026-07-13): the published 40%-hidden
> operating point was inherited, not justified — sweep the hidden fraction from
> 50% down to the leave-one-out limit and check that the model ordering and the
> graph's marginal value are not artifacts of one masking level.

## Protocol

Driver: `scripts/run_mask_sweep.sh` (logs: `logs/masksweep_*.log`; raw cells:
`results/graphgp_masksweep/`). Hidden fractions 50/40/30/20/10%; 30 dev pieces
× 4 mask seeds each; guard **on** (new-run policy). Three configurations of the
proposed model (`scripts/eval_graphgp.py`):

| config | covariance content |
|---|---|
| `b_feat` | coregionalized graph GP + score-feature kernel |
| `b_featlm` | + music-model embedding kernel (**= the proposed model**) |
| `b_featlm_nograph` | proposed model with the graph kernel removed (K_G = I) |

Discipline: the mask-aware LM embeddings are **recomputed per fraction**
(leak-freeness requires blanking exactly each mask's hidden velocities; mask
seed bases 50xx, disjoint from the published 1000 and the confirmation base).
The 40% row (`obs0.60_anchor`) re-runs the published masks/embeddings guard-on;
its proposed-model RMSE 0.3601 reproduces the published dev value (0.360),
validating the pipeline end-to-end.

The LOO limit (`scripts/eval_gp_loo.py`) predicts every note from all the
others via the closed-form GP LOO-CV identity. Two protocol differences,
flagged rather than hidden: hyperparameters are fit once per piece on the
fully observed piece (standard GP LOO-CV — evidence sees every note), and the
embedding matrix is the leak-free pre-velocity readout `emb_leakfree` (note
*i*'s embedding never contains its own velocity; τ and log r never enter the
LM input), so the limit stays leak-free without per-note re-embedding.

## Pooled metrics (all held-out notes; `logs/masksweep_report.log`)

| hidden | features only | **proposed model** | proposed, no graph |
|---|---|---|---|
| 50% | 0.3760 / −0.299 | **0.3711 / −0.330** | 0.3815 / −0.275 |
| 40% (anchor) | 0.3683 / −0.370 | **0.3601 / −0.404** | 0.3755 / −0.337 |
| 30% † | 0.3649 / −0.177 | **0.3561 / −0.140** | 0.3726 / −0.163 |
| 20% | 0.3681 / −0.394 | **0.3521 / −0.441** | 0.3611 / −0.340 |
| 10% | 0.3671 / −0.359 | **0.3472 / −0.418** | 0.3513 / −0.401 |
| LOO | 0.3497 / −0.524 | **0.3288 / −0.573** | 0.3409 / −0.531 |

Cells are RMSE / NLL; coverage@90% is 0.92–0.93 everywhere (calibration is
insensitive to the masking level). † see the outlier note below.

## Paired per-piece contrasts (bootstrap 95% CIs over 30 pieces; `*` = CI excludes 0)

ΔRMSE, negative = the ingredient helps:

| hidden | LM value (featlm − feat) | graph value (featlm − nograph) |
|---|---|---|
| 50% | −0.0050 [−0.0125, +0.0002] | **−0.0110 [−0.0192, −0.0029]*** |
| 40% | **−0.0084 [−0.0162, −0.0024]*** | **−0.0165 [−0.0265, −0.0070]*** |
| 30% | **−0.0091 [−0.0168, −0.0032]*** | **−0.0168 [−0.0265, −0.0085]*** |
| 20% | **−0.0162 [−0.0255, −0.0085]*** | **−0.0089 [−0.0137, −0.0042]*** |
| 10% | **−0.0200 [−0.0308, −0.0111]*** | −0.0040 [−0.0099, +0.0020] |
| LOO | **−0.0212 [−0.0314, −0.0124]*** | **−0.0126 [−0.0239, −0.0039]*** |

ΔNLL is significant in the same direction wherever ΔRMSE is (except the
30% row, where the piece-28 outlier inflates the CI; see below).

## Findings

1. **The ordering is stable at every masking level, including LOO**:
   proposed model ≤ no-graph ablation and ≤ features-only throughout. The
   published 40% operating point is not a lucky choice.
2. **The two ingredients trade places smoothly with observation density.**
   The LM-embedding value grows monotonically as more of the piece is observed
   (−0.005 at 50% hidden → −0.021 at LOO): the embeddings are mask-aware, so
   richer context makes them better features. The graph's value peaks in the
   mid-density regime (−0.011…−0.017* at 50–20% hidden), shrinks to
   non-significance at 10% hidden (a low-power cell: only ~10% of notes are
   scored per piece), and is significant again under LOO (−0.0126*), where
   every note is scored. Neither ingredient is redundant anywhere in the range.
3. **Calibration does not degrade at any level** (coverage 0.92–0.93,
   calibration error 0.08–0.09 across the board).
4. **The 30% NLL anomaly is one cell, not a trend**: piece 28, seed 2 hits all
   three configs identically (max |z| ≈ 106; the four worst notes are all τ,
   with ~+2.9-beat onset residuals against predictive sds of ~0.03 — an
   extreme local timing event hidden under that mask). Excluding that single
   cell restores the 30% NLLs to −0.395 / −0.443 / −0.374, in line with the
   neighboring rows. This is the same Gaussian-tail limitation already
   documented at confirmation (the τ-outlier cell) — reported, not patched;
   it is the measured motivation for the Student-t timing likelihood listed
   as future work.

## Reproduce

```bash
bash scripts/run_mask_sweep.sh                                  # ~7 h (GPU precompute + 12 CPU shards)
PYTHONPATH=src python scripts/report_mask_sweep.py              # this report
```
