# GP-first: reformulating Phase 1 as one orthodox graph Gaussian process

*Branch `graph-gp-first`, started 2026-07-09. Internal until there are numbers.*

## Why

The supervisor's frame is graph GPs. Review of the three papers in `related_works/`
against our pipeline:

- **Venkitaraman, Chatterjee & Händel (2018), "Gaussian Processes Over Graphs"** —
  multi-output GP where the graph couples the outputs through a Kronecker factor:
  `C = B² ⊗ K + β⁻¹I`, `B = (I + αG)⁻¹` from the Laplacian spectrum; exact inference;
  provably smaller predictive variance than the graph-free GP.
- **Borovitskiy et al. (2021), "Matérn Gaussian Processes on Graphs"** — the GP over
  vertices via functional calculus on the Laplacian: `K = U Φ(Λ)⁻¹ Uᵀ`, Matérn
  `(2ν/κ² + Δ)^(−ν)` / diffusion `e^(−κ²Δ/2)`; **all** hyperparameters (ν, κ, σ²,
  noise) by marginal likelihood; vector-valued outputs via the separable
  coregionalization kernel; exact conjugate inference.
- **Blanco-Mulero, Heinonen & Kyrki (2021), e-GGP** — autoregressive GP on evolving
  graph structure with neighborhood sub-tree kernels. Least applicable here (our
  score graph is static per piece); its exact-GP + NLL discipline matches ours.

### The honest audit of the current pipeline

Already orthodox: the prior IS a graph GP (`λI + ηL` = the regularized-Laplacian
kernel; the 2026-07-09 kernel sweep is literally the Borovitskiy spectral
construction, ν on an integer grid, marglik hyperparameters, exact inference), and
calibration is evaluated more seriously than in any of the three papers.

Four genuine deviations:

1. **Channels are three independent scalar GPs.** Both GP papers prescribe explicit
   output coupling (Kronecker / coregionalization). Cross-channel covariance between
   timing, articulation, loudness is ignored.
2. **The graph's own parameters are fixed** (ℓ_b=2, ℓ_p=4; chord/VL weights = 1).
   In GP terms these are kernel hyperparameters; orthodoxy learns them by evidence.
3. **The mean is a two-stage plug-in** (ridge head fit on head pieces → GP on the
   residual). The orthodox move: a linear kernel on features IS the marginalized
   Bayesian linear mean — one model, one marginal likelihood, no plug-in.
4. **Narrative gravity** sits on the LM rather than the GP.

The current model is the **special case** (diagonal coregionalization + fixed graph +
plug-in mean) of the orthodox one — so each orthodoxy step gets its own measured
marginal value, and nothing is thrown away: the published pipeline becomes the
ablation chain of the GP-first model.

## The model

One GP over the 3N-dimensional field (channel-major stacking; per-note masks keep the
observed block Kronecker-compatible):

```
K_total = B ⊗ K_G(s) + Σ_f diag(c_f) ⊗ X_f X_fᵀ + diag(ς²) ⊗ I
```

- `K_G(s) = U g(ν; s) Uᵀ` — shape-normalized spectral graph kernel (`g(0)=1`, one
  shape parameter; additive / Matérn-α / diffusion), all scale living in `B`;
- `B` — 3×3 PSD coregionalization (log-Cholesky, 6 params);
- `X_f` — per-note feature matrices (25 score features + bias; optionally mask-aware
  LM embeddings), `c_f` per-channel scales: the **marginalized Bayesian linear mean**;
- `ς²` — per-channel noise, floored at 0.05·Var(observed residual) as established.

Everything fit **jointly by exact log marginal likelihood** (`gp.MultiOutputGraphGP`;
scipy L-BFGS-B import-guarded, Nelder–Mead fallback + polish). Exact conjugate
posterior; predictive std includes the channel noise. Modeling difference to measure
honestly: the linear-mean weights are now effectively per-piece Bayesian (posterior
from that piece's observed notes under a learned prior scale) instead of the current
cross-piece point-estimate head.

## Staged evaluation (identical strict protocol, cached masks)

| Config | Adds | Question |
|---|---|---|
| `a_diag` | nested special case (3 single-channel GPs) | **gate**: reproduce published zero+graph |
| `a_icm` | coregionalization B | does cross-channel coupling help? |
| `a_icm_m2` | Matérn-2 shape | shape robustness (expected tie) |
| `b_feat` | linear kernel on score features | mean-as-kernel vs plug-in head |
| `b_featlm` | + mask-aware LM embeddings | does the LM survive inside the evidence? |
| `c_graph` | learned ℓ_b, ℓ_p | does the evidence want a different graph? |
| `c_harm` | harmonic graph, learned ℓ_b, ℓ_p, chord, vl | edges + evidence together |
| `d_corpus` | ONE corpus-level hyperparameter set (fit on head pieces, frozen) | canonical train/test; kills per-piece fitting fragility |

Report: `scripts/eval_graphgp.py --stage report` — pooled + med/worst cell +
per-channel, paired per-piece bootstrap vs the published baselines loaded from
`results/kernels/additive.pkl` (zero+graph, LM+graph) and
`results/kernels_featlm/{additive,harmonic_vl}.pkl` (candidate, adopted headline).

## Contracts pinned by tests (`tests/test_graphgp.py`)

- B-diag joint marglik **equals** the sum of per-channel `SpectralGaussianField`
  margliks under the matched reparameterization (exact nesting).
- Linear-kernel GP **equals** the explicitly marginalized linear mean.
- ICM transfers information across channels on rank-1 data.
- Bitwise held-out-target leak invariance through fit + posterior.

## Decision framework (the user's call; this branch produces evidence)

Paired vs the adopted headline (0.3795 / −0.3459 / 0.922 strict):
better on both axes → GP-first becomes the thesis model, current work absorbed as its
ablations; tie → likely still the preferable framing for a graph-GP audience; worse →
keep the current headline and report this as the honest orthodoxy ablation.

## Results (2026-07-09, `logs/graphgp_final_report.log`)

Identical strict protocol and masks; paired per-piece bootstrap vs the adopted
headline (feat+LM+harmonic, 0.3795 / −0.3459 / 0.922). Gate passed first: `a_diag`
reproduces the published zero+graph cell within noise (0.4036/−0.3062 vs
0.4041/−0.3083; both deltas ns).

| Config | RMSE | NLL | cov@.9 | ΔRMSE vs headline | ΔNLL vs headline |
|---|---|---|---|---|---|
| a_diag (nested special case) | 0.4036 | −0.306 | 0.923 | +0.0234* | +0.039 |
| a_icm (+ coregionalization) | 0.4029 | −0.315 | 0.923 | +0.0224* | +0.031 |
| a_icm_m2 (Matérn-2 shape) | 0.4032 | −0.302 | 0.920 | +0.0227* | +0.043* |
| b_feat (features as kernel, no LM) | 0.3679 | −0.370 | 0.923 | **−0.0128*** | −0.025 |
| **b_featlm (+ LM embeddings)** | **0.3590** | **−0.4051** | **0.927** | **−0.0218*** | **−0.0602*** |
| c_graph (learned ℓ_b, ℓ_p) | 0.3647 | −0.384 | 0.918 | −0.0156* | −0.039 |
| c_harm (harmonic, learned) | 0.3606 | −0.382 | 0.917 | −0.0196* | −0.037 |
| c_harm_lm (everything) | 0.3561 | −0.396 | 0.920 | −0.0240* | −0.0506* |
| d_corpus (frozen corpus params) | 0.3670 | −0.183 | 0.929 | −0.0135* | +0.163* |
| d_hybrid (frozen + per-piece noise) | 0.3671 | −0.350 | 0.941 | −0.0135* | −0.004 |

Head-to-head of the two leaders: c_harm_lm vs b_featlm ΔRMSE −0.0022 [−0.010,+0.006]
ns, ΔNLL +0.0096 ns — statistically tied; **b_featlm wins by parsimony** (fixed plain
graph, no learned graph parameters, best NLL and coverage).

### Reading the ladder

1. **Mean-as-kernel is the big effect.** Folding the score features into the kernel
   (per-piece Bayesian weights under one evidence) is worth −0.036 RMSE vs the
   equivalent plug-in system (b_feat 0.3679 vs feat-mean+graph ≈ 0.404/0.3879-tier) —
   far larger than any kernel-family or edge-family effect measured before.
2. **The LM survives inside the evidence.** Embeddings-as-kernel add −0.0090* RMSE
   and −0.0354* NLL on top of features (paired b_featlm vs b_feat) — the LM's
   calibration + loudness contribution (v RMSE 0.0749 → 0.0718) transfers into the
   orthodox model.
3. **Coregionalization alone is marginal** at zero mean (a_icm ≈ a_diag); its value
   shows up combined with the feature kernels.
4. **Learning the graph by evidence helps without the LM** (c_harm 0.3606) but adds
   nothing once the LM embeddings are in (c_harm_lm ties b_featlm) — the embeddings
   already carry the local-context information the extra edges encode.
5. **What must be per-piece is the noise.** Corpus-frozen hyperparameters recover
   well (0.3670) but blow up NLL on one atypical piece (+3.1 on piece 7); refitting
   only the 3 noise parameters per piece recovers most calibration
   (d_hybrid −0.350). Full per-piece evidence remains best.
6. **Per-channel, b_featlm is the best measured on every channel**: τ 0.1518 (below
   the previous "measurement floor" plateau of ~0.156), log r 0.5988, v 0.0718.
7. **Zero-leak audit passed bitwise** on the real run path
   (`scripts/audit_graphgp_leakfree.py`; embedding-side invariance was already
   proven on the shared precompute path, and the unit contract is in
   `tests/test_graphgp.py`).

### Graph ablation inside the GP-first model (review-driven, 2026-07-09)

Removing the graph term (K_G = I; `*_nograph` configs) from the winners, paired vs
b_featlm: **b_featlm_nograph +0.0174* RMSE [+0.008,+0.028] and +0.0690* NLL
[+0.055,+0.085]** (b_feat_nograph: +0.0177*/+0.0847*). The graph's marginal value
survives — and its calibration contribution is the largest single effect measured in
the model. Per-piece Bayesian features alone are strong on recovery (0.3755, still
better than the old headline's 0.3795) but the graph is what makes the confidence
honest. The thesis claim — *structure + calibration* — holds inside the orthodox
formulation.

### GP-first candidate headline

**One multi-output graph GP** (ICM over τ/log r/v; additive spectral graph kernel;
score features + mask-aware LM embeddings as linear kernels = marginalized Bayesian
linear mean; per-channel floored noise; everything by exact per-piece marginal
likelihood): **RMSE 0.3590 / NLL −0.4051 / coverage 0.927 strict** — significantly
better than the adopted two-stage headline on both axes, with a strictly simpler
story (one model, one evidence, no plug-in head, no leak surface through a fitted
read-out head). Decision on adopt/ditch/absorb: the user's.
