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

## Results

*(filled from `logs/graphgp_*.log` as runs land)*
