# Theory alignment — the proposed model vs. the graph-GP literature (2026-07-14)

> **Purpose (supervisor comment, 2026-07-13):** compare the method against
> graph-GP theory component by component; the model should follow the
> literature *exactly*, and every deviation must be forced by a constraint,
> never a strategic choice. This document is that audit. Verdict up front:
> **every component is a standard construction; the deviations found are
> (i) an identifiability-forced relocation of the scale parameter,
> (ii) degeneracy floors motivated by measured failures, and (iii) one naming
> error on our side — the "additive" kernel *is* the graph Matérn ν = 1 and
> should be called that.** No unforced deviation from theory was found.

Reference forms are quoted from Borovitskiy, Terenin, Mostowsky & Deisenroth,
*Matérn Gaussian Processes on Graphs*, AISTATS 2021 (arXiv:2010.15538),
equations (8)–(16); multi-output structure from Bonilla, Chai & Williams,
*Multi-task Gaussian Process Prediction*, NeurIPS 2008 (ICM); feature kernels
from Rasmussen & Williams, *GPML* §2.7 (Bayesian linear regression as a GP).

## 1. Component-by-component map

| Ours (`src/score_bundle/gp.py`) | Literature construction | Match |
|---|---|---|
| `Δ = D − W` from `graph.laplacian` | Borovitskiy et al. eq. (8), unnormalized Laplacian | exact |
| `matern-α`: `g(ν;s) = (s/(s+ν))^α` | graph Matérn `(2ν/κ² + Δ)^(−ν)`, eq. (12) | exact up to their own eq.-(16) rescaling — see §2 |
| `additive`: `g(ν;s) = 1/(1+sν)` | the **same** kernel at ν = 1 (see §3) | exact; misnamed on our side |
| `diffusion`: `g(ν;s) = e^{−sν}` | graph diffusion/heat `e^{−κ²Δ/2}`, eq. (12), `s = κ²/2` | exact |
| `B ⊗ K_G` coregionalization | intrinsic coregionalization model (Bonilla et al. 2008) | exact |
| `Σ_f diag(c_f) ⊗ X_f X_fᵀ` | marginalized Bayesian linear mean = linear kernel (GPML §2.7) | exact |
| `diag(ς²) ⊗ I` per-channel noise | heteroscedastic-by-output Gaussian likelihood | standard |
| hyperparameters by exact log marginal likelihood | type-II ML, the default in all of the above | standard |
| non-uniform prior variance across notes (diag of `K_G` not flattened) | the paper *keeps* vertex-dependent variance (their Fig. 1 discussion) | exact |

The two-stage pipeline's spectral machinery (`prior.SPECTRAL_KERNELS`,
`model.SpectralGaussianField`) evaluates the same families in covariance form;
the p-step random-walk kernel was already handled as the Matérn family
reparameterized (see `docs/kernel_comparison_results.md`), which matches the
paper's own account of Smola & Kondor's kernel (their eqs. 17–18).

## 2. Deviation 1 — shape normalization `g(0) = 1`, scale lives in `B`

Ours: `K_G(s) = U g(ν;s) Uᵀ` with `g(0) = 1`; the paper's kernels carry a free
prefactor `σ²`.

This is not a new kernel: for the Matérn with `s = 2ν/κ²`,

```
(s/(s+λ))^ν  =  (2ν/κ²)^ν · (2ν/κ² + λ)^(−ν),
```

i.e. our `g` is the paper's kernel multiplied by the constant `(2ν/κ²)^ν` —
**the same rescaling the paper itself applies in its eq. (16)** to take the
ν → ∞ diffusion limit. What we remove is only the freedom of `σ²`, and that
removal is *forced by identifiability*: in `B ⊗ K_G`, the map
`(B, K_G) → (cB, K_G/c)` leaves the covariance invariant, so a scale inside
`K_G` is unidentifiable the moment a coregionalization matrix is present.
Pinning `g(0) = 1` puts all scale in `B` (one place, three channels). The same
argument forces the feature-kernel scales `c_f` to be per-channel diagonals
rather than carrying their own `B`-like factor. Constraint-forced; zero
modeling content.

## 3. Deviation 2 (naming, ours to fix) — "additive" *is* graph Matérn ν = 1

```
additive:   g(λ) = 1/(1 + sλ)  =  (s'/(s' + λ))   with  s' = 1/s
matern-1:   g(λ) = s/(s + λ)
```

The same one-parameter family with the parameter inverted. The kernel
comparison unknowingly ran it twice — rows A3 ("additive", 0.3930) and B4
("Matérn α=1", 0.3932) differ only by optimizer path, which is why they tie to
the third decimal. Consequences:

* the thesis model's graph kernel should be *named* what it is: **graph Matérn
  ν = 1** (equivalently the regularized-Laplacian kernel — Smola & Kondor's
  name for the same object);
* the kernel table's honest reading is: Matérn ν ∈ {1, 2, 3} and the heat
  kernel all tie on this task (see §6);
* code keys (`additive`, `matern1`) stay as-is for provenance — the pickles
  and logs reference them — with the equivalence documented here and in
  `gp.py`'s comments; the *draft* uses the literature name.

## 4. Deviation 3 — floors (noise floor, predictive-variance floor, spectral clip)

* **Per-channel noise floor** (5% of observed variance): forced by the
  *measured* empirical-Bayes collapse — without it a minority of evidence fits
  drive `ς² → 0` and NLL diverges (documented in
  `docs/phase1_calibration_results.md`, 2026-07-02 correction).
* **Predictive-variance floor** (`diag(Σ_y) + ς²` for held-out notes): not a
  deviation at all — held-out observations are `y = f + ε`, so *omitting* the
  noise term would be the theory error. The literature's noise-free plots
  predict `f`; we predict `y`.
* **Spectral clip** of `g` to `[10⁻¹², 10¹²]`: floating-point guard, inert in
  healthy fits.

All three are degeneracy/numerics constraints, none is a modeling choice.

## 5. Laplacian choice

The paper defines both the unnormalized `Δ = D − W` (its default, eq. 8) and
the symmetric-normalized variant (its eqs. 13–14) and explicitly leaves the
choice application-dependent (citing von Luxburg 2007). We default to the
unnormalized Laplacian and *measured* the normalized one as a statistical tie
(kernel comparison row B6). Aligned; the draft should state the choice and
cite the tie rather than treating the default as self-evident.

## 6. Residual freedom the theory allows and we restrict

* **Smoothness ν fixed at 1** (rather than learned): the paper notes integer ν
  gives Markov/sparse structure; continuous ν is possible spectrally. We fixed
  ν = 1 *after measuring* ν ∈ {1, 2, 3} and the heat kernel as statistical
  ties on this task (`docs/kernel_comparison_results.md`), so the restriction
  is evidence-supported, not arbitrary — but it is the one place where a
  purist implementation could expose ν as a learned hyperparameter. Expected
  gain, per the measured ties: none. If a reviewer asks "why ν = 1", the
  answer is the tie table, and §7's spectral-overlap explanation of *why* it
  ties.
* **Covariance form instead of sparse precision**: for integer ν the Matérn
  precision is a sparse polynomial in `Δ` (the GMRF connection the paper
  makes). We work in covariance form because (i) pieces are small (N ≤ ~2000),
  (ii) the feature kernels and `B ⊗ ·` structure are covariance-side and dense
  anyway, and (iii) the diffusion *precision* overflows while its covariance
  is benign (measured; see the kernel-comparison implementation note). An
  implementation constraint, not a model change.

## 7. Why the spectral families tie here (the "grid search" question)

There is no grid search to find: the shape `s` is fit *continuously* per piece
by exact marginal likelihood, jointly with `B`, `{c_f}`, `ς²` (L-BFGS-B with a
Nelder–Mead polish, `gp.fit`). The observed near-ties across Matérn ν = 1/2/3
and heat are then expected rather than suspicious:

1. ν = 1 vs. "additive" is an *exact* reparameterization (§3) — that pair ties
   by identity;
2. the remaining family members are one-parameter monotone spectral filters of
   the same `Δ`; with `s` free and evidence-fitted, they realize nearly the
   same effective filter over the graph's bounded spectrum, differing only in
   tail curvature;
3. the graph term is one of three covariance components — the feature kernels
   and the noise absorb most variance, making tail-shape differences
   second-order on this task.

The regime that *does* move both metrics is changing the connectivity itself
(harmonic chord + voice-leading edges, in the two-stage regime), which is the
graph-GP-consistent conclusion: the information is in `Δ`, not in the fine
shape of `Φ(λ)`. A one-figure demonstration (fitted `g(ν)` curves overlaid per
kernel from the existing result pickles) is a cheap addition if wanted.

## 8. Actions

1. Draft naming: "additive kernel" → **graph Matérn ν = 1 (regularized
   Laplacian)** everywhere in `draft.tex`; cite Borovitskiy et al. for the
   family and Smola & Kondor for the regularized-Laplacian name (folds into
   the pending terminology sweep).
2. Draft model chapter: state the identifiability argument of §2 in one
   sentence where `g(0) = 1` is introduced (it currently reads as a
   convention; it is a forced choice).
3. Draft related work: one sentence on the Laplacian choice + the measured
   normalized-Laplacian tie.
4. `gp.py` comment: note the additive ≡ Matérn-1 equivalence at the
   `SHAPE_KERNELS` table (done in the same commit as this document).
