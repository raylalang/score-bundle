# Kernel-comparison experiment — handoff for Claude Code + Fable

**Purpose.** Run the experiment the supervisor asked for: compare graph-GP kernels
on the held-out ASAP imputation task, **from the simplest to the more experimental**,
on *both* recovery and calibration, holding everything else fixed so that **only the
kernel changes**. Produce a master table + significance and a short write-up to drop
into the thesis (`docs/thesis/draft.tex`, the `Kernel comparison` section already stubbed).

This is a spec, not code. Implement it against the existing pipeline; keep the
published protocol intact.

---

## 1. What to hold fixed (identical across every kernel row)

- **Prior mean:** the leak-free network mean `emb_leakfree` (`μ_LM`), the published
  default. Also include a `μ = 0` reference (so we can see the kernel effect without a
  learned mean).
- **Protocol:** 30 held-out ASAP pieces × 4 mask seeds, **40% hidden**, **strict
  mask-aware** embeddings, **contamination-filtered** (1036 → 653), `--noise-floor-frac
  0.05`, **EB guard on**. Use the *same* mask realizations for every kernel (identical
  seeds) so the paired per-piece tests are valid.
- **Hyperparameter fit:** per piece, per channel, by empirical-Bayes marginal
  likelihood (the existing routine), extended to each kernel's own parameters.
- **Metrics:** pooled **and** per-channel (τ, log r, v): RMSE, NLL, coverage@0.9,
  cal-err. Report per-piece bootstrap 95% CIs, **medians + worst-cell**, and paired
  per-piece diffs against the additive-Laplacian baseline (row A3).

The only thing that varies row to row is the **precision/kernel construction**.

---

## 2. Kernels to compare (simplest → experimental)

### Tier A — baselines and the current default
1. **Independent** — diagonal precision `Q = λ I` (η = 0). No coupling. Should be the
   floor.
2. **Temporal chain (AR(1)-style)** — graph = score-time chain only (each note coupled
   to its time neighbours). Already exists as the temporal baseline.
3. **Additive Laplacian** — `Q = λ I + η L_G` (the current default). **This is the
   reference row for all significance tests.**

### Tier B — other standard graph-GP kernels
4. **Graph Matérn / SPDE** — `Q = σ_g⁻² (κ² I + L_G)^α`, for **α = 1, 2** (and α = 3 if
   cheap). Already implemented in `prior.py`. Fit `(σ_g, κ)` with α fixed per row.
5. **Diffusion / heat kernel** — covariance `K = exp(−t L_G)` (so `Q = exp(t L_G)`);
   the α→∞ limit of Matérn, a very smooth graph GP. Fit the diffusion time `t`.
   Implement via the **eigendecomposition of L_G** (cache per piece): `K = U diag(exp(−t νⱼ)) Uᵀ`.
6. **Normalized-Laplacian variants** — replace the combinatorial `L_G` with the
   symmetric normalized `L_sym = D^{-1/2} L_G D^{-1/2}` (and/or the random-walk
   `L_rw = I − D⁻¹W`) inside the additive and Matérn forms. Tests whether
   degree-normalization helps on irregular note graphs (dense chords vs. sparse lines).
7. *(optional)* **p-step random walk** — `Q = (I + η L)^p` for small integer p.

### Tier C — more experimental: music-theory-informed graphs
*(These realize the "music theory in the kernel" option from the thesis Future-Work
menu, §12. Keep the combinatorial graph as the control and ablate each edge family.)*

8. **Tonal-distance edges** — replace the raw pitch term `(p_i − p_j)²` in the edge
   weight with a **music-theoretic pitch distance**: circle-of-fifths distance, a
   Tonnetz / neo-Riemannian lattice distance, or Tymoczko voice-leading distance.
   New `W_ij` → new `L_G`, plugged into the additive/Matérn form.
9. **Harmonic + voice-leading edges** — add edges for same-chord membership,
   cadential (functional) relationships, and voice-leading proximity; ablate each
   family (chord-only, +cadence, +voice-leading).
10. *(stretch)* **Connection-Laplacian** — per-note fibre with edge transforms `A_ij`
    (transposition / tonal gauge); reduces to the flat graph GP when every `A_ij = I`.
    Only if time allows; this is the geometry direction (`docs/geometry_direction.md`).

---

## 3. Codebase integration (where to touch)

- **`src/score_bundle/prior.py`** — already has additive (`λI + ηL`) and Matérn
  `σ_g⁻²(κ²I + L)^α`. Add: **diffusion** kernel (eigendecomp), **normalized /
  random-walk Laplacian** option, optional `(I+ηL)^p`. Keep a common interface that
  returns either a sparse precision or, for dense kernels, a factorized covariance.
- **`src/score_bundle/graph.py`** — add music-theoretic edge builders (tonal distance,
  chord/voice-leading edges) behind flags; combinatorial graph stays the default.
- **`src/score_bundle/model.py`** — generalize `fit_laplacian_field*` to take a
  **kernel spec + parameter vector**; reuse the noise floor + guard. For dense kernels
  (diffusion), fit in the **eigenbasis of L_G** (cache `U, ν` per piece).
- **Eval** — extend `scripts/eval_asap_robust.py` (or a new `scripts/eval_kernels.py`)
  with `--kernel {independent, chain, additive, matern1, matern2, diffusion,
  norm_additive, norm_matern, tonal, harmonic, ...}`. Identical masks across kernels;
  write one combined table + per-piece paired diffs vs `additive`.
- **Reuse:** contamination filter, `emb_leakfree`, `noise_floor_frac=0.05`,
  `guard=True`, `MetricAccumulator` (pooled + median + worst).

---

## 4. Output table (target shape)

Pooled, strict, identical masks, μ = μ_LM. Bold best per column; `*` = paired
per-piece bootstrap significant vs **additive** (row A3).

| Kernel | RMSE | NLL | cov@.9 | cal-err |
|---|---|---|---|---|
| A1 Independent (diagonal) | … | … | … | … |
| A2 Temporal chain | … | … | … | … |
| **A3 Additive Laplacian (baseline)** | … | … | … | … |
| B4 Graph Matérn (α=1) | … | … | … | … |
| B4 Graph Matérn (α=2) | … | … | … | … |
| B5 Diffusion / heat | … | … | … | … |
| B6 Normalized-Laplacian additive | … | … | … | … |
| C8 Tonal-distance edges | … | … | … | … |
| C9 Harmonic + voice-leading edges | … | … | … | … |

Plus a **per-channel** appendix (τ / log r / v × RMSE, cov) for the same rows, and a
`μ = 0` block to isolate the kernel effect from the learned mean.

---

## 5. Pitfalls / sanity checks

- Matérn α≥2 and the diffusion kernel **densify** — use the cached eigendecomposition
  of `L_G`; N is small per piece, so this is fine.
- **Identifiability:** fit α on a discrete grid (1, 2, 3), not continuously; watch the
  `(σ_g, κ)` vs `(λ, η)` trade-off; keep the EB guard on.
- Keep the **kernel the only thing that changes** between rows; if a kernel needs
  different mean-centering, flag it explicitly.
- Expect kernels to differ most on **loudness and articulation**; τ is near the warp
  measurement floor (mirror the per-channel breakdown in the thesis).
- **Sanity:** `independent` should be worst; `additive` should reproduce the published
  headline (≈ 0.393 / −0.322 / 0.921). If it doesn't, something changed between rows.

---

## 6. Deliverable back to the thesis

- `docs/kernel_comparison_results.md` — master table, per-channel appendix,
  significance, and a 3–5 sentence verdict: *which kernel wins, and is the extra
  complexity (Matérn / diffusion / music-theory edges) worth it over the plain additive
  Laplacian?*
- Then hand the numbers back to Cowork to fill the `Kernel comparison` table in
  `docs/thesis/draft.tex`.
