# Posterior decomposition — consolidated dev record (2026-08-05)

> **Status: development-set studies around the exact posterior decomposition**
> (`MultiOutputGraphGP.posterior_components` / `posterior_component_cov`,
> thesis eq:decomp + §5.3 "Attribution: posterior decomposition").
> Consolidates and supersedes the scattered notes in `results/*.md`
> (posterior_components_dev, component_redundancy_dev, lm_channel_scales_dev,
> piece_dominance_dev, rate_shares_dev — raw tables remain there).
> Confirmation set untouched throughout.

## The machinery (tested, exact)

Mean: `m = m_graph + Σ_f m_feat_f`, each term `E[y_comp | obs]`; noise has no
held-out mean term (independence), acts as uniform shrinkage via the shared
weights, and reappears as the exact residual at observed notes and as the
additive predictive-variance floor. Covariance: per-component variances and
pairwise cross-covariances, with `Σ var + 2 Σ cross = latent posterior
variance` pinned by unit test. Both mask paths (Phase-2 cells included).

## Findings (config b_featlm, 30 dev pieces unless noted)

> Provenance note (2026-08-13 audit): finding 1 is 30 pieces × 4 anchor
> seeds; findings 3–5 (redundancy correlations, ARD switch-off, dominance)
> are **anchor seed 0 only** (see the raw results/*.md headers); finding 6
> is 30 × 4.

1. **Shares of the mean (anchor, 4 seeds):** score features carry most
   (τ .69 / logr .60 / v .40); embeddings largest on velocity (.34); graph
   modest (.12/.22/.27). Features recover, graph calibrates — the ablation
   attribution, now inside one fit.
2. **Stability across rates (seed 0, obs .50–.90):** pattern stable; the
   embedding share grows with observation density (v .28→.42, logr .14→.32,
   endpoint values) at the expense of both other components — the graph
   share declines with density (logr .26→.15, v .32→.24). Reproduction
   note (2026-08-13 audit): these shares reproduce only to ±0.04 across
   BLAS/thread conditions (`evidence/logs/rate_shares_repro_omp2.md`), so
   endpoint trends are the stable claim — intermediate-rate ordering
   (which rate holds the peak) is not. The anchor-rate shares of finding 1
   and the redundancy correlations of finding 3 reproduce byte-identically
   (`evidence/logs/repro_components.log`, `repro_redundancy.log`); the
   B-diagonal contrasts reproduce to the 4th decimal with identical
   signs and stars (`repro_bdiag.log`). In-fit counterpart of the
   masksweep contrast growth.
3. **Redundancy (cross-correlations at held-out notes):** graph × embeddings
   ≈ 0 (−.004/−.025/−.059) — complements, not rivals. Explaining-away lives
   between graph and score features (logr −.19, v −.18) and score × LM
   (v −.18). Opposite-sign pulls (both > .25σ): 20% of held-out articulation
   notes for graph×features.
4. **Per-channel ARD switch-off:** the embedding kernel is off (share < 1%)
   on τ/logr for ~60% of pieces, on for a minority; nearly always on for v.
   Verified behaviorally: tuned within-piece ridge gives embeddings R² ≈ 0 on
   τ, −0.44 on logr (misleads), +0.42 on v (+0.22 after residualizing on
   score features).
5. **Dominance is a piece property:** every component dominates somewhere
   (τ 3/21/6, logr 4/20/6, v 6/14/10 for graph/feat/LM); extremes invert
   wholesale — pure-graph Liszt (g 1.00 on τ) vs pure-embedding Bach
   (e .93/.89 on τ/logr; the documented tail piece). Graph-dominated timing
   pieces are the steady ones (mean shrunk to noise).
6. **B-diagonal check (anchor, 4 seeds; `scripts/eval_bdiag.py`):** severing
   the coupling costs **velocity only** (full−diag RMSE −0.0075*, NLL
   −0.093*), is neutral on articulation (+0.002 ns; NLL −0.023 ns) and
   marginally *helps* timing (+0.0005*). So (i) the coregionalization earns
   its keep on loudness; (ii) the embeddings' articulation gain in the
   ladder is NOT v→logr flow through B — it is carried by the minority of
   pieces with nonzero logr embedding scales (finding 4).
7. **Harmonic-edge mechanism (see `docs/kernel_multirate_results.md`,
   closed):** the density gradient of the harmonic advantage matches the
   decomposition story — extra per-piece edge weights need observed coverage,
   like every per-piece quantity here.

## Reproduce

`scripts/make_posterior_components.py` (figure + shares),
`scripts/eval_component_redundancy.py` (correlations),
`scripts/make_component_contrast.py` (contrast figure + disagreement),
`scripts/eval_bdiag.py`, `scripts/eval_rate_shares.py`.
