# Notion sync content — prepared 2026-08-05 (for Ray/Cowork to apply)

> The "Score-Bundle Models" Notion page is owned by the Cowork workflow; this
> file is the prepared delta so whoever applies it doesn't have to
> reconstruct the record. Everything below is already in the repo
> (docs/ + draft.tex); this is the summary form.

## What changed since the last sync (July → August 2026)

**Posterior decomposition (new thesis machinery + section).** The model's
posterior mean splits exactly by prior component (graph / score features /
music-model embeddings) — functional-ANOVA identity, `eq:decomp` in the
draft, `MultiOutputGraphGP.posterior_components` + `posterior_component_cov`
in code, both unit-pinned. Thesis section 5.3 is now "Attribution: posterior
decomposition". Measured findings (all development):
- Covariance shares of the mean: score features carry most of it
  (τ .69 / log r .60 / v .40); embeddings largest on velocity (.34); graph
  modest (.12/.22/.27) — features recover, graph calibrates, now shown
  inside one fit.
- Graph × embedding components are nearly orthogonal per note
  (corr −0.004/−0.025/−0.059): complements, not rivals. Explaining-away
  concentrates between graph and score features on articulation (opposite
  -sign pulls at 20% of held-out notes).
- Dominance is per-piece: the evidence switches kernels on/off per piece and
  channel (embeddings off on τ/log r for ~60% of pieces, on for a minority;
  verified mechanistically — embeddings carry no within-piece linear signal
  on τ, mislead on log r, strong unique signal on v). Extremes invert the
  division of labour wholesale (pure-graph Liszt vs pure-embedding Bach
  pieces; new contrast figure in the draft).

**Harmonic-edge question reopened and half-settled.** The 07-16 multi-rate
runs are documented: chord+voice-leading edges beat additive at every
masking level in the two-stage regime. Under the final model, the
"redundant once embeddings are in the kernel" tie was anchor-rate-only:
at 10% hidden `c_harm_lm` beats `b_featlm` on RMSE (−0.0089*, dev; NLL
majority-better, mean ns via the documented tail piece). Runs at the
remaining rates are in progress; if the dev verdict is uniform, a SECOND
preregistered confirmation set (fresh shuffle positions 80+) decides
adoption — prereg package will be prepared, go/no-go is Ray's.

**Phase-2 real-data path started.** `extract_f0` implemented (probabilistic
YIN, 10 ms hop, voicing + confidence; end-to-end test recovers synthetic
vibrato through the full chain), URMP loader written against the documented
layout, and a preregistration DESIGN draft committed
(`docs/phase2_prereg_design.md`): composition-level splits, as-given
estimator noise default, τ-policy ladder, and a URMP-specific tracker
-calibration step (their ground-truth F0s share our hop grid). Full URMP
download pending the registration form (Ray action).

**Thesis hygiene pass (Ray-directed).** draft.tex cleaned for clarity over
history: sidelined results (sustain-overlap, scaling section, three
development-era digest figures, duplicate kernel row, unused machinery)
removed; 47 pp, compiles clean. Dev/confirmation labeling untouched.

## Standing state (unchanged)

Thesis model = multi-output graph GP, preregistered confirmation passed
(RMSE 0.376 vs 0.393*, graph NLL −0.074*, coverage 0.925); replication set
replicates; downstream tasks re-validated; deep baselines lose on both axes.
