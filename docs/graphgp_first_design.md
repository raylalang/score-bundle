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

## PREREGISTERED confirmation protocol (written 2026-07-09, BEFORE any confirmation data)

The 30 dev pieces have adjudicated ~6 selection decisions; the confirmation set fixes
the family-wise inflation. **This section is written before the confirmation cache
exists and will not be edited after the run; the run happens exactly once.**

- **Data:** `.cache/asap_arrays_named50.pkl` eval pieces 31–50 (`--eval-start 30`,
  n=20) — same seed-0 folder shuffle and contamination filter, first 30 verified
  identical to the dev set, pieces 31–50 never used for any decision. Masks:
  `default_rng(2000 + s)`, s = 0..3 (disjoint rng base from dev's 1000+s), 40% hidden,
  strict mask-aware embeddings, `noise_floor_frac 0.05`, published protocol otherwise.
- **Systems, run once each** (no tuning, no iteration, all results reported):
  1. `b_featlm` — the GP-first candidate (primary);
  2. `b_featlm_nograph` — the graph contrast inside GP-first;
  3. `b_feat` — the LM contrast inside GP-first;
  4. old adopted headline — feat+LM mean + harmonic(chord+VL) graph, old pipeline,
     guard on;
  5. old `LM + plain graph` — the published anchor.
- **Preregistered claims:** (C1) b_featlm beats the old headline on RMSE and NLL,
  paired per-piece 95% CI excluding 0; (C2) the graph contribution inside GP-first
  (b_featlm vs nograph) is significant on NLL; (C3) b_featlm coverage@0.9 lies in
  [0.88, 0.95].
- **Decision rule (set in advance):** adopt GP-first as the thesis model iff C1 holds
  on both axes (or holds on RMSE with NLL not significantly worse) AND C2 holds;
  otherwise keep the current headline and report GP-first as the orthodoxy ablation.
  Dev-set numbers stay labeled "development" in the thesis either way.

## CONFIRMATION RESULTS (one-shot, 2026-07-09, `logs/confirmation_verdict.log`)

*Run exactly once per the preregistration above; nothing here was iterated.*

| System | RMSE | NLL | cov@.9 |
|---|---|---|---|
| **GP b_featlm** | **0.3755** | −0.3002 | 0.925 |
| GP b_featlm_nograph | 0.4052 | −0.2261 | 0.922 |
| GP b_feat | 0.3850 | −0.2639 | 0.923 |
| old headline (feat+LM+harm) | 0.3928 | **−0.3108** | 0.922 |
| old feat+LM+graph | 0.3987 | −0.2953 | 0.920 |
| old LM+graph | 0.4012 | −0.2885 | 0.921 |

- **C1 RMSE: CONFIRMED** — b_featlm − old headline = −0.0137* [−0.0246, −0.0040].
- **C1 NLL: NOT confirmed** — +0.0109 ns [−0.129, +0.245]. The dev-set NLL advantage
  did not replicate as a pooled number. Diagnosis: the *median* fresh piece is
  better-calibrated under b_featlm (−0.353 vs −0.315) but ONE fresh piece has NLL
  +2.19 (old headline's worst: +0.42) — the unguarded per-piece evidence
  overconfidence mode, demonstrated in the wild.
- **C2: CONFIRMED** — graph contribution inside GP-first, ΔNLL −0.0736* [−0.092, −0.057].
- **C3: PASS** — coverage 0.925 ∈ [0.88, 0.95].

**Preregistered decision rule → ADOPT** (C1-RMSE significant, NLL not significantly
worse, C2 holds). Honest headline claims after confirmation: the recovery advantage
and the graph's calibration contribution generalize; the pooled calibration
advantage does not (tie), pending the guard upgrade below. Note all systems' NLL is
worse on fresh pieces than on dev — the dev set flatters every model (reused-set
inflation, now quantified).

**Guard upgrade (separately reported, not part of the one-shot result):**
`gp.fit_guarded` adds an overconfidence screen (calibration-split NLL vs mean-only
+0.5 nats; fallbacks noise-floor ×5 → decoupled diagonal). Guarded A/B runs on dev
(expected no-op) and on the confirmation pieces (does it tame piece 5?) are reported
below when they land.

## Downstream re-validation (2026-07-09, `logs/downstream_gpfirst_report.log`)

All six tasks re-run with identical rng for old-pipeline and GP-first rows
(`scripts/eval_downstream_gpfirst.py`; non-strict `emb_leakfree` for both sides,
matching the original downstream condition).

| Task | Old verdict | GP-first verdict |
|---|---|---|
| Anomaly | clear win | **stronger win** — best AUROC/AP on every channel (v 0.995, log r 0.986, τ 0.978; old best 0.992/0.979/0.975) |
| Denoise (oracle noise) | win | **slightly better** (RMSE 0.182/0.276 vs 0.186/0.282; coverage 0.90) |
| Denoise (blind) | failure | **still a failure for both** — structure does not identify the noise level in either parameterization |
| Selective | qualified win | **transfers** (rmse@50% 0.076 vs 0.080; similar excess) |
| Era | honest negative | **still negative** for both (raw 0.10, GP-denoised 0.20, majority 0.60) |
| Completion (excerpt) | partial | **old pipeline more robust** — see boundary finding |

**Boundary-of-validity finding (completion).** With only an opening excerpt observed
(prefix 25%), GP-featlm produced one catastrophic extrapolation cell (RMSE 1.6e4):
the per-piece Bayesian feature weights, fit on the excerpt alone, extrapolate the
embedding features into unseen regions of the piece. The cross-piece head never
does this (old LM prefix-25%: 0.414). The sharpened thesis boundary: **per-piece
Bayesian adaptation needs observed coverage of the feature space — it wins at
interpolation-style tasks (imputation, anomaly, denoising, selective) and is the
wrong tool for excerpt extrapolation, where the cross-piece head stays the honest
choice.** Note the guard cannot catch this mode: its screen is a split of the
*observed* notes, i.e. it validates interpolation, not extrapolation.

## Decision framework (the user's call; this branch produces evidence)

Paired vs the adopted headline (0.3795 / −0.3459 / 0.922 strict):
better on both axes → GP-first becomes the thesis model, current work absorbed as its
ablations; tie → likely still the preferable framing for a graph-GP audience; worse →
keep the current headline and report this as the honest orthodoxy ablation.

## Results v2 — one code state (2026-07-09, `logs/graphgp_v2_report.log`)

The entire ladder was re-run from a single commit (`6c280ed`) into
`results/graphgp_v2/` (the original run mixed pre-/post-NaN-guard shards). v2
reproduces v1 within optimizer noise everywhere (b_featlm 0.3601/−0.4038/0.927 vs
v1 0.3590/−0.4051/0.927); **all thesis numbers use v2**.

**Attribution correction from the disentangler.** `b_fixedmean` (the published
candidate's cross-piece mean used as a FIXED mean inside the ICM GP) lands at
0.3898/−0.3274 — statistically at the old two-stage pipeline's level (+0.0098*
RMSE vs the old headline). So evidence-integration and channel coupling per se add
approximately nothing; **the driver of the GP-first gain is the per-piece Bayesian
feature adaptation under the joint evidence** (b_feat 0.3683). The honest claim is
that, not "orthodoxy wins" in the abstract.

v2 paired contrasts vs `b_featlm` (`logs/graphgp_v2_report_vsbfl.log`):
graph removed +0.0165* RMSE / +0.0666* NLL; LM removed +0.0084* / +0.0344* — both
ingredients significant on both axes inside the one-code-state rerun.

## Results v1 (superseded by v2; kept for provenance — `logs/graphgp_final_report.log`)

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
