# Phase-1 calibration results — held-out ASAP imputation

First end-to-end evidence for the core thesis claim: a **score-graph residual prior on top
of a learned LM mean** improves both recovery *and* calibration over hand-built baselines on
real, held-out, contamination-filtered performances.

> **⚠ Correction (2026-07-02): read the [corrected results](#corrected-results-2026-07-02)
> first.** The original tables below have two problems that were found and fixed:
> (1) a **velocity-target leak** into the LM mean, which inflated the `v`-channel LM
> numbers, and (2) the known **EB noise-variance collapse**, now fixed by a noise floor.
> The corrected headline still supports the thesis claim — and with tighter significance —
> but with smaller (honest) margins for the LM mean. Details in the new section.

## Corrected results (2026-07-02)

### Finding A — the LM mean leaked the velocity target

The per-note embedding `h_i` is read off the hidden state at each note's **VELOCITY
token**; a causal transformer's hidden state at position *t* includes the input token at
*t* itself, so `h_i` contains the note's own (bin-quantized) performed velocity and the
ridge head could partially *decode* the `v` target rather than predict it. Masking was
applied in y-space only, never in the LM's input, so this affected held-out notes too.
The tell: LM `v` RMSE 0.039 ≈ the 32-bin quantization width (≈0.031).

Fix: **score-only embeddings** — every note is tokenized with a constant placeholder
velocity (64), so `mu_LM` is a purely score-conditioned prior mean
(`scripts/extract_asap_arrays.py` caches both variants). On the same 50 eval pieces the
LM-mean `v` RMSE moves 0.057 → 0.112 (score-only), while `tau` (0.143 vs 0.144) and
`log r` (0.724 vs 0.723) are unchanged — exactly the leak signature. All corrected
numbers below use score-only embeddings. (Conditioning on *observed* notes' velocities
would be legitimate; mask-aware embeddings are future work — score-only is conservative.)

### Finding B — the EB noise floor fixes the collapse

`fit_laplacian_field(..., noise_floor=...)` clamps `noise_var` inside the EB objective
(evals use `noise_floor_frac=0.05`, i.e. 5% of the observed residual variance). This is
the fit-side counterpart of the predictive-variance floor of finding #1 below.

### Corrected headline (30 eval pieces × 4 mask seeds, score-only embeddings)

`scripts/eval_asap_robust.py --embeddings emb_scoreonly [--noise-floor-frac 0.05]`,
logs in `logs/robust_scoreonly{,_floor}.log`:

```
                             no floor                        noise_floor_frac = 0.05
mean   graph      RMSE [95% CI]        NLL [95% CI]       RMSE [95% CI]        NLL [95% CI]      cov@.9
zero   off      0.5664 [0.508,0.598]  -0.007 [-0.12,0.12]   0.5664 [0.508,0.598]  -0.007 [-0.12,0.12]  0.869
zero   graph    0.4038 [0.364,0.424]  -0.307 [-0.41,-0.20]  0.4041 [0.364,0.424]  -0.308 [-0.41,-0.21]  0.923
ridge  off      0.4233 [0.377,0.448]  -0.219 [-0.32,-0.12]  0.4233 [0.377,0.448]  -0.219 [-0.32,-0.12]  0.917
ridge  graph    2.6450 [0.370,2.017]  130.41 [-0.38,388.3]  0.9566 [0.360,0.847]   4.614 [-0.40,14.89]  0.918
LM     off      0.4505 [0.407,0.474]  -0.105 [-0.20,-0.00]  0.4505 [0.407,0.474]  -0.105 [-0.20,-0.00]  0.894
LM     graph    0.4065 [0.360,0.433]   0.076 [-0.40, 0.91]  0.3939 [0.356,0.417]  -0.314 [-0.41,-0.22]  0.922
```

With the floor, **`LM + graph` is the best cell on both axes and the paired bootstrap is
now significant on both**:

```
paired per-piece diff (negative = LM+graph better)      RMSE                NLL
LM+graph vs LM mean-only     -0.0551 [-0.0736,-0.0377]*   -0.2083 [-0.2483,-0.1714]*
LM+graph vs ridge mean-only  -0.0267 [-0.0443,-0.0113]*   -0.0943 [-0.1378,-0.0544]*
LM+graph vs zero+graph       -0.0098 [-0.0164,-0.0034]*   -0.0055 [-0.0437,+0.0271]
```

Honest reading: the graph residual is doing most of the calibration work (zero+graph is
nearly as good on NLL); the (score-only) LM mean adds a small but significant RMSE gain.
The large LM-vs-zero dynamics gap in the original table was mostly the leak. The
`ridge + graph` τ blow-up (finding #2 below) is tamed but not cured by the floor —
it is a bad-mean artefact, not a fit artefact.

## Setup

- **Prior mean source `μ`** ∈ {`zero`, `ridge` (hand-built score-feature ridge), `LM`
  (`μ_LM` from the MAESTRO-pretrained MusicGPT per-note embeddings)}.
- **Graph residual** ∈ {`off`, `on`}: `off` predicts the mean with a homoscedastic residual
  std (isolates the mean); `on` runs the closed-form GMRF posterior centered on that mean
  (`y − μ ~ N(0, Q_G⁻¹)`), with per-piece empirical-Bayes `(λ, η, noise_var)`.
- **Targets** `y = [τ, log r, v]` (onset residual, articulation, centred velocity) from the
  ASAP beat-grid warp.
- **Data hygiene.** 20 held-out ASAP pieces, **piece-disjoint** from the 40-piece head-fit
  split; **contamination-filtered** — any ASAP performance whose MAESTRO twin was in Phase-0
  pretraining is dropped (1036 → 653 aligned performances), so the LM never saw the eval
  performance. LM: `checkpoints/maestro_scaled/best.pt`, val perplexity 10.85.
- Reproduce: `python scripts/eval_asap_calibration.py --asap-root ../data/asap-dataset
  --maestro-root ../data/maestro-v3.0.0 --checkpoint checkpoints/maestro_scaled/best.pt`.

## Pooled result (over τ / log r / v)

```
mean source    graph      RMSE      NLL   cov@.9  cal-err
zero           off      0.5507   0.0007    0.869    0.187
zero           on       0.3782  -0.3551    0.924    0.089
ridge          off      0.3898  -0.2468    0.919    0.084
ridge          on       0.5187   0.9280    0.907    0.079
LM             off      0.4101  -0.3320    0.881    0.060
LM             on       0.3731  -0.6042    0.926    0.080   ← best RMSE & best NLL
```

Lower RMSE / NLL / cal-err is better; coverage closer to 0.90 is better.

**`LM + graph` is the best cell on both recovery (RMSE 0.373) and calibration (NLL −0.604,
coverage 0.926).** The graph residual improves calibration over no-graph for the zero and LM
means (coverage 0.87/0.88 → 0.92/0.93, NLL down). This is the headline: the learned LM prior
mean plus a structured graph residual beats every baseline on both axes.

## Per-channel breakdown

```
[tau]                                    [log r]                                  [v]
mean        graph  RMSE    NLL   cov     mean        graph  RMSE    NLL   cov     mean        graph  RMSE    NLL   cov
zero        off   0.185 -0.872 0.947     zero        off   0.927  1.517 0.790     zero        off   0.126 -0.642 0.868
zero        on    0.175 -0.941 0.948     zero        on    0.625  0.922 0.916     zero        on    0.086 -1.047 0.909
ridge       off   0.175 -0.900 0.950     ridge       off   0.643  0.986 0.925     ridge       off   0.108 -0.827 0.882
ridge       on    0.656  2.951 0.899     ridge       on    0.608  0.889 0.918     ridge       on    0.086 -1.056 0.902
LM          off   0.186 -0.729 0.934     LM          off   0.683  1.045 0.911     LM          off   0.060 -1.312 0.799
LM          on    0.177 -0.840 0.946     LM          on    0.620  0.913 0.917     LM          on    0.039 -1.886 0.915
```

- **Dynamics (`v`) — the clean win.** `LM + graph` reaches RMSE 0.039 / NLL −1.89 /
  coverage 0.915. The LM embedding predicts velocity far better than zero or ridge
  (LM-off 0.060 vs zero-off 0.126), and the graph adds calibrated structure on top.
- **Articulation (`log r`) — the graph fixes calibration.** A zero mean badly undercovers
  (0.79); the graph residual lifts every mean to ~0.92 coverage. Point error converges to
  RMSE ~0.61 across means.
- **Timing (`τ`) — residuals are tiny** (RMSE ~0.18) and already near the noise floor; the
  graph barely moves it and it slightly over-covers (~0.95). Not where the graph earns its keep.

## Two findings worth carrying into the writeup

1. **Predictive-variance floor is necessary.** The GMRF posterior was wildly overconfident
   (NLL ~1e22, undercoverage) until the held-out predictive std included the observation
   noise: a held-out *observation* `y = f + ε` has variance `diag(Σ_y) + noise_var`, and the
   latent `diag(Σ_y)` collapses toward 0 at well-pinned nodes. Fixed in
   `imputation_eval._predict_channel`; pinned by
   `test_graph_predictive_std_includes_observation_noise`.
2. **The graph helps a good mean and can hurt a bad one.** `ridge + graph` *hurts* pooled
   RMSE (0.39 → 0.52), entirely via the `τ` channel (ridge-on τ: RMSE 0.656, NLL 2.95): the
   ridge baseline extrapolates timing poorly and the graph propagates that bad mean. This is
   a baseline-quality artefact, not a defect of the graph prior — worth a footnote.

## LM-size ablation (does a better LM mean → a better prior?)

A second MusicGPT was pretrained on MAESTRO at ~1.3× the size (d=512, L=8, 25.6M params,
val perplexity **9.66** vs the small model's **10.85**; curves in
`figures/lm_training_curve{,_big}.png`) and run through the **identical** held-out eval
(`logs/eval_big.log`). The downstream `LM + graph` cell improves modestly but consistently:

```
                LM+graph  pooled            [v] (dynamics)
LM (val ppl)    RMSE      NLL    cov     RMSE     NLL
small (10.85)   0.3731   -0.604  0.926   0.039   -1.89
big   (9.66)    0.3700   -0.647  0.930   0.032   -2.04
```

The gain concentrates exactly where the learned mean carries the most signal — **velocity/
dynamics** (RMSE −18%, NLL −0.15) — while timing/articulation are unchanged. So "better LM →
better `μ_LM`" holds, but its leverage is channel-specific. (The `ridge + graph` τ blowup is
*worse* in this run, NLL 16.7 on τ — same baseline-quality artefact as finding #2, unrelated
to LM size.)

## Robustness (bootstrap CIs + a numerical caveat)

`scripts/eval_asap_robust.py` re-runs the comparison over **30 eval pieces × 4 mask seeds**
with **percentile-bootstrap 95% CIs over pieces** and a paired bootstrap, and writes
`figures/reliability_diagram.png` + `figures/pit_histogram.png`. It surfaced an important
**numerical fragility**, not in the method but in the empirical-Bayes fit:

- **Per-piece, the headline holds cleanly.** A single-seed per-piece diagnostic
  (`scripts/diag_robust_blowup.py`, `logs/diag_blowup.log`) shows `LM + graph` beats
  `LM`-mean-only on **every** piece — pooled RMSE 0.388 vs 0.442, NLL −0.544 vs −0.235,
  **median NLL −0.583, worst-piece NLL 0.46** (no blowup).
- **But the pooled *mean* NLL is not robust.** Across the full 4-seed run, a *minority of
  held-out mask realizations* drive the in-sample marginal-likelihood EB fit to degenerate
  hyperparameters (tiny `noise_var` → overconfident posterior), spiking that cell's NLL as
  high as ~20. Pooled-mean NLL then reads +6.9 (CI `[-0.65, +20.4]` — the heavy right tail is
  the tell), even though the **median** cell is healthy. RMSE, being bounded by the data
  scale, does **not** blow up (mean 0.388).

**Fix (next step, not yet applied):** floor `noise_var` inside `fit_laplacian_field`'s
objective (the same principle as the predictive-variance floor in finding #1), and/or report
median/trimmed NLL. The calibration-split variant (`fit_laplacian_field_calib`) is *not* a
reliable cure here (its single-seed mean NLL −0.413 is slightly worse than marglik's −0.544).

## Open follow-ups

- **Robustify the EB fit:** `noise_var` floor (or a weak prior) inside
  `fit_laplacian_field`, then re-run `eval_asap_robust.py` for a clean CI table; report
  median/trimmed NLL alongside the mean.
- Calibration-split hyperparameters, and a per-channel temperature/variance-scaling baseline,
  to push coverage to exactly 0.90.
- The `τ` channel is near the noise floor; consider whether the warp resolution limits it.
- Aria frozen-feature upper-bound baseline (Step 4) is stubbed — aria is not installed here.
