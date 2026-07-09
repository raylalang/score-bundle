# Kernel comparison — results (2026-07-09)

> **Status (2026-07-10):** headline superseded by the GP-first model
> (`docs/graphgp_first_design.md`); this study stands as the two-stage-regime record
> that (i) justifies the additive kernel shape used inside the GP-first covariance and
> (ii) motivated the reformulation. Its harmonic-edge finding is regime-scoped: the
> edges are significant on the plain graph, and measured-redundant once the LM
> embeddings enter the kernel as features.

The experiment the supervisor asked for (spec: `kernel_comparison_experiment.md`):
compare graph-GP kernels on the held-out ASAP imputation task, simplest → experimental,
on recovery **and** calibration, with **only the kernel changing** row to row.

*Every number is from the 2026-07-09 re-run (`logs/kernels_report.log`,
`logs/kernels_run_*.log`; per-row raw results in `results/kernels/`).*

## Protocol (identical for every row)

Published strict protocol, unchanged: leak-free network mean `μ_LM` (strict mask-aware
embeddings — held-out velocities never enter the LM input; head fit on `emb_leakfree`,
l2=10), contamination-filtered cache (1036 → 653), 30 held-out pieces × 4 mask seeds,
40% hidden, `noise_floor_frac=0.05`, predictive-variance floor, **EB guard ON**,
**identical mask realizations for every kernel** (the same rng sequence as the published
headline run — the additive row below reproduces the published strict headline to the
fourth decimal, which validates the new spectral machinery end-to-end). A `μ = 0` block
isolates the kernel effect from the learned mean. Hyperparameters are fit per piece per
channel by marginal likelihood (each kernel's own parameters + noise, log-space
Nelder–Mead), with the same guard ladder as the published protocol.

Implementation: every kernel is a spectral function of the graph Laplacian
(`prior.SPECTRAL_KERNELS`; covariance eigenvalues `g(ν)`), run through a covariance-form
field (`model.SpectralGaussianField`) because the diffusion kernel's *precision*
`exp(tν)` overflows while its covariance is benign. The p-step random walk `(I+ηL)^p`
is not a separate row — it is the Matérn family reparameterized
(`κ² = 1/η, σ_g² = η⁻ᵖ`). One eigendecomposition per piece is shared across all fits.

## Master table — μ = μ_LM (strict), pooled over 30 pieces × 4 seeds × 3 channels

`Δ` columns are **paired per-piece bootstrap 95% CIs vs the additive baseline**
(negative = better than additive; `*` = CI excludes 0). guard = calib / conservative
fallbacks out of 720 fits.

| Kernel | RMSE | NLL | cov@.9 | cal-err | med-cell | worst | ΔRMSE vs additive | ΔNLL vs additive | guard |
|---|---|---|---|---|---|---|---|---|---|
| A1 independent (diagonal) | 0.4458 | −0.190 | 0.919 | 0.085 | 0.4331 | 0.741 | +0.0523 [+0.037,+0.070]* | +0.131 [+0.104,+0.162]* | clean |
| A2 temporal chain | 0.3922 | −0.292 | 0.916 | 0.063 | 0.3859 | 0.629 | −0.0003 [−0.010,+0.008] | +0.031 [+0.011,+0.052]* | clean |
| **A3 additive Laplacian (baseline)** | **0.3930** | **−0.322** | 0.921 | 0.067 | 0.3865 | 0.595 | — | — | clean |
| B4 graph Matérn (α=1) | 0.3932 | −0.321 | 0.921 | 0.064 | 0.3871 | 0.595 | +0.0002 [+0.000,+0.000]* | +0.002 [−0.000,+0.004] | clean |
| B4 graph Matérn (α=2) | 0.3921 | −0.321 | 0.919 | 0.064 | 0.3834 | 0.595 | −0.0008 [−0.003,+0.001] | +0.001 [−0.003,+0.006] | 1 calib |
| B4 graph Matérn (α=3) | 0.3923 | −0.318 | 0.918 | 0.062 | 0.3889 | 0.598 | −0.0006 [−0.003,+0.001] | +0.004 [−0.003,+0.012] | 1 cons |
| B5 diffusion / heat | 0.3929 | −0.312 | 0.917 | 0.062 | 0.3869 | 0.609 | −0.0000 [−0.002,+0.002] | +0.010 [−0.001,+0.022] | clean |
| B6 normalized-Laplacian additive | 0.3924 | −0.326 | 0.922 | 0.068 | 0.3975 | 0.601 | −0.0007 [−0.003,+0.002] | −0.005 [−0.020,+0.010] | 1 cons |
| C8 tonal-distance edges | 0.4026 | −0.300 | 0.921 | 0.061 | 0.4057 | 0.635 | +0.0093 [+0.003,+0.016]* | +0.022 [+0.005,+0.039]* | clean |
| C9 harmonic (chord edges) | 0.3842 | −0.331 | 0.923 | 0.066 | 0.3749 | 0.596 | **−0.0089 [−0.013,−0.005]*** | −0.009 [−0.018,+0.000] | 1 calib |
| **C9 harmonic + voice-leading** | **0.3852** | **−0.335** | 0.922 | 0.066 | 0.3762 | 0.596 | **−0.0076 [−0.011,−0.005]*** | **−0.013 [−0.020,−0.006]*** | 1 cons |

Sanity gates: the additive row reproduces the published strict headline
(0.3930 / −0.322 / 0.921) exactly; independent is worst everywhere; its μ=0 cell
reproduces the published zero-mean 0.5664.

## μ = 0 block (kernel effect without the learned mean)

Same ordering, same conclusions — the kernel effect is mean-independent:

| Kernel | RMSE | NLL | ΔRMSE vs additive | ΔNLL vs additive |
|---|---|---|---|---|
| independent | 0.5664 | −0.072 | +0.157* | +0.236* |
| temporal chain | 0.4077 | −0.256 | +0.004 | +0.052* |
| **additive (baseline)** | 0.4041 | −0.308 | — | — |
| Matérn α=1 / 2 / 3 | 0.4038 / 0.4042 / 0.4049 | −0.306 / −0.306 / −0.310 | all ns | all ns |
| diffusion | 0.4072 | −0.309 | +0.003 | −0.001 |
| normalized additive | 0.4042 | −0.311 | +0.000 | −0.003 |
| tonal-distance | 0.4158 | −0.275 | +0.011* | +0.033* |
| harmonic (chord) | 0.3957 | −0.314 | **−0.008*** | −0.006 |
| **harmonic + voice-leading** | **0.3953** | **−0.321** | **−0.009*** | **−0.013*** |

## Per-channel appendix (μ = μ_LM; RMSE / cov@.9)

| Kernel | τ RMSE | τ cov | log r RMSE | log r cov | v RMSE | v cov |
|---|---|---|---|---|---|---|
| independent | 0.1664 | 0.942 | 0.7474 | 0.915 | 0.0989 | 0.900 |
| temporal chain | 0.1674 | 0.940 | 0.6528 | 0.904 | 0.0847 | 0.904 |
| additive | 0.1581 | 0.943 | 0.6571 | 0.910 | 0.0810 | 0.909 |
| Matérn α=1 | 0.1591 | 0.942 | 0.6572 | 0.910 | 0.0811 | 0.910 |
| Matérn α=2 | 0.1585 | 0.941 | 0.6554 | 0.907 | 0.0811 | 0.908 |
| Matérn α=3 | 0.1576 | 0.939 | 0.6559 | 0.907 | 0.0811 | 0.907 |
| diffusion | 0.1576 | 0.938 | 0.6570 | 0.907 | 0.0814 | 0.906 |
| normalized additive | 0.1538 | 0.942 | 0.6571 | 0.913 | 0.0806 | 0.910 |
| tonal-distance | 0.1580 | 0.943 | 0.6742 | 0.913 | 0.0827 | 0.909 |
| harmonic (chord) | 0.1580 | 0.945 | 0.6415 | 0.914 | 0.0798 | 0.911 |
| harmonic + voice-leading | 0.1579 | 0.942 | 0.6432 | 0.913 | 0.0796 | 0.910 |

As predicted in the spec, kernels differ on articulation and loudness; τ is pinned near
the warp measurement floor and barely moves (the normalized Laplacian's small τ edge,
0.1538, is the one exception — degree normalization helps most where dense chords meet
sparse lines).

## Reading

1. **The additive Laplacian survives Tier B.** Matérn (α = 1, 2, 3), diffusion, and the
   normalized Laplacian are all statistically indistinguishable from it on RMSE, and
   none significantly improves NLL. (Matérn α=1's ΔRMSE is nominally significant but
   +0.0002 — a magnitude with no practical content.) The extra parameters and the dense
   solves buy nothing; among classical graph-GP kernels the plainest one is the right
   default, and the marginal-likelihood fit apparently already finds the right
   smoothness class within the additive family.
2. **Structure beyond the chain is real.** The temporal chain ties on RMSE but is
   significantly worse-calibrated (ΔNLL +0.031*): pitch/chord coupling is what makes
   the error bars honest, not just time adjacency.
3. **Music theory helps as extra edges, not as a replacement metric.** Replacing
   semitone distance with circle-of-fifths distance (tonal row) is significantly
   *worse* on both axes — expressive coupling follows register proximity, not tonal
   proximity. But *adding* same-chord and stepwise voice-leading edges to the plain
   graph is significantly *better*: chord edges carry the recovery gain
   (−0.0089* RMSE, concentrated in articulation and loudness), and the voice-leading
   edges convert the calibration trend into significance
   (harmonic+VL: −0.0076* RMSE **and** −0.013* NLL — the only row that beats the
   baseline on both axes). The effect persists at μ=0, so it is a property of the
   graph, not an interaction with the network mean.
4. **Guard behavior.** 0–1 fallbacks per 720 fits per row (worst case one conservative
   cell); the screen never fired on the additive baseline. The knife-edge collapse
   remains rare and contained under every kernel.

## Verdict (3–5 sentences, for the thesis)

Among standard graph-GP kernels, nothing beats the plain additive Laplacian: Matérn,
diffusion, and normalized-Laplacian variants tie on recovery and don't improve
calibration, so the extra complexity is not worth it. The kernel *is* sensitive to the
graph it runs on: chord + voice-leading edges — score information the combinatorial
graph doesn't see — give the only significant both-axes improvement over the baseline
(0.3852 / −0.335 vs 0.3930 / −0.322, paired per-piece), while *replacing* the pitch
metric with a music-theoretic one hurts. The right way to put music theory into the
prior is additional edge families on top of register proximity, not a different notion
of pitch distance. This also means the published headline was left on the table by the
plain graph: the harmonic+VL graph with the plain LM mean already beats the
feat+LM+plain-graph candidate on RMSE (0.3852 vs 0.3879 strict).

## Stack cells (winning kernel × candidate-headline mean)

*(run 2026-07-09, `logs/kernels_featlm_report.log`, `results/kernels_featlm/`)*

Same protocol, mean = the candidate-headline **feat+LM strict** representation
(feat-lin score features + mask-aware LM embeddings, concat head at l2=10). Sanity
gate passed: the additive cell reproduces the published strict candidate
(0.3879 / −0.333 / 0.921) exactly.

| Mean × kernel | RMSE | NLL | cov@.9 | ΔRMSE vs feat+LM+additive | ΔNLL |
|---|---|---|---|---|---|
| feat+LM × additive (published candidate) | 0.3879 | −0.3330 | 0.921 | — | — |
| feat+LM × harmonic (chord) | **0.3784** | −0.3418 | 0.923 | −0.0095 [−0.0133,−0.0061]* | −0.0088 [−0.0176,+0.0012] |
| feat+LM × harmonic + voice-leading | 0.3795 | **−0.3459** | 0.922 | −0.0082 [−0.0117,−0.0053]* | −0.0129 [−0.0193,−0.0060]* |

The kernel gain survives intact on top of the feature-stacked mean (same magnitude as
on the plain LM mean), i.e. **the mean upgrade and the graph upgrade are orthogonal and
compose**. Best strict cells measured to date: RMSE 0.3784 (feat+LM × chord graph) and
NLL −0.3459 (feat+LM × chord+VL graph) — vs the previous headline 0.3930 / −0.322 and
the candidate 0.3879 / −0.333.

## Zero-leak audit (pre-adoption, 2026-07-09)

Requested before the headline decision; **every check passed**.

- **Code audit.** The score support (`pitch`, `onset`, `duration`, `voice`) is built
  exclusively from the score MIDI (`features.load_asap`: beat onsets/durations from the
  score's own beat grid) — performance data enters only through the targets `y` and the
  raw `velocity` array. So every graph builder (combinatorial, chain, tonal, harmonic,
  harmonic+VL) and `rich_score_features` receive score-side quantities only; the head is
  fit on piece-disjoint head pieces; the EB fit, guard screen, and noise floor use
  observed nodes only; `SpectralGaussianField.posterior` touches `y` only at observed
  indices.
- **Bitwise invariance, library level** (`tests/test_kernels.py::
  test_heldout_targets_cannot_influence_predictions`): corrupting held-out `y` to 1e6
  leaves the guarded fit, hyperparameters, held-out predictions, and predictive stds
  bitwise unchanged (additive / matern2 / diffusion).
- **Bitwise invariance, real pipeline end-to-end** (`scripts/audit_kernel_leakfree.py`, run on
  the actual `eval_kernels.py` stages with the real cache + checkpoint): (a) corrupting
  **held-out velocities** in the LM-input source leaves the strict mask-aware `μ_LM`
  bitwise unchanged (and reproduces the sweep's cached `μ` to 5e-14); (b) corrupting
  **held-out y targets** to 1e6 leaves all predictions and stds bitwise unchanged while
  the ground-truth column (and hence the dirty-run RMSE, 1e6) proves the corruption
  landed.
- **Disclosed benchmark-definition property (not a leak):** the `v` target is centered
  by the full-piece velocity mean *before* masking (`normalize_velocity`), so the
  target's definition — identically for every kernel row, method, baseline, and all
  published numbers — involves a piece-level constant that includes held-out notes. No
  predictor input contains held-out information.

## Edge-weight sensitivity (2026-07-09, post-adoption hardening)

The harmonic edge families use untuned defaults (`chord_weight=1`, `vl_weight=1`).
Six variants (each weight ×3 and ÷3; `logs/kernels_sensitivity_report.log`, identical
masks, μ_LM strict; paired vs additive / vs the (1,1) default):

| (chord, vl) | RMSE | NLL | ΔRMSE vs additive | ΔNLL vs additive | vs (1,1) |
|---|---|---|---|---|---|
| (1, 1) — default | 0.3852 | −0.3352 | −0.0076* | −0.0130* | — |
| (0.3, 0.3) | 0.3889 | −0.3306 | −0.0040* | −0.0083* | RMSE +0.0036* worse |
| (0.3, 1) | 0.3892 | −0.3317 | −0.0035* | −0.0095* | RMSE +0.0041* worse |
| (1, 0.3) | 0.3844 | −0.3331 | −0.0086* | −0.0108* | ns both |
| (1, 3) | 0.3872 | −0.3338 | −0.0053* | −0.0116* | RMSE +0.0023* worse |
| (3, 1) | 0.3820 | −0.3265 | −0.0112* | −0.0044 ns | RMSE −0.0036* better, NLL trend worse |
| (3, 3) | 0.3843 | −0.3307 | −0.0085* | −0.0085 ns | ns both |

**Verdict: no knife-edge.** Every variant beats the plain graph on RMSE, significantly;
nothing reverses anywhere in the ×9 weight range. The weights trade off smoothly —
heavier chord edges buy more recovery at some calibration cost (c=3 loses NLL
significance), lighter ones attenuate the gain — and the untuned (1,1) default is the
setting that keeps **both axes** significant, which is the adoption criterion. The
adoption does not rest on a lucky weight; if one wanted pure RMSE, (3,1) is marginally
better (0.3820) at a real calibration cost.

## Decision (delegated by the user, 2026-07-09)

**Adopted headline: feature + network mean + harmonic (chord + voice-leading) graph —
RMSE 0.3795 / NLL −0.3459 / coverage 0.922 strict.** Rationale: it is the only
configuration in which every ingredient — the graph itself, the feature-stacked mean,
and the harmonic edge families — carries a paired per-piece improvement significant on
*both* recovery and calibration; the added edges are score structure, which is the
thesis claim; and the pre-adoption audit above found zero leakage. Chord+VL is chosen
over chord-only (slightly better RMSE) because the headline claim requires both-axes
significance (chord-only's NLL CI just misses). `LM mean + plain graph`
(0.3930 / −0.322) remains fully reported as the ablation path; nothing about the
published protocol changed.

## Reproduce

```bash
python scripts/eval_kernels.py --stage precompute            # strict μ_LM (GPU, once)
python scripts/eval_kernels.py --stage run                   # all 11 rows (numpy)
python scripts/eval_kernels.py --stage report                # master table + significance
# stack cells:
python scripts/eval_kernels.py --stage precompute --mean feat_lm --inputs .cache/kernel_sweep_inputs_featlm.pkl
python scripts/eval_kernels.py --stage run --kernels additive,harmonic,harmonic_vl \
    --inputs .cache/kernel_sweep_inputs_featlm.pkl --out-dir results/kernels_featlm
```
