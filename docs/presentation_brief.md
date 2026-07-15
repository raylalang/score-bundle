# Presentation brief — Score-Bundle Models (prepared 2026-07-15, for the 2026-07-16 talk)

Everything below is verified against committed logs/results tonight; each number
names its source. Development numbers are labeled DEV; the only headline numbers
are the confirmation ones.

## The one-sentence claim

> One multi-output Gaussian process on the score graph — channels coupled, score
> features and music-model embeddings marginalized into the kernel, every
> hyperparameter learned by exact per-piece evidence — recovers hidden expressive
> detail better than the strongest two-stage pipeline it provably subsumes
> (preregistered, run once: RMSE 0.376 vs 0.393, paired −0.014\*), with the
> graph's calibration contribution independently confirmed (ΔNLL −0.074\*) and
> nominal-90% intervals covering 92.5%.

## Suggested arc (≈13 slides)

1. **Problem** — score says *what*, performance says *how*; recover per-note
   τ / log r / v **with calibrated uncertainty** (uncertainty is a primary output,
   not decoration). Fig: none needed; one bar of score + rendered deviations.
2. **Model in one equation** — eq. (gpfirst): `B⊗K_G(s) + Σ_f diag(c_f)⊗X_fX_fᵀ + diag(ς²)⊗I`.
   Fig: `figures/arch_phase1.pdf` (blue graph branch / amber evidence branch /
   vermillion likelihood).
3. **Theory pedigree** — every component is a standard construction: graph Matérn
   (Borovitskiy et al. 2021; our "additive" IS Matérn ν=1, an identity), ICM
   (Bonilla et al.), linear kernel = marginalized Bayesian linear mean (GPML §2.7).
   The only deviations are constraint-forced (g(0)=1 by B⊗K identifiability;
   floors by measured degeneracy). Source: docs/graphgp_theory_alignment.md.
4. **Discipline** — dev/confirmation split; preregistration IN GIT: prereg commit
   `3681d90` (Jul 9, 17:24) strictly precedes confirmation commit `b3d3389`
   (Jul 9, 19:47); all five systems' results reported, including the failed claim.
   Leak story: found, fixed, pinned by structural tests, bitwise end-to-end audit.
5. **Headline (confirmation, run once)** — table from draft Tab. headline-conf:
   0.376/−0.300/92.5% vs two-stage 0.393/−0.311/92%; paired ΔRMSE −0.0137\*;
   graph ΔNLL −0.0736\* [−0.092, −0.057]; pooled-NLL tie disclosed and diagnosed
   (one τ-tail cell, NLL +27.5 with fine RMSE 0.113 and coverage 0.90).
   Fig: `figures/proposed_confirmation.png`. Source: logs/confirmation_verdict.log.
5b. **What the posterior looks like** — one dev piece, hidden truths against
   the 90% band (49/50, 49/50, 46/50 inside). The dynamics panel shows the
   phrase structure the model exploits; the timing band is honest about how
   little of τ is predictable. Fig: `figures/posterior_example_dev.png`.
6. **Attribution (DEV)** — fixed-mean control 0.390/−0.327 ≈ two-stage level ⇒
   coupling/joint-fitting per se ≈ nothing; the win is **per-piece Bayesian
   feature weighting**; the graph is what keeps confidence honest (−0.017 RMSE /
   −0.067 NLL when removed, paired\*). Per-channel: graph's calibration win ≈
   articulation (coverage 0.784→0.91), recovery win ≈ loudness; timing is a
   plateau **by construction** (rubato lives in the tempo warp T, not in τ).
7. **Robustness of the operating point (DEV)** — masking sweep 50%→LOO:
   ordering stable everywhere; graph significant 50–20% + LOO; embedding value
   grows with density (−0.005→−0.021); calibration flat 0.92–0.93.
   Fig: `figures/masksweep_dev.png`. Source: docs/masking_sweep_results.md.
8. **Why kernels tie (DEV)** — the families nearly coincide where the evidence
   looks; the information is in the graph Λ_G, not the filter shape. The regime
   that moves both axes is changing *connectivity* (chord+VL edges, two-stage
   era) — and even that goes redundant under embeddings.
   Fig: `figures/spectral_overlay_dev.png`.
9. **What we learned about MUSIC (DEV)** — the triangulated tonality result:
   (i) tonal metric as geometry: significantly WORSE; (ii) explicit tonal
   features next to the kernel: nothing; (iii) linear probes: the embeddings
   do NOT encode tonality (scale degree/mode/in-scale at chance) but DO encode
   bass (AUC 0.99), meter, phrase, register. Consistent conclusion: **for piano
   timing/articulation/dynamics, tonal structure carries no measurable marginal
   signal; the structure that matters is rhythm, voicing, and register.**
   Sources: docs/kernel_comparison_results.md, docs/theory_features_results.md.
10. **Boundaries (measured, both sides)** — six downstream tasks: structure helps
    note-level judgments (anomaly AUROC ≤0.995), not whole-performance summaries
    (era honest negative); per-piece adaptation wins interpolation, fails excerpt
    extrapolation (cross-piece head stays the honest tool).
10b. **The calibrated deep rival (DEV, new)** — heteroscedastic MLP + deep
    ensemble on the same information set: 0.450–0.454 / −0.07..−0.10 vs GP
    0.360/−0.404 (both axes, −0.090\*/−0.30\* paired). They land exactly at the
    cross-piece-mean level: what they cannot imitate is per-piece adaptation +
    the graph. Source: docs/deep_baseline_results.md.
10c. **Fresh-pieces replication (new, post-hoc)** — the ladder re-run on 30
    pieces never seen by any decision (positions 50–79 of the same shuffle,
    identity-gated): ordering replicates; graph −0.0187\*/−0.0797\* and
    embeddings −0.0102\*/−0.0334\* both significant on both axes; the graph's
    fresh-pieces NLL contribution (−0.080) matches its confirmed value
    (−0.074). Not preregistered — label it replication, not confirmation.
    Source: docs/replication_set_results.md.
11. **Statistical solidity (DEV recheck, 2026-07-15)** — the headline dev contrast
    is significant in EVERY channel separately; percentile/basic/BCa/Wilcoxon/sign
    all agree; all load-bearing stars survive Benjamini–Hochberg (q=0.05, 19
    contrasts) — the non-survivors are exactly the ones already labeled unproven;
    piece-level coverage min 0.897, none <0.85; composer-clustered bootstrap keeps
    every key contrast significant. Source: logs/robustness_recheck.log.
12. **Phases 2–3 — and Phase 2's blocker fell last night.** Per-(note,channel)
    cell masks are implemented and unit-pinned exact; the specified vibrato NLLS
    estimator (with covariances + identifiability rule) is built and tested; a
    synthetic end-to-end pilot passed: graph beats no-graph on every channel,
    estimator-missing vibrato cells recovered at ~4× better than prior scale
    with 0.87–0.93 coverage, and the open noise-design decision got a measured
    answer (variances-as-given > learned scale, reversing the draft's guess).
    Honest scope: synthetic; real audio (tracker, alignment) untouched; Phase-2
    claims need their own preregistered confirmation. Source:
    docs/phase2_synthetic_pilot.md. Phase-3 conjugate helpers tested,
    z-inference open. Figs: `figures/arch_phase2.pdf`, `figures/arch_phase3.pdf`.
13. **Close** — contribution = standard theory, assembled, measured harder than
    its sources: preregistered one-shot confirmation, exact nesting of the
    baseline, measured attribution, mapped boundaries, and honest negatives kept.

## Verified numbers (single source of truth for the deck)

| Number | Value | Set | Source |
|---|---|---|---|
| Headline RMSE / NLL / cov | 0.376 / −0.300 / 92.5% | CONF | logs/confirmation_verdict.log |
| Two-stage strongest (conf) | 0.393 / −0.311 / 92% | CONF | same |
| Paired ΔRMSE (GP − two-stage) | −0.0137\* | CONF | same |
| Graph ΔNLL inside model | −0.0736\* [−0.092, −0.057] | CONF | same |
| Dev proposed model | 0.360 / −0.404 / 92.7% | DEV | logs/graphgp_v2_report.log |
| Dev per-channel ΔRMSE (GP − two-stage) | τ −0.0089\*, log r −0.0361\*, v −0.0059\* | DEV | logs/robustness_recheck.log |
| Graph value dev (removed) | +0.017 RMSE / +0.067 NLL | DEV | graphgp_first_design.md |
| Embedding value dev (removed) | +0.008 / +0.034 | DEV | same |
| 12-seed replication | pooled moves ≤0.003/≤0.007; all calls unchanged | DEV | logs/dev12_report.log |
| Mask sweep | ordering stable 50%→LOO | DEV | docs/masking_sweep_results.md |
| Kernel study: chord+VL edges | −0.008\*/−0.013\* (two-stage regime) | DEV | docs/kernel_comparison_results.md |
| Tonal metric replacement | +0.009\*/+0.022\* (worse) | DEV | same |
| Theory features | −0.002 ns / +0.001 ns | DEV | docs/theory_features_results.md |
| Probe: tonality in embeddings | R²≈0, AUC≈0.5 (not encoded) | DEV | results/probe_embeddings*.pkl |
| Probe: bass/meter/phrase | AUC 0.99 / R² 0.14–0.35 / 0.48 | DEV | same |
| Anomaly AUROC (proposed) | v 0.995, log r 0.986, τ 0.978 | DEV | docs/downstream_tasks_results.md |
| Calibrated deep baselines | hetero 0.454/−0.069; ensemble 0.450/−0.104; GP wins −0.090\*/−0.30\* paired | DEV | docs/deep_baseline_results.md |
| Posterior example (piece 0) | hidden truths in 90% band: 49/50, 49/50, 46/50 | DEV | figures/posterior_example_dev.png |
| Student-t on the dev tail cell | NLL +34.7 → −0.96 (t-predictive), RMSE/cov unchanged | DEV | results/robust_tail_piece28.pkl |
| Replication set (30 fresh pieces) | ordering replicates; graph −0.019\*/−0.080\*, emb −0.010\*/−0.033\* | REPL | docs/replication_set_results.md |
| Phase-2 pilot (synthetic) | missing vibrato cells recovered ~4× better than prior, cov 0.87–0.93; as-given noise > learned scale | SYNTH | docs/phase2_synthetic_pilot.md |
| Probes, nonlinear (RFF-2048) | tonal verdicts replicate: not encoded | DEV | results/probe_embeddings_rff.pkl |
| Contamination filter | 1036 → 653 performances | — | draft §contam |
| LM pretraining ppl | 10.85 (scaled 9.66: no downstream gain) | — | draft ch. results |

## The ten questions most likely to come, with the prepared line

1. **"Why not a deep model with a variance head?"** NOW MEASURED (2026-07-16):
   a heteroscedastic MLP head and a 5-member deep ensemble on the identical
   information set land at 0.450–0.454 RMSE / −0.07 to −0.10 NLL vs the GP's
   0.360/−0.404 — GP wins both axes, −0.090\*/−0.30\* paired. They sit exactly
   at the cross-piece-mean level: coverage fine, note-ranking of uncertainty
   far behind — per-piece adaptation + graph is what they cannot imitate.
   Residual honest scope: frozen-embedding heads; a fine-tuned end-to-end deep
   model remains untested.
2. **"What is novel vs Borovitskiy?"** Nothing in the kernel algebra — say it
   first. The novelty: the domain instantiation, the measurement standard
   (calibration + preregistration + bitwise audits, beyond the source papers),
   the exact-nesting result, and the measured findings. "Orthodoxy, applied and
   measured harder than its authors did."
3. **"Your NLL advantage vanished."** Yes — reported as a failed preregistered
   claim, diagnosed to one τ-tail cell (Gaussian tail; likelihood-level, hits all
   systems in the 30%-sweep replica identically). What IS confirmed: graph's
   ΔNLL −0.074\* inside the model, coverage in band, calibration profile
   indistinguishable dev↔conf. Never quote dev −0.404 as headline.
4. **"Lenient decision rule."** Content criticizable, timing not: committed
   before data (git-ordered hashes above); the inconvenient outcome is printed.
5. **"Pooled RMSE is mostly articulation."** True for magnitude, and the draft
   says so; but significance holds in every channel separately (τ/log r/v all \*),
   and variance-standardized pooling makes the win *stronger* (27/30 pieces).
6. **"40% masking arbitrary."** Inherited, then measured: sweep 50%→LOO,
   ordering stable everywhere. "Arbitrary but not lucky."
7. **"Student-t exists — why not applied?"** The confirmation set is spent;
   applying the fix post hoc and re-scoring is exactly the selection the protocol
   forbids. Both halves are now measured on dev: no-harm on tail-free data
   (small premium), and the payoff on the dev tail replica — the piece-28 cell
   goes from Gaussian NLL +34.7 to −0.96 under the t-predictive with RMSE and
   coverage unchanged, EM weights isolating the outlier notes. It awaits its
   own preregistered confirmation before touching any thesis number.
8. **"What did you learn about music?"** The tonality triangulation (slide 9) +
   the two boundaries + per-channel decomposition. The τ plateau is by
   construction: rubato lives in the warp.
9. **"Identifiability of per-piece evidence?"** g(0)=1 resolves the provable
   non-identifiability; the rest is handled by floors/safeguard with measured
   failure modes reported (1-in-120 collapse; the tail the guard provably cannot
   catch). Point to them before he does.
10. **"Phase 2 realistic?"** The draft is a pre-mortem: per-(note,channel)
    missingness (mask generalization needed — exactness survives), alignment
    error correlated exactly like the signal (fold scale into noise, or withhold
    τ), estimator-output targets (claim weaker in kind). Phase 2 is gated so a
    negative costs the thesis nothing; a Phase-2 claim needs a fresh confirmation
    set.

## Traps (do not say)

- "The GP beats the pipeline on calibration" (pooled NLL tied; the graph's
  *contribution* is what's confirmed).
- "The embeddings know music theory" (probes: tonality NOT encoded; say rhythm/
  voicing/register).
- "Confirmed on both axes" (adopted under the preregistered RMSE-wins clause).
- Any dev number without the word "development".
- "Phase 3 not started" (conjugate helpers are tested; the z-inference loop is
  not started) — and never present GP-first as planned-from-start: the two-stage
  pipeline was the development path, the reformulation came from the literature
  review, and the confirmation adjudicated it. The selection story is a strength.

## Deck figure inventory (all committed)

- `docs/thesis/figures/arch_phase1.pdf` (+2,3) — model architecture, phase story
- `docs/thesis/figures/proposed_confirmation.png` — the one-shot confirmation
- `docs/thesis/figures/masksweep_dev.png` — masking-level robustness (NEW)
- `docs/thesis/figures/spectral_overlay_dev.png` — why kernels tie (NEW)
- `docs/thesis/figures/digest_headline.png`, `digest_kernels.png`,
  `digest_channels.png`, `digest_collapse.png` — development-era digests
- `docs/thesis/figures/proposed_reliability.png`, `proposed_pit.png` —
  calibration profile, dev vs conf
- `docs/thesis/figures/posterior_example_dev.png` — what the posterior looks
  like on one piece (NEW; hidden truths inside the band)
- `docs/thesis/draft.pdf` — the full draft (45 pp), compiled clean tonight
