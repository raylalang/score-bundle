# Notion sync content — prepared 2026-08-20 (for Ray/Cowork to apply)

> Supersedes `notion_sync_2026-08.md` (whose unapplied items are folded in
> here with their outcomes). The "Score-Bundle Models" Notion page is owned
> by the Cowork workflow; this file is the complete delta so the applier
> reconstructs nothing. Everything below is in the repo; this is the
> summary form. All Phase-2 numbers are development-labeled.

## Headline changes since the last applied sync (July)

**1. Phase 2 ran end to end on real audio and its claim set is REGISTERED.**
- Corpus: URMP (downloaded 2026-08-06 via Dryad, 44/44 load; the
  registration form was never needed). Dev/confirmation split frozen at the
  composition level (13-piece confirmation pool, untouched); arrangements of
  one composition share identical track recordings, so the composition split
  is literally required.
- Measured groundwork: tracker calibrated against URMP ground truth
  (2–5 cents per instrument, confidence predictive → variances as-given +
  lowest-quintile frame filter); GT-validated octave-failure rule
  (|c| > 150 cents ⇒ missing; 0/358 slips agree with truth).
- **The evaluated bundle is six channels** `[c, log γ, log f, ℓ, τ, δ_vib]`:
  timing adopted via an onset-anchored leave-one-out warp (76/78 dev tracks,
  no audio aligner; median residual 79 ms, lag-1 +0.59; aligner error in the
  noise row), and the vibrato onset delay adopted by a criterion committed
  before its numbers existed (identifiable on 95%/97% of vibrato-identifiable
  notes on GT/tracked curves; 18 ms median tracker-vs-truth agreement) using
  a gated estimator that matches the thesis equation exactly.
- **Six-channel dev results (as-given noise variant = declared default,
  paired vs the no-graph ablation):** recovery significant on intonation and
  both vibrato channels (−0.89 cents / −0.26 / −0.30); calibration
  significant on both vibrato channels and on timing (dNLL −3.8 / −0.43 /
  −0.29); coverage 0.88–0.91 at nominal 90% on all six. Reported honestly:
  loudness/timing/delay recovery ns; the loudness NLL cell is +0.04
  *against* the graph (starred — the bundle's one adverse cell); brass
  intonation ns in the family breakdown.
- **Registration (2026-08-17, git tag `phase2-registration-2026-08-17`):**
  claims frozen — C1 intonation recovery; C2 vibrato calibration (both
  channels); C3 coverage in [0.85, 0.95] on all six; C4 timing calibration
  (secondary, non-gating); δ_vib deliberately carries no claim. One shot,
  every number reported. The 13-piece pool is UNSPENT; a guarded runner is
  staged so the run takes one afternoon when the spend is agreed.

**2. Circle-of-fifths geometry: first positive (exploratory, dev).**
The tonal metric that *hurt* on piano expression *helps* on intonation, both
axes (paired tonal − plain, as-given: dRMSE −0.21 cents, dNLL −0.05, both
significant) — and re-imposes the replacement penalty on timing. First
evidence that a music-theoretic *geometry*, not only music-theoretic edges,
earns its place. Adoption would need its own preregistered confirmation.

**3. Harmonic-edge question CLOSED (no adoption).** The completed multi-rate
sweep gave a density gradient, not a uniform win (wins at ≤30% hidden, tie
at the 40% operating point, nothing at 50% plus one guard-invisible fit
collapse). Prereg criterion not met → thesis keeps the plain graph.

**4. Posterior decomposition consolidated (thesis §5.3).** Exact
per-component split of the posterior: features carry the mean, the graph
calibrates; graph × embedding components near-orthogonal (complements);
coupling's value is the velocity channel only; dominance is per-piece.

**5. Full-repo audit (2026-08-13..20).** Every quoted number re-verified
against its evidence log; reproduction tolerances measured and recorded
(anchor tables byte-identical; rate shares ±0.04 across BLAS conditions);
committee-level math pass over the whole model chapter (three errors
corrected, incl. scoping the exact-nesting claim to the shared-shape slice);
notation collisions resolved; obsolete records pruned under the
delete-records-keep-evidence rule; thesis at 54 pp, 178 tests, bitwise
leak audits green.

## Standing state (unchanged)

Thesis model = multi-output graph GP; Phase-1 preregistered confirmation
passed (RMSE 0.376 vs 0.393*, graph NLL −0.074*, coverage 0.925);
replication set replicates; downstream tasks re-validated; deep baselines
lose on both axes.
