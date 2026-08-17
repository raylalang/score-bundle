# Phase-2 evaluation design — REGISTRATION (2026-08-17)

> **Status: REGISTERED as of the commit that carries this banner.**
> Registration = this committed revision freezes the claims, systems,
> estimator chain, and decision rule BEFORE any confirmation piece is
> touched. The 13-piece confirmation pool (frozen 2026-08-06, unit-pinned)
> remains untouched at registration time; running it is a separate,
> deliberate act, performed ONCE, with every number reported whatever the
> outcome. Development evidence backing each frozen choice is cited inline.
> (Document history: drafted 2026-08-05 before any real Phase-2 data;
> blockers closed 2026-08-06..13 on development evidence only.)

## Corpus

**URMP** (44 multi-instrument pieces, separately recorded monophonic tracks;
`../data/urmp/Dataset/`, downloaded 2026-08-06 via Dryad and MD5-verified —
the registration form was never needed). Each track supplies: audio (48 kHz mono), frame-level
ground-truth F0 (46 ms windows, 10 ms hop), note-level onset/pitch/duration
annotations, and the score MIDI. Loader:
`score_bundle.phase2.urmp.load_urmp_meta`.

**Contamination / split rule (design decision, fixed now):** URMP reuses
compositions across arrangements (e.g. five Art-of-the-Fugue variants, three
Rejouissance). Splits are therefore by **composition**, never by arrangement:
all variants of one composition land on the same side.

**FROZEN SPLIT (2026-08-06, data-blind, deterministic).** Constructed by
`score_bundle.phase2.splits.construct_split` (seed 0; inputs = only the
published Table 1 metadata; unit-pinned to these literals):

- **Confirmation pool — 7 compositions, 13 pieces, UNTOUCHED from here on:**
  Chorale (43), Elise (33), Fugue (28, 29, 30, 32, 34), Jesus (9),
  Pirates (24, 25), Surprise (15, 16), Waltz (14).
- **Development — 20 compositions, 31 pieces:** everything else.
- Constraints verified: Jupiter pinned to development (its violin track was
  used in the 2026-08-06 smoke test); both sides contain every ensemble type
  (duet/trio/quartet/quintet) and all three instrument families
  (strings/woodwind/brass); confirmation holds 13 of 44 pieces.

## Targets and channels (FROZEN)

**The evaluated bundle is six channels per note of the known score:**
`[c, log γ, log f_vib, ℓ, τ, δ_vib]` — intonation (cents), vibrato extent
and rate (log), loudness (log RMS, centred per track), timing (onset-anchored
warp residual, seconds), vibrato onset delay (seconds). Per-(note, channel)
cell masks throughout (`MultiOutputGraphGP` 2-D mask path, unit-pinned).
Articulation log r stays out of the Phase-2 bundle (no measured offset
chain on this corpus). Channel decisions closed on development evidence:

- **τ ADOPTED** (policy option 1): URMP's annotated onsets anchor the warp
  with no audio aligner — score↔performance matching (exact order or pitch
  DTW, `phase2/warp.py`), local ±8-note leave-one-out tempo line
  (draft eq:localwarp); the aligner error enters the noise row as the tempo
  line's OLS predictive variance. Feasibility: 76/78 dev tracks, median
  τ std 79 ms, lag-1 +0.59 (`results/tau_feasibility_dev.md`).
- **δ_vib IN**, by the criterion pre-stated in `scripts/eval_delta_vib.py`
  (committed before any number was computed): coverage 94.8%/96.6% of
  vibrato-identifiable notes (GT/pyin), median pyin-vs-GT delay agreement
  18 ms, no intonation degradation, spread 8.1× the reported SE
  (`results/delta_vib_dev.md`). Values come from the GATED estimator
  (`fit_vibrato_note_gated`, the draft eq:vibrato model); its delta-method
  variance is the noise row; unidentifiable delays are missing cells.

Estimation chain (all thresholds measured, none tunable at confirmation
time): `extract_f0` (pyin, 10 ms hop, per-instrument ranges) → voiced frames,
lowest confidence quintile discarded → per-note NLLS `fit_vibrato_note`
(ungated; supplies c, log γ, log f_vib with Gauss–Newton/delta-method
variances and the ≥8-frame/≥1.5-cycle identifiability rule) → the
GT-validated octave-failure rule (|c| > 150 cents ⇒ missing) → gated fit for
δ_vib → loudness from four-chunk log RMS → τ from the onset-anchored warp.
Estimator-supplied variances are used **as given** (primary variant;
synthetic pilot and real-data re-confirmation both prefer it); the learned
per-channel scale is reported alongside as the documented fallback.

**Tracker calibration — MEASURED (2026-08-06, development side,
`results/tracker_calibration_dev.md`):** pyin vs URMP ground truth on all
101 development tracks (78 unique recordings after deduplication): median |dev| 2–5 cents per instrument (an order
below vibrato extents); voicing recall ≥ 0.95 everywhere; pyin's confidence
is informative (median error monotone in probability; gross-error rate
9.6% → 0.4% across quintiles). **Decisions adopted:** estimator variances
as-given; frames in the lowest confidence quintile discarded before the
per-note NLLS fit; low-register tracks flagged (the one outlier is K515's
second viola — octave slips). Bonus finding: arrangements of one composition
share identical track recordings, so the composition-level split is
literally required.

## The τ policy — RESOLVED (option 1 adopted)

The warp must come from audio-to-score alignment; its error enters τ
directly and is correlated along score time (the known largest threat).
Option 1 was measured feasible and adopted (see Targets above): annotated
onsets anchor the warp, no audio aligner, aligner error in the noise row.
Honest limitation, stated in the thesis: the noise row is diagonal, so it
carries the error's scale but not its correlation along score time.

## Evaluation

Identical discipline to Phase 1: held-out imputation at the cell level, both
axes (RMSE + NLL/coverage/PIT with the predictive-variance floor), paired
per piece with bootstrap CIs; calibration is the primary axis because the
targets carry real estimator noise. Honest scope: recovery = agreement with
the estimator's targets, weaker in kind than Phase 1 — state it wherever
numbers appear.

## REGISTERED CLAIMS AND DECISION RULE (frozen at this commit)

**Protocol constants.** Confirmation pool = the frozen 13 pieces
(composition-level, `phase2/splits.py`); unique-recording deduplication by
GT-F0 MD5; tracks with fewer than 30 usable notes dropped; 30% of notes
hidden; seeds (0, 1); mask draw exactly as in `scripts/eval_phase2_real.py`
at this commit. Systems, all three reported in full: **graph GP (as-given)
= PRIMARY**, graph GP (learned scale), no-graph ablation. Contrasts paired
per (track, seed); bootstrap B = 2000, rng seed 31, 95% CI ("starred" = CI
excludes 0). Both scorings reported (vs estimator targets with predictive
sd — primary; vs GT-derived quasi-truth with latent sd). Code state = this
commit; OMP_NUM_THREADS=4.

**Claims (as-given vs no-graph unless stated):**
- **C1 (recovery, intonation):** paired dRMSE on `c` negative and starred.
- **C2 (calibration, vibrato):** paired dNLL negative and starred on
  `log γ` and on `log f_vib` (both channels).
- **C3 (coverage):** as-given coverage@90 in [0.85, 0.95] on every channel
  of the six (vs estimator targets).
- **C4 (timing calibration, secondary):** paired dNLL on `τ` negative and
  starred. (Development basis: as-given dNLL −0.292\*, dRMSE −0.002 ns —
  the graph's value on the adopted timing channel is calibration, matching
  the Phase-1 story; no recovery claim is registered for τ.) `δ_vib`
  carries **no registered claim**: its development contrasts are
  graph-neutral (dRMSE −0.001 ns, dNLL +0.057 ns, as-given); it remains a
  bundle channel and all its numbers are reported. C4 does not gate C1–C3.

**Decision rule.** C1, C2, C3 all pass ⇒ the thesis's Phase-2 headline
("the unchanged graph prior extends to real audio with calibrated
uncertainty, at confirmation level") is CONFIRMED. Any failure is reported
verbatim next to the claim it fails. One shot: no reruns, no added seeds,
no post-hoc masks or filters; the pool is spent whatever the outcome.

**Development basis for the registered claims (6-channel run, as-given vs
no-graph, `results/phase2_real_results.md` @ this commit):** C1 dRMSE
−0.891 [−1.513, −0.378]\*; C2 dNLL −3.772\* (log γ) and −0.426\* (log f);
C3 coverage 0.88–0.91 across the six channels; C4 τ dNLL −0.292\*.
Reported-not-claimed (dev): loudness dRMSE −0.003 ns with dNLL +0.042
against the graph under as-given; δ_vib graph-neutral; brass intonation
family cell ns in the 6-channel run. The learned-scale variant stars all
six recovery contrasts but remains the non-default (worse-calibrated vs
quasi-truth, 0.72–0.86 vs 0.82–0.86 coverage).

**Exploratory (reported, never confirmed):** the circle-of-fifths metric on
the intonation channel — the first test of a music-theoretic *geometry*
where the target is pitch itself; and the per-family breakdown.

## Registration checklist (all closed)

1. ~~Full URMP download~~ — done 2026-08-06 (Dryad; 44/44 load,
   `evidence/logs/urmp_load_check.log`).
2. ~~Tracker-calibration study~~ — done (as-given + confidence filter).
3. ~~Frozen split list~~ — done (`phase2/splits.py`, unit-pinned).
4. ~~τ policy~~ — adopted (option 1); ~~δ_vib~~ — IN by pre-stated criterion.
5. ~~Final claim set + decision rule~~ — frozen above, informed by the
   development results (`results/phase2_real_results.md`, 6-channel run).
