# Phase-2 evaluation design — DRAFT, not yet registered (2026-08-05)

> **Status: design document.** Written before any real Phase-2 data has been
> modelled, per the thesis rule that Phase-2 claims need their own fresh,
> preregistered confirmation set. Nothing here is registered yet:
> registration = freezing the claims/systems/decision rule in a committed
> revision of this file BEFORE the confirmation pieces are touched. Until
> then this is a working draft; every choice below is open to revision from
> development evidence only.

## Corpus

**URMP** (44 multi-instrument pieces, separately recorded monophonic tracks;
`../data/urmp/`, download pending the registration form — see
`URMP_doc.pdf`). Each track supplies: audio (48 kHz mono), frame-level
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

## Targets and channels

Per note of the known score (aligned support), the Phase-2 vector
(draft eq:phase2-channels): timing τ, articulation log r, loudness ℓ
(log RMS), intonation c (cents), vibrato rate f_vib and extent γ; the onset
delay δ_vib remains channel-candidate only (identifiability question —
decide on development data). Per-(note, channel) cell masks throughout
(`MultiOutputGraphGP` 2-D mask path, unit-pinned).

Estimation chain: `extract_f0` (pyin, 10 ms hop) → voiced-frame cents
(`cents_from_f0`) → per-note NLLS (`fit_vibrato_note`) with Gauss–Newton
variances and the identifiability rule. Loudness from the note's RMS
envelope. Estimator-supplied variances are used **as given** (synthetic-pilot
default); the learned per-channel scale is the documented fallback for a
mis-calibrated tracker.

**Tracker calibration — MEASURED (2026-08-06, development side,
`results/tracker_calibration_dev.md`):** pyin vs URMP ground truth on all
~90 development tracks: median |dev| 2–5 cents per instrument (an order
below vibrato extents); voicing recall ≥ 0.95 everywhere; pyin's confidence
is informative (median error monotone in probability; gross-error rate
9.6% → 0.4% across quintiles). **Decisions adopted:** estimator variances
as-given; frames in the lowest confidence quintile discarded before the
per-note NLLS fit; low-register tracks flagged (the one outlier is K515's
second viola — octave slips). Bonus finding: arrangements of one composition
share identical track recordings, so the composition-level split is
literally required.

## The τ policy

The warp must come from audio-to-score alignment; its error enters τ
directly and is correlated along score time (the known largest threat).
Policy to settle on development data, in order of preference:
1. If URMP's note-level onset annotations are reliable, derive the warp from
   annotated onsets (no audio aligner needed) and keep τ with the aligner
   error folded into its noise row;
2. otherwise withhold τ from Phase-2 claims and report the audio-native
   channels only.

## Evaluation

Identical discipline to Phase 1: held-out imputation at the cell level, both
axes (RMSE + NLL/coverage/PIT with the predictive-variance floor), paired
per piece with bootstrap CIs; calibration is the primary axis because the
targets carry real estimator noise. Honest scope: recovery = agreement with
the estimator's targets, weaker in kind than Phase 1 — state it wherever
numbers appear.

## Claims to be registered (to be finalized after development)

Sketch, to be frozen at registration time:
- C1: the graph prior improves held-out cell recovery of the intonation
  channel vs the no-graph ablation (paired, significant).
- C2: it improves calibration (NLL) on the vibrato channels where cells are
  missing by identifiability.
- C3: coverage within [0.85, 0.95] at nominal 90%.
- Exploratory (reported, not confirmed): the circle-of-fifths metric on the
  intonation channel — the first test of a music-theoretic *geometry* where
  the target is pitch itself.

## What blocks registration

1. Full URMP download (form; Ray).
2. Tracker-calibration study on development pieces.
3. τ-policy decision and δ_vib in/out decision (development evidence).
4. Frozen split list (compositions), committed with the final decision rule.
