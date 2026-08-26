# Phase 2: within-note time variation (drift study)

**DEV ONLY — EXPLORATORY.** 2026-08-27. Script: `scripts/eval_drift_dev.py`
(read-only over the dev caches + URMP annotations + audio; no GP fits; the
confirmation split untouched). Prompted by the committee comment (meeting
2026-08) that the constant-parameter vibrato model of eq:cents-curve /
eq:vibrato may be too simple, since loudness and intonation shift over time
within a note.

## Question

The six channels are per-note time-averages. How much real time variation
does that compression discard, is it music or tracker artifact, and does the
discarded structure belong in the graph GP (i.e., does it correlate across
notes)?

## Method

Every dev note identifiable in both the tracked and the ground-truth curve
is refit with the basis `[1, t-mean, sin, cos]` at its fitted rate — the
eq:vibrato model plus one linear intonation-drift term — separately on the
tracker frames and on the GT curve (slope + Gauss–Newton SE each). Loudness:
OLS slope over the four chunk log-RMS values of eq:loudness, with SE from
the chunk residual. Structure: lag-1 autocorrelation of the slopes along the
note sequence per track; slope-vs-slope / slope-vs-duration correlations.
Long notes additionally get a first-half vs second-half extent split.

## Results

**A. The drift is real music, not tracker noise** (n = 5,145 notes in both
curves): a significant slope (|s| > 2 SE) on 65.5% (tracker) / 67.1% (GT) of
notes independently; on the 2,738 both-significant notes, sign agreement
97%, Spearman 0.91, winsorized-2% Pearson 0.88 (the raw Pearson 0.25 is
tail-dominated by ~130 cents/s note-edge slopes and is not the right
statistic). Median GT total drift over a note: 10.5 cents (q90 31.9) —
roughly an order of magnitude above the c cell's median reported SE
(0.9 cents). Per family (median |slope| cents/s / total cents): strings
15.3/10.2, winds 10.1/7.8, brass 16.1/13.7.

**B. Loudness moves even more, but much of it is envelope** (n = 16,925):
41.3% of notes carry a significant chunk slope; the median within-note
change (0.65 log-RMS) is 137% of the across-note sd of ell (0.47). Of the
significant slopes 65% are negative — decay envelope, not expressive
shaping — rising to 81% in brass (median slope −2.62/s); strings are nearly
balanced (57% negative), which is where genuine swells live.

**C. The decisive negative: the missing structure is graph-white.** Lag-1
autocorrelation along the note sequence: intonation drift slope +0.03
(55 tracks), loudness slope +0.06 (78 tracks). Drift and loudness slopes
are uncorrelated with each other (−0.042, n = 5,617) and drift magnitude
with duration (−0.049). Compare τ (lag-1 +0.59), the channel where the
graph earns its keep.

**D. Vibrato extent growth is mild:** on 1,941 long notes the second-half
extent exceeds the first on 53% (median difference +0.27 cents; medians
4.6 vs 5.9 — a skewed tail of ramping notes, mostly already absorbed by the
δ_vib gate).

## Verdict

1. **The committee comment is correct and now quantified:** the per-note
   channels are time-averages of quantities that measurably move within the
   note; c's 0.9-cent reported precision is precision about the average,
   not about a constant.
2. **The registered claims are not threatened:** the mismatch is priced
   into the cell variances (they are residual-based, so drifting notes
   report themselves as more uncertain — this is why calibration held),
   and the quasi-truth passes through the same parameterization, so the
   graph-vs-no-graph contrasts compare like with like.
3. **No new GP channel:** a drift/decay-slope channel would be graph-white
   (C above) — the graph prior could add nothing, so per-note resolution
   is empirically the right level for the *graph* model. Within-note
   structure belongs to the frame-level likelihood of Phase 3 (the
   amplitude envelope A_i(t) and a time-varying pitch curve model).
   Drift/decay slopes remain post-confirmation candidates under the δ_vib
   template (include as output, claim nothing) if a transcription use-case
   wants them.
4. **The registered estimator is unchanged** (tag
   phase2-registration-2026-08-17); the confirmation one-shot runs on
   eq. 3.32/3.33 exactly as frozen.

Thesis: one paragraph in §3.9 (after the observation-noise discussion)
carries points 1–3 with these numbers.
