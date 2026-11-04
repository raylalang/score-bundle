# Motif study, stage one, first pilot (DEVELOPMENT ONLY, exploratory, no claim)

2026-11-04, about three hours from first alignment to this record.
`scripts/eval_motif_pilot_dev.py` (hypotheses H1 and H1b and the
falsification rule in the docstring, committed before the run); cells in
`results/motif_pilot/` (gitignored pickles; `align.md`, `report.md` and
`effects.json` kept); logs `logs/motif_pilot.shard*.log`; figure
`docs/thesis/figures/proposal_pilot_results.png`; proposal deck frame 10.

## What was run

The score graph gains one edge family from the BPS-motif annotations
(all 32 Beethoven sonata first movements): a note in occurrence k of
type q gets the relative position u = (b − b_k^start)/(b_k^end −
b_k^start), and notes of two different occurrences of the same type are
joined by exp(−(u_i − u_j)²/2ℓ_u²), ℓ_u = 0.1 fixed (many-to-many, no
alignment of unequal occurrences, nothing learned inside the family). The
graph is W = W_time (+ W_pitch) + w_m W_motif with log w_m one more graph
parameter learned by the per-piece evidence. Everything else is the
piano model with score features only (no music-model embeddings, so no
contamination question and no GPU): ICM coupling, per-piece bias column,
floored noise, exact evidence, the published predictive-variance floor.

- Data: the ASAP Beethoven first movements with an annotated
  counterpart, first 400 matched notes of the first aligned performance
  per movement (the thesis's excerpt convention). ASAP score notes were
  matched to the annotation's note list by (crotchet onset, pitch) after
  a global offset search: 21 of 23 movements match at 0.94 to 1.00 of
  their excerpt notes (`results/motif_pilot/align.md`); 23-1 (0.81) and
  32-1 (0.64) were dropped by the 0.85 rule. Folders 3-1 and 24-1 are in
  the piano confirmation set and were excluded before anything ran.
- Arms, paired on identical masks: `time` (the time-only Gaussian graph,
  the attribution winner), `time_pitch` (the thesis's score graph),
  `time_motif`, `time_pitch_motif`.
- Masking, two seeds each: `occ` hides whole occurrences (each
  occurrence of a type with at least two occurrences hidden with
  probability 0.4, at least one left observed; all non-motif notes
  observed); `random` is the thesis protocol (40 percent at random).
- Scored on hidden motif notes whose type has another occurrence with an
  observed note: Gaussian log score and RMSE per channel, paired per
  piece (seeds averaged), bootstrap B = 2000 over 21 pieces.
- 336 fits, 14 shards, about 35 minutes on a loaded machine (about 60 s
  per fixed-graph fit, 110 s per fit with the learned weight).

## Primary rows (occurrences hidden, hidden motif notes, vs the time-only graph)

| arm | timing ΔNLL [95%] | timing ΔRMSE (s) | articulation ΔNLL | articulation ΔRMSE | velocity ΔNLL | velocity ΔRMSE |
|---|---|---|---|---|---|---|
| time + pitch | −0.014 [−0.046, +0.016] | −0.0026 [−0.0054, −0.0003]* | −0.003 [−0.067, +0.055] | +0.014 [−0.017, +0.046] | +0.041 [−0.022, +0.108] | +0.005 [+0.001, +0.009]* |
| time + motif | **−0.117 [−0.245, −0.006]*** | **−0.0026 [−0.0054, −0.0003]*** | −0.056 [−0.128, +0.014] | −0.033 [−0.066, −0.007]* | +0.069 [+0.005, +0.138]* | +0.002 [−0.001, +0.004] |
| time + pitch + motif | −0.083 [−0.232, +0.013] | −0.0058 [−0.0145, −0.0005]* | −0.065 [−0.192, +0.066] | −0.027 [−0.073, +0.018] | +0.174 [+0.003, +0.399]* | +0.006 [+0.001, +0.011]* |

Levels of the time-only arm on the same notes: timing NLL −1.156, RMSE
0.0755 s, coverage 0.970; articulation 0.938 / 0.637 / 0.887; velocity
−1.017 / 0.086 / 0.878. The motif arm's velocity coverage falls to 0.823.

Under the thesis's random masking (hidden motif notes): timing ΔNLL
+0.090 [−0.201, +0.544] (one wide outlier piece), timing ΔRMSE −0.0022*,
articulation ΔRMSE −0.018*, velocity ΔNLL −0.024*. Full tables in
`results/motif_pilot/report.md`.

Learned weight w_m (occurrences hidden, geometric mean of two seeds):
switched off (< 0.1) on 1-1, 2-1, 7-1, 10-1, 12-1, 18-1, 27-1, 31-1;
above 2 on 4-1 (6.6), 9-1 (2.1), 11-1 (9.1), 15-1 (32), 17-1 (2.3),
22-1 (6.9); the rest between 0.2 and 1.4. The evidence treats the family
as a switch per piece, as it treats the feature kernels.

## Reading (2026-11-04)

1. **H1 is supported, at the edge.** When whole occurrences are hidden,
   joining their notes to the observed returns of the same motif lowers
   the timing log score on those notes by 0.117 nats against the
   time-only graph, interval [−0.245, −0.006]; the pitch term gives
   −0.014 on the same notes. H1b (timing RMSE) is supported too, but
   the pitch arm gives the identical RMSE gain, so on timing accuracy
   the motif edges do what the pitch term does, and the extra is in
   calibration.
2. **The clearest gain is articulation accuracy**: RMSE 0.637 → 0.611
   (−0.033 [−0.066, −0.007]), about five percent, with the log score
   moving the same way but not significantly.
3. **Velocity gets worse in calibration**: +0.069 nats, coverage 0.878
   → 0.823. The motif edges tie velocities of returns together more
   tightly than performers play them, so the bands narrow too far. A
   per-channel motif weight (one w_m per channel, or the family entering
   through its own coregionalization block) is the obvious next design
   step and was not in this pilot.
4. **Under random masking the family is mostly invisible**, as it
   should be: with 60 percent of every occurrence observed, the two-beat
   time window already interpolates inside the occurrence. The study's
   information is in the occurrence-hidden case, which is also the
   application's case (a passage not yet played, its earlier statements
   heard).
5. **Scope and what this is not.** Development side only, 21 pieces, two
   seeds, 400-note excerpts, score features without the music-model
   embeddings, annotated occurrences only. The primary interval's lower
   edge is 0.006 nats from zero; a registered test needs its power check
   first (the strings-and-winds power table says about six pieces for an
   effect of this size at this spread, but the spread here is wider).
   Nothing registered was touched, no confirmation pool was used, and
   no thesis claim follows from this record.

## What this changes

The proposal's mock figure (deck frame 9) is replaced in the reading by
frame 10, the measured pilot. The recommendation stands: the motif study
on annotations is the bounded end-goal study, now with one measured
pilot behind it; its next design step is per-channel weighting, and
stage two (predicted occurrences from the ISMIR model) waits on a
registered stage one.
