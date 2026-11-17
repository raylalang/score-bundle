# Motif study, stage two: edges from predicted occurrences (DEVELOPMENT ONLY, exploratory, no claim)

2026-11-17, 14:23 to 15:04 JST (14 shards, `OMP_NUM_THREADS=2`, 252
fits). `scripts/eval_motif_pilot_dev.py --occ-source pred --pred-q 86`
(hypothesis H3 and its falsification rule in the docstring, committed at
d147ee1 before the run), runner `scripts/run_motif_stage2.sh`; cells in
`results/motif_stage2/` (gitignored pickles; `align.md`, `report.md`,
`effects.json` kept); logs `logs/motif_stage2.shard*.log`.

## What was run

The edge family is built from the ISMIR 2026 motif model's held-out
predictions instead of the annotations. The predictions are the
camera-ready sweep's cross-validated outputs
(`/home/ray/Research/ISMIR/graph-motif-discovery/runs/camera_ready_sweep/sweep_q86/fold*/motifs/NN-1.json`,
occurrence quantile 0.86 = the paper's configured `tau_occ_quantile`;
every piece is predicted by the fold that held it out). A predicted
occurrence is a list of (crotchet onset, pitch) notes; it is mapped to
the excerpt by the same key after the piece's alignment offset, its span
is the [min, max] onset of the matched notes, its type is the model's
pattern index, and occurrences with fewer than two matched notes are
dropped. The relative position u, the Gaussian in u (ℓ_u = 0.1) and the
two forms of the family are those of pilots one and two. The masks and
the scoring set stay on the ANNOTATED occurrences, identical to the
pilot-two cells (same seeds 0 to 3; the time-only arm's rows are
bit-identical to pilot two's), so the only change is the edge source.

- Arms: `time`, `time_motif_pred` (shared weight), `time_motif_pc_pred`
  (per-channel block); occurrence-hidden scheme; four seeds; 21 pieces.
- Membership of predicted against annotated motif notes, note level
  (`results/motif_stage2/align.md`): precision 0.12 to 1.00, recall 0.44
  to 0.99, F1 median 0.78; two pieces have precision below 0.2 (30-1 at
  0.18 and 31-1 at 0.12, where the model predicts 136 motif notes
  against 39 and 25 annotated).

H3 (fixed before the run): the predicted-occurrence edges give a timing
log-score gain against `time` whose point estimate is at least half of
the annotated edges' gain on the same cells; falsification: the interval
covers zero.

## Rows (occurrences hidden, hidden motif notes, vs the time-only graph, four seeds)

| arm | timing ΔNLL [95%] | timing ΔRMSE (s) | articulation ΔNLL | articulation ΔRMSE | velocity ΔNLL | velocity ΔRMSE | velocity coverage |
|---|---|---|---|---|---|---|---|
| time + predicted motif (shared weight) | −0.034 [−0.085, +0.003] | +0.0000 [−0.0007, +0.0008] | −0.025 [−0.063, +0.013] | −0.0135 [−0.0317, +0.0030] | −0.004 [−0.051, +0.035] | −0.0003 [−0.0018, +0.0012] | 0.865 |
| time + predicted motif (per channel) | +0.036 [−0.010, +0.095] | +0.0014 [+0.0005, +0.0024]* | −0.007 [−0.039, +0.023] | −0.0028 [−0.0089, +0.0031] | −0.019 [−0.049, +0.002] | −0.0008 [−0.0027, +0.0003] | 0.897 |
| for comparison, annotated edges on the same cells (pilot two, shared weight) | −0.071 [−0.230, +0.097] | −0.0027 [−0.0055, −0.0004]* | −0.047 [−0.117, +0.025] | −0.0345 [−0.0600, −0.0125]* | +0.033 [−0.012, +0.078] | +0.0001 [−0.0021, +0.0022] | 0.835 |
| annotated, per channel (pilot two) | −0.005 [−0.076, +0.054] | +0.0010 [−0.0041, +0.0058] | −0.044 [−0.113, +0.004] | −0.0189 [−0.0381, −0.0053]* | −0.086 [−0.200, −0.018]* | −0.0057 [−0.0106, −0.0023]* | 0.944 |

Time-only levels as in pilot two (timing −1.240 / 0.0698 s / 0.970;
articulation 0.998 / 0.732 / 0.891; velocity −1.015 / 0.0863 / 0.878).

Per piece (shared weight, predicted against annotated edges, the power
script's per-piece differences): the timing log-score differences
correlate at +0.59 across pieces and the articulation RMSE differences at
+0.86, with 15 of 21 pieces below zero in both; the predicted edges act
on the same pieces as the annotated ones, at about half the size
(15-1: −0.43 against −0.98 on timing; 4-1: −0.11 against −0.19 on
articulation). The split by membership F1 (above or below the median
0.78) does not order the effect: timing −0.058 above and −0.009 below,
articulation −0.009 above and −0.019 below. The learned weight w_m is
above 1 on 15-1 (7.8), 4-1 (4.6), 22-1 (4.2), 17-1 (1.3) and 31-1 (13.1,
the piece with precision 0.12), and below 0.1 on ten pieces.

## Reading (2026-11-17)

1. **H3 is not supported by the rule written before the run.** The
   predicted edges' timing log-score gain is −0.034 with interval
   [−0.085, +0.003], covering zero, and its point estimate is 0.48 of the
   annotated −0.071 on the same cells, just under half. The rule's
   falsification clause fires, by a margin of 0.003 nats on the interval.
2. **The predicted edges carry a measurable fraction of the annotated
   signal.** On the effect that is robust in pilot two, articulation
   RMSE, the predicted edges give −0.0135 [−0.0317, +0.0030], 0.4 of the
   annotated −0.0345, and the per-piece effects of the two edge sources
   correlate at +0.86. A model's occurrences on a score without
   annotations give a smaller version of the same thing.
3. **Precision matters more than recall where it fails.** Two pieces with
   precision below 0.2 (30-1, 31-1) show the family as the evidence sees
   it, large weight on 31-1 with no gain on the annotated notes; the
   remaining 19 pieces have F1 0.60 to 0.98, and within them the effect
   does not order by F1.
4. **The per-channel block with predicted edges does nothing** and gives
   a small timing error loss (+0.0014 s*). With the noisier edge set the
   evidence does not find a channel to keep the family on.
5. **Scope.** Development side only, 21 pieces, four seeds, the same
   excerpts and masks as the pilots, the annotation's own pieces (the
   model was trained on the other folds of the same corpus, so these are
   its held-out predictions, not predictions on an unseen corpus).
   Nothing registered was touched and no thesis claim follows.

## What this changes

Nothing in the registration (`docs/motif_prereg_design.md` tests the
annotated edges). The thesis chapter states stage two as measured: the
family can be built from a model's occurrences and keeps about half of
its effect, with the interval covering zero at this size. The third
stage named in the proposal, inference on scores outside the annotated
corpus, is Future Work.
