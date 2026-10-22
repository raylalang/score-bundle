# Piano: the headline model with the pitch term removed from its graph (DEV, exploratory, no claims)

2026-10-23. `scripts/eval_graphgp.py --configs b_featlm_timegauss`,
development set (30 pieces × 4 seeds = 120 cells, 40% of notes hidden),
paired per piece against the existing `b_featlm` cells in
`results/graphgp/` (the headline GP-first configuration: score features
+ music-model embeddings + the score graph). `b_featlm_timegauss` is the
same model with `build_adjacency(score, ell_p=1e6)`: the pitch term
removed, the 2-beat Gaussian time window kept (on polyphonic piano that
window links chord notes, Δb = 0, with full weight). Hypothesis H-P1 and
the falsification rule are in the script docstring, committed before
the run. Report: `logs/graphgp_timegauss.report.log`.

| row | RMSE (pooled) | NLL | cov@0.90 | dRMSE vs b_featlm [95% CI] | dNLL [95% CI] |
|---|---|---|---|---|---|
| b_featlm (headline) | 0.3590 | −0.4051 | 0.927 | — | — |
| b_featlm_nograph | 0.3755 | −0.3359 | 0.925 | +0.0174 [+0.0076, +0.0281]* | +0.0690 [+0.0546, +0.0846]* |
| b_featlm_timegauss (pitch term removed) | 0.3504 | −0.4237 | 0.923 | −0.0082 [−0.0163, −0.0010]* | −0.0186 [−0.0298, −0.0082]* |

Per channel (RMSE / cov@0.90): timing 0.1518/0.948 → 0.1513/0.946;
articulation 0.5988/0.913 → 0.5835/0.910; velocity 0.0718/0.921 →
0.0712/0.912.

## Reading (2026-10-23)

1. **H-P1 is falsified, in the stronger direction.** Removing the
   pitch term from the graph does not cost accuracy or calibration on
   piano; it improves both, significantly though modestly: pooled RMSE
   −0.008 [−0.016, −0.001] and NLL −0.019 [−0.030, −0.008], paired per
   piece. The gain sits in articulation (0.599 → 0.584) with timing and
   velocity flat. The graph's total value on this set is +0.017 RMSE
   (the no-graph row), so the pitch-less graph is worth about half
   again as much as the score graph.
2. **The same answer as strings and winds.** On the tracked bundle the
   score graph equalled the pitch-less graph on intonation and was
   slightly worse on vibrato (`results/phase2_timegauss_prior_dev.md`);
   on piano it is slightly worse on articulation. On both corpora the
   structure that does the work is smoothness in score time over a
   window of about two beats, which on piano includes the vertical
   coupling of chord notes at the same beat. Linking notes by pitch
   proximity on top of that adds nothing and costs a little.
3. **What it does and does not change.** The confirmed piano headline
   (0.376 vs 0.393 against the two-stage baseline, coverage 0.925) and
   the graph's calibration contribution stand: they compare the full
   model with alternatives, and the graph term remains the carrier.
   What changes is the attribution of the graph's value: temporal
   smoothing, not the score's pitch structure. The headline model was
   selected before this was known; a pitch-less graph as the default
   would be a change to a reported protocol and needs its own
   registration if adopted.
4. Scope: development set, selection-reused (every model's NLL
   flatters there); 120 cells; the confirmation pool untouched.
