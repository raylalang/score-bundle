# Music-theory features — dev A/B results (2026-07-14)

> **Status: development-set study; honest negative.** Supervisor comment
> (2026-07-13): the 25 hand-built score features carry no tonality — add real
> music theory (scales, meter, harmony, phrase, repetition, voice). Done
> (`baselines.rich_score_features(theory=True)`, 14 columns, score-only,
> default off, tests pin musical sanity). Measured verdict: **the theory
> features add no significant value on this task, and exactly nothing next to
> the music-model embeddings.** Confirmation set untouched.

## The 14 columns

Local key by Krumhansl–Schmuckler over a ±16-note window (key clarity, mode,
scale-degree sin/cos, in-scale flag, circle-of-fifths motion of the local key),
metrical-weight hierarchy (5 levels), vertical dissonance + bass flag,
LBDM-style phrase-boundary salience (IOI and pitch, per voice),
motif-repetition counts (per-voice interval+rhythm 4-grams — adopted from the
motif-discovery work), within-voice tessitura and position.

## Protocol

`b_theoryfeat[lm]` (25 + 14 columns) vs `b_feat[lm]` (25 columns), identical
otherwise: published 40%-hidden anchor masks, 30 dev pieces × 4 seeds,
guard-on (`results/graphgp_theoryfeat/` vs
`results/graphgp_masksweep/obs0.60_anchor/`), paired per-piece bootstrap CIs.

## Results (pooled; paired ΔRMSE/ΔNLL vs the 25-column baseline)

| row | RMSE | NLL | cov@.9 | ΔRMSE [95% CI] | ΔNLL [95% CI] |
|---|---|---|---|---|---|
| features only | 0.3683 | −0.370 | 0.923 | — | — |
| + theory | 0.3656 | −0.371 | 0.922 | −0.0023 [−0.0049, +0.0001] | −0.0011 [−0.0079, +0.0050] |
| features + LM emb | 0.3601 | −0.404 | 0.927 | — | — |
| + theory | 0.3602 | −0.401 | 0.926 | +0.0007 [−0.0022, +0.0034] | +0.0031 [−0.0063, +0.0141] |

Per-channel: no channel significant in either regime; the largest (still
non-significant) trend is articulation without the LM
(log r: −0.0038 [−0.0089, +0.0005]), plausible for phrase/meter features but
not established.

## Reading

The pattern replicates the harmonic-edge finding at the feature level: an
explicit music-theory signal shows a small positive trend on the plain model
and is **measured-redundant once the LM embeddings enter the kernel** — the
pretrained music model already carries the tonal/metrical/repetition
information the hand-built columns encode. This strengthens the thesis
attribution story (what the embeddings contribute *is*, in part, this
structure) rather than weakening the feature set: the columns stay in the
codebase as an interpretable probe, default off.

## Reproduce

```bash
# baseline cells: bash scripts/run_mask_sweep.sh (anchor block)
for K in $(seq 0 11); do OMP_NUM_THREADS=2 PYTHONPATH=src \
  python scripts/eval_graphgp.py --stage run --guard \
  --configs b_theoryfeat,b_theoryfeatlm \
  --out-dir results/graphgp_theoryfeat --shard $K/12 & done; wait
```
