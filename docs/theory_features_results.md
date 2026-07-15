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

## Follow-up: linear probes of the embeddings (2026-07-15)

The Reading above originally attributed the negative to *subsumption* — "the
embeddings already carry the tonal/metrical/repetition signal." A direct probe
study (`scripts/probe_embeddings.py`; ridge probes fit on the head pieces,
scored out-of-sample on the 30 dev pieces; results in
`results/probe_embeddings*.pkl`) shows that is **only half true**:

| theory column | R² emb | R² feat (control) | AUC emb | verdict |
|---|---|---|---|---|
| in-scale flag | −0.06 | −0.00 | 0.52 | **not encoded** |
| scale degree (sin/cos) | −0.05 | −0.00 | — | **not encoded** |
| mode (major/minor) | −0.17 | −0.04 | 0.48 | **not encoded** (also at piece level: 0.48) |
| repetition count | −0.05 | −0.00 | — | not encoded (0.12 raw) |
| metrical weight | 0.14 | 0.43 | — | weakly encoded |
| phrase salience (IOI) | 0.48 | 0.57 | — | encoded |
| bass flag | 0.73 | 0.86 | 0.99 | encoded |
| register (voice pitch-z) | 0.90 | 1.00 | — | encoded |

(The `--raw` variant, which keeps piece-level information the GP's
per-piece-standardized kernel cannot use, does not change the tonal verdicts.)

**Corrected reading (now in the thesis):** the embeddings encode voicing and
rhythm — bass membership, meter, phrase, register — but **not tonality**. The
theory columns' failure is therefore *not* tonal subsumption; combined with the
tonal-metric kernel negative, the consistent conclusion is that **tonal
structure carries no measurable marginal signal for these three expressive
channels at this scale**, while the rhythmic/voicing structure the columns add
genuinely is already in the kernel. The embeddings' value is rhythm, texture,
and dynamics convention — not harmony.

**Nonlinearity check (2026-07-16):** a 2048-dimensional random-Fourier-features
probe (`--rff-dim 2048`, `results/probe_embeddings_rff.pkl`) replicates every
tonal verdict — in-scale AUC 0.55, mode 0.47, scale degree R² ≈ 0 — so "not
encoded" is not an artifact of probe linearity. The rhythmic/voicing verdicts
also replicate (bass AUC 0.98, meter 0.23, phrase 0.44).
