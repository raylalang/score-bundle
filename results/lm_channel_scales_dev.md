# Why the LM component is flat on τ / log r — dev investigation (2026-07-31)

Question (from the decomposition snapshot): the LM-feature component of the
posterior mean is flat on timing and articulation but active on velocity. Why?

## 1. Proximate mechanism: the evidence switches the kernel off per channel

Per-piece fitted prior-variance share of the embedding kernel (config
`b_featlm`, 30 dev pieces, anchor seed 0):

| channel | piece 0 (the figure) | median | IQR | frac < 1% | frac > 10% |
|---|---|---|---|---|---|
| τ | 0.000 (c_emb = 4e−12) | 0.001 | [0.000, 0.172] | 0.60 | 0.27 |
| log r | 0.000 (c_emb = 2e−9) | 0.000 | [0.000, 0.118] | 0.60 | 0.30 |
| v | 0.072 (c_emb = 4e−6) | 0.103 | [0.065, 0.171] | 0.07 | 0.50 |

On the figure piece the embedding kernel is numerically zero on τ and log r —
hence the flat curves. Across the dev set the pattern is bimodal: ~60% of
pieces switch the LM off on τ/log r entirely, a ~quarter use it substantially;
on velocity it is nearly always on.

## 2. Behavioral reason: the embeddings have no within-piece signal there

Within-piece OOS ridge probe (fit on observed 60%, scored on hidden notes,
per-piece/channel regularization tuned on an inner split — the honest analogue
of the GP's marginalized linear kernel). Median OOS R² over 30 dev pieces:

| predictor | τ | log r | v |
|---|---|---|---|
| embeddings alone | −0.01 | −0.44 | +0.42 |
| score features alone | +0.03 | +0.25 | +0.41 |
| embeddings residualized on score features | −0.01 | −0.02 | +0.22 |

On velocity 97% of pieces show positive embedding signal and 87% still do
after removing the score-feature content — genuine unique information. On
timing the embeddings carry essentially nothing within-piece; on articulation
a naive linear read-out is actively harmful (the evidence's shrink-to-zero is
the correct response).

## Reading

The flat components are per-piece ARD working as intended: the evidence
learns, piece by piece, that the embeddings' exploitable within-piece linear
signal lives in velocity (dynamics conventions), and shrinks `c_emb` to zero
elsewhere. Consistent with the cross-piece record (the LM mean's per-channel
edge is loudness) and the probe study (embeddings encode voicing/rhythm/
register, i.e. what velocity conventions follow). One open nuance: the dev
ladder shows adding embeddings also improves log r RMSE slightly
(0.615 → 0.601) — likely carried by the minority of pieces with a nonzero
log r scale and/or by cross-channel coupling through B; not yet isolated.

Logs: `~/.claude/jobs/b215b88d/tmp/lm_scales.log`, `lm_probe2.log` (inline
scripts; both reconstructible from this doc's description).
