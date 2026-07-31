# Per-piece component dominance — dev analysis (2026-07-31)

Question: are there pieces where some components dominate the posterior mean
more than others? (Follow-up to the aggregate covariance-share table in
`results/posterior_components_dev.md`, whose large ± already hinted at it.)

Covariance shares of the posterior mean at held-out notes, per piece
(30 dev pieces, anchor seed 0, config `b_featlm`; g = graph, f = score
features, e = LM embeddings; shares sum to 1 per channel and can exceed
[0,1] individually because cross-terms land where they belong).

## Dominant component per (piece, channel)

| channel | graph | score features | LM features |
|---|---|---|---|
| τ | 3 | 21 | 6 |
| log r | 4 | 20 | 6 |
| v | 6 | 14 | 10 |

Every component is the dominant one for some pieces. The aggregate story
(features carry the mean) is the majority pattern, not a law.

## Extreme pieces

- **Piece 14 (Liszt)**: the mean is almost pure graph — g share 1.00 on τ,
  0.76 on v; the LM kernel is switched off on every channel. A piece whose
  expressive structure propagates from neighbours and matches no cross-note
  feature pattern.
- **Piece 28 (Bach)**: the opposite — LM shares 0.93 (τ) and 0.89 (log r),
  features near zero. Notably this is the documented NLL tail piece
  (`results/robust_tail_piece28.pkl`): the piece whose timing mean leans
  almost entirely on the embeddings is also the one whose timing NLL
  occasionally blows up. Association, not established cause.
- **Piece 5 (Debussy)**: LM share 0.69 on velocity, the largest embedding
  dominance on dynamics.
- **Piece 24/25 (Chopin)**: essentially feature-only pieces (f ≈ 1.0 on τ and
  log r).
- Two Chopin pieces (10, 11) have LM-dominated timing (e = 0.61, 0.72) even
  though the median piece has e ≈ 0 on τ — the bimodal ARD switch of
  `results/lm_channel_scales_dev.md` seen from the piece side.

## Reading

Per-piece evidence fitting does not just reweight components mildly — it
reassigns dominance per piece and per channel, from pure-graph (Liszt 14) to
pure-embedding (Bach 28) extremes. This is the strongest argument yet that
the per-piece Bayesian weighting IS the mechanism (the attribution finding),
and it suggests per-piece share profiles as a compact interpretable
descriptor of a performance's structure. Composer-level patterns (Chopin
LM-timing, Debussy LM-dynamics) are suggestive but n-per-composer is too
small for claims.

Full 30-piece table: run log (per-piece lines); shares array at
`~/.claude/jobs/b215b88d/tmp/piece_shares.npy`. Reconstruct: fit_components +
cov_shares from `scripts/make_posterior_components.py`, seed-0 anchor masks.
