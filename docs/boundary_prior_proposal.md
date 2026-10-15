# Proposal: the boundary-prior extension of the prior-value comparison (NOT RUN)

2026-10-15. Written after the first span (`results/prior_value_dev.md`)
and alongside its tuning-scalar follow-up. A proposal in the format the
2026-10-14 brief demands: hypothesis, why simpler alternatives do not
answer it, the measurable outcome, the falsification rule. Nothing here
has been run; running it needs Ray's go and ~1.5 h wall on 8 CPU shards.

## What exists

`phase3/curve.py` `fit_chain` optimizes, per chain, the per-note
constants `c`, the per-note curve knots `u`, and the per-joint boundary
offsets `db` (ms from the initial boundary). The constants `c` can carry
a proper cross-note prior (`c_prior_cov`, the first span). The offsets
`db` carry **no prior at all** (flat): the only regularizers are the
lognormal prior on the fade width and the ordering hinge. The chain
study (`results/phase3_chain_dev.md`) therefore measured the
*likelihood's* boundary information alone: ~18 ms median edge error,
deployment from score-nominal starts 39.4 → 19.5 ms, and a tail.

## Hypothesis

H-B1 (ambiguity): where the waveform carries little boundary
information, a cross-joint prior over the offsets `db` recovers a
boundary better than the flat prior, which can only return the
likelihood's local optimum (or the initial value when the likelihood is
blind).

H-B2 (control): where the waveform carries strong boundary information,
the prior changes nothing.

Score-specific edge (secondary, underpowered at chain scale, see below):
a score-graph prior over joints beats a plain temporal chain only where
the score adds information beyond adjacency, i.e. joints at the same
metric position sharing a timing deviation.

## Design (smallest decisive version)

Three priors over the `db` vector of a chain, matched marginal variance,
identical likelihood, data, optimizer, steps, fade width fixed at the
chain study's winning sharp setting:

- independent: `db ~ N(0, σ_b² I)`;
- temporal: `db ~ N(0, σ_b² K_chain)` over the joints in order;
- score graph: `db ~ N(0, σ_b² K_G)` with the score graph restricted to
  the joint notes (time and pitch proximity, as in the first span).

Plus, in every arm, a **free per-chain offset scalar** `s ~ N(0, σ_s²)`,
`db = s·1 + db~`, marginalized as a rank-one term, for the same reason
the tuning scalar exists in the intonation follow-up: the likelihood
finds the acoustic transition and the annotation marks the musical
onset, a systematic ~18 ms offset shared along a chain. Without it the
prior would be miscentred in every arm, as the zero-mean `c` prior was.

`σ_b²` = the development tracks' own variance of annotated-onset
deviation from the tempo line (the τ channel's population scale, as
`var_c` was for intonation), never tuned.

Ambiguity is created by **masking the audio around one joint** (the
joint's fade region ± a guard, the masked-imputation design one level
down): the hidden boundary is then inferable only through the prior
from the observed joints. Clean = full audio. Two further strata exist
for free and are pre-stated: (i) unison joints (the chain study's
"unidentifiable to a harmonic likelihood" case, which turned out better
than expected via amplitude dips) and (ii) deployment inits from the
score-nominal tempo line instead of the annotation.

## Measurable outcome

Hidden-joint error |b̂ − b_annot| in ms, paired across arms per joint;
coverage at 90% from the joint Laplace over (c, u, db); wall per fit.
Deadpan reference: the tempo-line boundary error (39.4 ms median in the
chain study).

## Falsification rule

H-B1 fails if the structured arms are not better than the independent
arm on masked joints (paired bootstrap CI covers zero). A rigorous
negative here is informative: it would say that boundary timing on
short chains is a local acoustic fact, not a structured one, and the
December joint loop should couple notes through pitch and dynamics, not
timing. H-B2 fails if the clean-audio arms differ.

## Why simpler alternatives do not answer it

- The chain study already measured the likelihood alone (flat prior);
  it cannot say whether structure helps because there is none in it.
- Running the three priors at the measurement level through the
  strings-and-winds bundle (the no-new-code preliminary) uses τ from
  the warp, i.e. tracker-chain timing, not waveform boundaries; it
  answers a different question (and the registered C4 timing claim
  already failed there).
- Adding the prior only to the deployment arm without masking would
  confound prior value with basin-of-attraction effects.

## Honest power caveat

Chains have 2–4 joints, so with one masked the prior sees 1–3 observed
joints. The temporal and score-graph arms nearly coincide on such
chains (the score-specific edge needs joints at shared metric positions
across a phrase). The decisive comparison at this scale is therefore
**structure vs none**; the score-vs-temporal contrast is reported but
expected ns, and the proper test of score-specific timing structure is
the whole-phrase joint loop (ROADMAP December row).

## Code needed (not written)

- `fit_chain(b_prior_cov=)`: one more Gaussian prior term over `db`
  (mirrors `c_prior_cov`, Cholesky, ~10 lines).
- A joint-level sample mask in the cell iterator (mirrors the note mask
  with the fade region's guard).
- The Laplace over (c, u, db) at the MAP (extend `chain_c_laplace` by
  the `db` block).
- A sharded script `eval_boundary_prior_dev.py` with the hypotheses
  above in its docstring; cells under their own prefix; results file
  `results/boundary_prior_dev.md`.

## Cost

Chain fits ran ~138 s median in the chain study; 47 chains × 3 arms ×
2 conditions ≈ 282 fits ≈ 11 h CPU ≈ 1.4 h wall on 8 shards at
OMP_NUM_THREADS=2, plus the Laplace. Estimate from one smoke cell first
(long-run rule).
