# SM-GP estimator vs the sine fit: the kill-cheap test (development, exploratory)

2026-09-04. The committed development study of `docs/gp_everywhere_memo.md`
/ `docs/sm_estimator_note.tex` §7, executed as ordered after the professor
meeting. **Development data only; exploratory; no claims. The registered
Phase-2 pipeline and its spent confirmation are untouched.**

- Code: `src/score_bundle/phase2/sm_estimator.py` at commit `611685a`
  (implementation frozen before the full run), harness
  `scripts/eval_sm_estimator.py` (4 shards, 16,925 note records,
  78 dev tracks).
- Data: the URMP development caches (`urmp_targets_dev.pkl`,
  `urmp_f0_dev.pkl`); frame rule = `targets.note_targets` verbatim.
- Population: 5,157 notes vibrato-identifiable in BOTH curves (M1/M2/M4);
  1,017 notes refused on the tracked curve but GT-identifiable (M3).
- Protocol: each estimator runs on the tracked curve and on the
  ground-truth curve of the same note and is scored against its own
  GT-curve output (own-estimand rule). Cluster bootstrap over tracks,
  B=2000, seed 0; `*` = 95% CI excludes zero.

## Verdict (by the pre-committed rule)

**The SM-GP does not beat the sine fit on parameter accuracy (M1), so the
estimator-replacement idea (Slot A) stops here.** Intonation ties; extent
and rate lose decisively. The curve-level measure (M4), which needs no
target definition, favors the GP on frame accuracy — a signal for Slot B
(the Phase-3 curve prior), not for Slot A.

## M1 — parameter accuracy, |tracked − own GT reference| (the kill criterion)

| quantity | median SM | median NLLS | paired Δmean (SM−NLLS) |
|---|---|---|---|
| intonation c (cents) | 2.085 | 1.914 | +0.165 [−0.175, +0.377] ns |
| extent log γ | 0.516 | 0.157 | +1.686 [+1.376, +1.984]* |
| rate log f | 0.198 | 0.045 | +0.164 [+0.141, +0.187]* |

Metric caveat, stated up front: this measure scores *reproducibility of
the estimator across two measurements of the same note*, and a rigid
functional is stable by construction. It was committed as the kill
criterion because the bundle consumes exactly these scalars, and an
estimator that cannot reproduce its own targets across tracker noise
feeds the graph GP noisy cells whatever its other virtues.

## M2 — calibration of the tracked estimate against its reference (combined variance)

| quantity | SM cov@90 / NLL | NLLS cov@90 / NLL |
|---|---|---|
| intonation c | 0.332 / 10400 | 0.619 / 3408 |
| extent log γ | 0.969 / 2.0 (n=1888 non-wide) | 0.846 / 5.9 |
| rate log f | 0.621 / 9.0 (n=1888 non-wide) | 0.633 / 16.2 |

Neither estimator is calibrated against its cross-curve reference on
intonation (the combined variances omit shared tracker structure). The SM
intonation variance is additionally overconfident by construction: the
realized-centre functional variance conditions on the fitted
hyperparameters (no hyperparameter uncertainty), a known limitation of
the frozen implementation. Extent: where the SM reports finite
uncertainty it is nearly nominal (0.969) where NLLS undercovers (0.846) —
but it reports wide/unusable curvature on 63% of notes (3,269/5,157),
which is the flip side of the same honesty.

## M3 — the 1,017 notes the sine fit refuses

The SM supplies posteriors where the rule refuses. Intonation: median
error 3.1 cents against a 15.4-cent reference spread — informative — but
undercovered (0.381). Extent/rate: covered at 0.917/0.672 on the 290
notes with finite curvature; wide on the rest. Capability demonstrated,
calibration not.

## M4 — curve level: fitted on tracked frames, scored on the GT curve's frames

| metric | SM median | NLLS median | paired Δmean |
|---|---|---|---|
| frame RMSE (cents) | 6.26 | 9.63 | −1.47 [−2.61, +0.004] (borderline) |
| frame NLL | 3.30 | 3.87 | mean poisoned by SM outlier cells (+1.7e9*) |
| frame cov@90 | 0.895 | 0.898 | −0.032* |

The GP describes the actual curve better than the sinusoid (median frame
RMSE 35% lower, median NLL better, coverage at nominal), with a
heavy-tail caveat: a small number of SM cells produce astronomical NLL
(near-zero predictive variance meeting a large offset — same
overconfidence mechanism as the M2 intonation row).

## The methodological finding (the durable part)

The same structural fact surfaced three times during implementation, each
requiring a redesigned read-out: **the evidence's process-level
parameters are not the channels' estimands.** The channels mean this
note's *realized* quantities. (1) Ensemble scale √(2w₁) is χ²₂-spread
around the realized amplitude (median 0.83×) → extent read from the
posterior vibrato component. (2) The process constant and the slow drift
component split a note-level offset arbitrarily → centre read as
c + realized drift mean (fixed the 9.6-cent median error down to 2.1).
(3) The unconstrained evidence prefers an incoherent vibrato band on real
curves, under which the rate is weakly identified → coherence bound
(bandwidth ≤ rate/8). Even after all three alignments, the decomposition
remains less reproducible across two measurements of the same note than
the rigid fit — that instability, not any single bug, is why Slot A dies.

## Decision

- **Slot A (estimator v2): dead** by the committed rule. The sine fit
  remains the bundle's estimator; nothing changes anywhere.
- **Slot B (Phase-3 within-note curve prior): strengthened.** The
  estimand-free curve-level measure favors the SM prior for describing
  the curve itself, which is exactly Slot B's job (the deviation prior in
  the collapsed waveform likelihood is already a crude GP). Any move
  there is a separate decision, discussed before built.
- Reproduce: `PYTHONPATH=src:scripts python scripts/eval_sm_estimator.py
  run --shard K/4` (×4) then `report`.
