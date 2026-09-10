# Slot B measured: the SM kernel as the Phase-3 deviation prior (development)

2026-09-10. Study D of the exploration week, completing the
spectral-kernel question: after Slot A (estimator replacement) died by
its own kill test, the curve-level evidence pointed at Slot B — the
within-note deviation prior of the collapsed waveform likelihood. This
study swaps the published eight-Hann-bump basis for the exact low-rank
representation of a two-component SM GP prior (vibrato band at the
scaffold rate with the coherence bound, drift band; top-8 eigenvectors,
matched pointwise scale and rank; everything else identical to the
published `infer_c_devprior`). Paired note for note on the published
376-note rig, scored against the quasi-truth centre. **Development,
exploratory, no claims.**

## Verdict: no gain — the pragmatic basis was already enough

| variant | median abs err (cents) | q90 | cov@90 |
|---|---|---|---|
| Hann bumps (published) | 2.29 | 6.77 | 0.03 |
| SM prior | 2.25 | 7.28 | 0.02 |

Paired |err|, SM − bumps: **+0.112 [+0.026, +0.202]\*** (median +0.000).
Per family: strings tie (+0.002, median −0.058), brass ≈ tie (+0.081),
**winds carry the harm (+0.226)** — the family where the tracker's
failure modes live, hence where the scaffold rate is least reliable, and
a vibrato-band prior centred at a wrong rate is worse than frequency-
neutral bumps. The bump baseline reproduces the published 2.29 median
exactly (pairing sanity). Coverage stays at the estimand-gap floor
(~0.02–0.03) under both priors, as the published study predicted for any
within-model refinement.

## Reading, and the closed loop

Both slots the kernel review named are now measured: **Slot A dead**
(the sine fit reproduces the parameter targets better), **Slot B
no-gain** (at matched rank, the principled spectral basis does not beat
the pragmatic bumps, and mis-specified band location hurts where the
scaffold is weak). The reviewed kernels' measured contribution to this
pipeline is currently conceptual — the curve-level scoring insight, the
coherence bound, the realized-vs-process estimand lessons — not
numerical. This is consistent with the thesis's own caution that the
binding Phase-3 constraints are the estimand bridge and the noise rows,
which no prior family touches. The GSM rung stays parked; any revival of
Slot B should make the band location robust (e.g., rate marginalized,
not scaffold-pinned) before spending further compute.

- Reproduce: `PYTHONPATH=src:scripts python scripts/eval_phase3_smprior.py
  run K/6` (×6) then `report`.
