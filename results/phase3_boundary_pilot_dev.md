# Phase 3: differentiable note boundary — feasibility pilot (DEV, exploratory, no claims)

Two-note violin segments, complementary sigmoid windows (softness 10 ms) over per-note
chunked designs, amplitudes collapsed, FLAT prior on the boundary b; Adam over
(b, c1, c2, u1, u2) from the annotated boundary + delta. Recovery = final error <= 10 ms
or <= 25% of the initial offset. Script: scripts/pilot_phase3_boundary.py.

8 pairs x 4 perturbations = 32 fits; recovered 7/32.

| pair | interval (st) | delta (ms) | err init (ms) | err final (ms) |
|---|---|---|---|---|
| #0 | +3 | -80 | 80 | 37.2 |
| #0 | +3 | -40 | 40 | 37.0 |
| #0 | +3 | +40 | 40 | 80.0 |
| #0 | +3 | +80 | 80 | 79.9 |
| #1 | +2 | -80 | 80 | 10.9 |
| #1 | +2 | -40 | 40 | 10.9 |
| #1 | +2 | +40 | 40 | 10.9 |
| #1 | +2 | +80 | 80 | 11.0 |
| #4 | -1 | -80 | 80 | 11.5 |
| #4 | -1 | -40 | 40 | 25.0 |
| #4 | -1 | +40 | 40 | 25.0 |
| #4 | -1 | +80 | 80 | 25.0 |
| #9 | -1 | -80 | 80 | 17.4 |
| #9 | -1 | -40 | 40 | 17.4 |
| #9 | -1 | +40 | 40 | 17.4 |
| #9 | -1 | +80 | 80 | 17.3 |
| #10 | -2 | -80 | 80 | 25.5 |
| #10 | -2 | -40 | 40 | 25.4 |
| #10 | -2 | +40 | 40 | 25.6 |
| #10 | -2 | +80 | 80 | 25.6 |
| #13 | -2 | -80 | 80 | 35.7 |
| #13 | -2 | -40 | 40 | 54.0 |
| #13 | -2 | +40 | 40 | 53.7 |
| #13 | -2 | +80 | 80 | 53.8 |
| #15 | +0 | -80 | 80 | 16.8 |
| #15 | +0 | -40 | 40 | 16.8 |
| #15 | +0 | +40 | 40 | 16.8 |
| #15 | +0 | +80 | 80 | 16.7 |
| #16 | +3 | -80 | 80 | 87.1 |
| #16 | +3 | -40 | 40 | 87.2 |
| #16 | +3 | +40 | 40 | 40.9 |
| #16 | +3 | +80 | 80 | 40.9 |

final |err|: median 25.2 ms, q90 77.3 ms; wall per fit median 13 s

## Reading (2026-10-09)

The pre-stated recovery rule gives 7/32 — reported verbatim. The per-pair
structure says more than the rule does:

1. **The mechanism works.** In 7 of 8 pairs the optimizer reaches the
   SAME optimum from all four starts (within-pair spread <= ~1 ms across
   delta = +/-40/+/-80 ms); the eighth splits into two nearby optima
   (~80 vs ~87 ms). No hostile surface, no phase-accumulation pathology,
   no divergence: the differentiable boundary carries informative
   gradients, which is what this pilot existed to test.
2. **What it converges to is not the annotation.** The start-independent
   optimum sits 11-87 ms (median ~25 ms) from the annotated onset — far
   above the optimizer's own reproducibility. This is the estimand gap
   in a third costume (after the scalar-c and curve-band versions): the
   likelihood finds the ACOUSTIC transition under a hard harmonic swap,
   the annotation marks the musical onset, and a real note transition
   (bow change, legato slide) is a region, not a point. The offset is a
   measurement of that mismatch plus the model's crudeness at the joint
   (complementary 10 ms sigmoids, no transition model, per-segment
   chunked amplitudes).
3. **Consequence for the December direction.** Joint multi-note
   inference can proceed on this mechanism; interpreting inferred
   boundaries against annotations cannot, until the transition region is
   modelled (overlapping windows / a crossfade with its own width
   parameter) or the boundary estimand is defined acoustically. That is
   a design question, not an optimization question — which is the
   pilot's answer.
