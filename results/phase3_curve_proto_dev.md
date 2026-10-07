# Phase-3 prototype: the within-note curve from the waveform (dev)

2026-10-07, built at Ray's direction for the zemi. **Development,
prototype, no claims.** Code: `scripts/proto_phase3_curve.py`; figure:
`docs/thesis/figures/phase3_curve_proto.png`.

The two verified halves glued for the first time — the within-note GP
prior (two-component kernel on a knot grid, fixed prototype
hyperparameters) and the amplitude-collapsed waveform likelihood
(chunked harmonic design, the dev-study machinery). MAP over (c, knots)
by coarse c grid + Nelder-Mead; per-knot diagonal Laplace. **No pitch
tracker and no estimator anywhere.** This is the phase3 module's
documented stub (inference over the nonlinear z beyond scalar grids)
and the feasibility probe for curve-level channels (the design
question's drawing-board depth).

## Result (three instrument-diverse dev notes, GT frames never seen)

| note | waveform curve vs GT | pYIN frames vs GT |
|---|---|---|
| violin (1,1)#31 | **2.5 cents** | 3.4 cents |
| cello (1,2)#0 | 4.8 cents | **2.7 cents** |
| clarinet (3,2)#96 | **1.0 cents** | 4.6 cents |

A real finding on the way: with 12 knots a 1.5 s note is BELOW Nyquist
for 5-6 Hz vibrato (violin 4.6, cello 9.0 cents — the curve could not
represent the oscillation). At 16 knots/s the vibrato appears and the
numbers above result. The knot rate is a resolution parameter, not a
tuning knob.

## Caveats (prototype honesty)

Three notes; fixed prior hyperparameters (not fit); diagonal Laplace
only; Nelder-Mead in up to 33 dims (the cello's remaining gap may be
optimization, low fundamental, or amplitude overshoot — not diagnosed);
knot interpolation is linear. Next steps if pursued: evidence-fit
hyperparameters, joint Laplace, batch evaluation on the dev tracks
against the quasi-truth protocol of `results/phase3_waveform_dev.md`.
