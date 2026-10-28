# Resynthesis demo (dev track, exploratory, no claim)

Generated 2026-10-28 11:46 by `scripts/demo_resynthesis.py` in 298 s (OMP_NUM_THREADS=2).

Passage: URMP dev track (1, 1), 3 chains, 13 notes, 8.2 s, the same passage and the same deployment fits as `docs/thesis/figures/endgoal_viz.png` (score-nominal tempo-line starts, no annotation in the fit, chains fitted independently, harmonics only).

Clips in `results/resynthesis_demo/` (gitignored, 16 kHz, peak-normalized to the recording):

- `recording.wav`: the track over the passage span.
- `resynthesis.wav`: Phi(z_hat) a_hat, the posterior-mean pitch curves and inferred boundaries under the amplitude posterior mean at the MAP (eq:amp-post). Unfitted notes are silent.
- `flat.wav`: the same amplitudes under score-nominal pitch (0 cents, no curve) and tempo-line boundaries: the expression removed, the timbre strand kept.
- `residual.wav`: recording minus resynthesis: timbre and noise, by construction.

Diagnostic (NOT an evaluation of the expression estimate; Chapter 8 evaluates that against the reference curves directly): fraction of the recording's energy the harmonic render explains, 1 - ||x - Phi a||^2 / ||x||^2.

- Whole passage: 0.796  (fitted span only: 0.888).

| chain | note | MIDI | inferred c (cents) | explained energy |
|---|---|---|---|---|
| 0 | 0 | 61 | +0.6 | 0.847 |
| 0 | 1 | 64 | +22.7 | 0.845 |
| 0 | 2 | 66 | +32.1 | 0.884 |
| 0 | 3 | 66 | +21.0 | 0.874 |
| 0 | 4 | 69 | +20.5 | 0.893 |
| 1 | 7 | 69 | -5.0 | 0.892 |
| 1 | 8 | 71 | +16.1 | 0.957 |
| 1 | 9 | 69 | +5.2 | 0.849 |
| 1 | 10 | 68 | -0.9 | 0.909 |
| 1 | 11 | 66 | +21.3 | 0.899 |
| 2 | 12 | 68 | +15.1 | 0.910 |
| 2 | 13 | 66 | +15.9 | 0.879 |
| 2 | 14 | 64 | +22.4 | 0.865 |

Honesty: the signal model is harmonics with piecewise-constant amplitudes over two chunks per note and sigmoid windows at the joints; no noise band, no transient model. The remainder is what the decomposition (score + expression + timbre + noise) assigns to timbre and noise, and what Future Work (`sec:future-phase3`) proposes to bring under the prior.

Figure: `docs/thesis/figures/resynthesis_demo.png`.
