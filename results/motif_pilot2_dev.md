# Motif study, stage one, second pilot: the per-channel weight (DEVELOPMENT ONLY, exploratory, no claim)

2026-11-17, 12:44 to 14:13 JST on a loaded machine (14 shards,
`OMP_NUM_THREADS=2`, 672 fits, about 9 fits per minute).
`scripts/eval_motif_pilot_dev.py` (hypothesis H2 and its falsification
rule in the docstring, committed at c714bf6 before the run), runner
`scripts/run_motif_pilot2.sh`; cells in `results/motif_pilot2/`
(gitignored pickles; `align.md`, `report.md` and `effects.json` kept);
logs `logs/motif_pilot2.shard*.log`, `logs/motif_pilot2.report.log`;
figure `docs/thesis/figures/proposal_pilot2_results.png`
(`scripts/make_proposal_figs.py --pilot2`); proposal deck frame 11.
Power tables: `scripts/power_motif.py results/motif_pilot2 --arm ARM
--scheme occ --channel CH --metric M`.

## What was run

The same 21 movements, excerpts, alignment, masks and scoring as pilot
one (`results/motif_pilot_dev.md`), with four seeds instead of two
(seeds 0 and 1 are the pilot-one masks, bit-identical) and a new form of
the motif family. In pilot one the family entered the adjacency,
W = W_time + w_m W_motif, with one weight shared by the three channels
through the coregionalization matrix B. In the per-channel form the
family is its own additive term with one scale per channel:

    K_m = U_m diag(g(ν_m; s_m)) U_mᵀ,  g = 1/(1 + s_m ν),  s_m = 1 fixed,

from the Laplacian of W_motif restricted to the motif notes (zero on every
other note), passed as one more feature block X_m = U_m diag(√g) so that
`MultiOutputGraphGP` learns c_m[channel] by the per-piece evidence through
its existing feature path (no change to `gp.py`; the factorization
X_m X_mᵀ = K_m is unit-pinned in `tests/test_motif_pilot.py`). These arms
have no learned graph parameter (n_graph = 0), so one fit costs about 75 s
against about 100 s for the shared weight.

- Arms, paired on identical masks: `time` (time-only Gaussian graph),
  `time_motif` (pilot one's shared weight), `time_motif_pc` (per-channel
  block), `time_pitch_motif_pc` (the thesis's score graph plus the block).
- Schemes: `occ` (whole occurrences hidden) and `random` (40 percent), as
  in pilot one. Four seeds. 21 × 2 × 4 × 4 = 672 fits.
- Scored on hidden motif notes whose type has another occurrence with an
  observed note; Gaussian log score and RMSE per channel, paired per
  piece with seeds averaged, percentile bootstrap B = 2000 over 21 pieces.

H2 (primary, fixed before the run): on hidden motif notes under the
occurrence scheme, `time_motif_pc` keeps the timing log-score gain of
pilot one against `time` (interval below zero) AND removes the velocity
penalty (velocity ΔNLL interval covering or below zero, coverage at or
above 0.87). Falsification: the timing gain lost means the family's
timing value came from the shared coupling, not from the edges; the
velocity penalty kept means the penalty is not the shared weight.

## Primary rows (occurrences hidden, hidden motif notes, vs the time-only graph, four seeds)

| arm | timing ΔNLL [95%] | timing ΔRMSE (s) | articulation ΔNLL | articulation ΔRMSE | velocity ΔNLL | velocity ΔRMSE | velocity coverage |
|---|---|---|---|---|---|---|---|
| time + motif (shared weight) | −0.071 [−0.230, +0.097] | −0.0027 [−0.0055, −0.0004]* | −0.047 [−0.117, +0.025] | **−0.0345 [−0.0600, −0.0125]*** | +0.033 [−0.012, +0.078] | +0.0001 [−0.0021, +0.0022] | 0.835 |
| time + motif (per channel) | −0.005 [−0.076, +0.054] | +0.0010 [−0.0041, +0.0058] | −0.044 [−0.113, +0.004] | **−0.0189 [−0.0381, −0.0053]*** | **−0.086 [−0.200, −0.018]*** | **−0.0057 [−0.0106, −0.0023]*** | 0.944 |
| time + pitch + motif (per channel) | −0.005 [−0.092, +0.086] | +0.0018 [−0.0031, +0.0083] | −0.046 [−0.117, +0.024] | −0.0194 [−0.0512, +0.0145] | −0.008 [−0.096, +0.069] | +0.0014 [−0.0022, +0.0056] | 0.923 |

Levels of the time-only arm on the same notes (four seeds): timing NLL
−1.240, RMSE 0.0698 s, coverage 0.970; articulation 0.998 / 0.732 /
0.891; velocity −1.015 / 0.0863 / 0.878. Coverage of the per-channel arm:
timing 0.975, articulation 0.921, velocity 0.944.

Under random masking (hidden motif notes): the shared weight gives timing
ΔNLL +0.035 [−0.179, +0.352] and the RMSE gains of pilot one (timing
−0.0022*, articulation −0.0144*); the per-channel block gives articulation
ΔRMSE −0.0100*, velocity ΔNLL −0.033* and ΔRMSE −0.0035* with coverage
0.948, and timing −0.031 ns. On all hidden notes under random masking the
per-channel block's timing log score is +0.087 [−0.000, +0.241], at the
edge of worse, and the shared weight's velocity log score is +0.017*
worse. Full tables in `results/motif_pilot2/report.md`.

## Seed stability (the shared weight, occurrences hidden)

| seeds | timing ΔNLL | articulation ΔRMSE | velocity ΔNLL |
|---|---|---|---|
| 0, 1 (pilot one, reproduced bit-exactly) | −0.117 [−0.245, −0.006]* | −0.033 [−0.067, −0.008]* | +0.069 [+0.005, +0.138]* |
| 2, 3 | −0.026 [−0.247, +0.259] | −0.032 [−0.055, −0.011]* | −0.004 [−0.044, +0.035] |
| 0 to 3 | −0.071 [−0.230, +0.103] | −0.033 [−0.057, −0.012]* | +0.033 [−0.012, +0.077] |

(These rows are the power script's per-piece differences, which agree with
the report's rows to within 0.002.) The pilot-one timing star does not
survive two more seeds. Per piece the four-seed timing difference has
median −0.004 with 15 of 21 pieces below zero; the mean is carried by
15-1 (−0.98), 22-1 (−0.90) and 17-1 (−0.55) against 4-1 (+1.11). The
articulation difference is the same on every seed pair, 15 of 21 pieces
below zero, median −0.012, the largest on 4-1 (−0.19), 17-1 (−0.13) and
11-1 (−0.09).

## Where the evidence puts the family (per-channel scale c_m, occurrences hidden, geometric mean over seeds)

Timing: c_m below 0.01 on 19 of 21 pieces (10-1 at 0.06 and 17-1 at 0.04
are the exceptions), so the evidence switches the family off on the
timing channel almost everywhere. Articulation: c_m above 0.1 on ten
pieces (31-1 1.28, 1-1 1.25, 17-1 0.69, 9-1 0.62, 27-1 0.42, 18-1 0.35,
21-1 0.34, 8-1 0.30, 10-1 0.24, 5-1 0.11), below 0.001 on seven.
Velocity: above 0.01 on seven pieces (8-1 0.11, 22-1 0.06, 31-1 0.04,
17-1 0.03, 28-1 0.02, 12-1 0.02, 26-1 0.01), off elsewhere. The
per-channel velocity gain is carried by 31-1 (−1.16 nats on 25 motif
notes), but 18 of 21 pieces are below zero (median −0.007) and the RMSE
gain holds on 19 of 21. Full tables in `results/motif_pilot2/report.md`.

## Power (pieces resampled, the report's bootstrap rule, 1000 draws)

| n pieces | articulation ΔRMSE, shared | per channel | timing ΔNLL, shared | per channel | velocity ΔNLL, per channel |
|---|---|---|---|---|---|
| 10 | 0.74 | 0.66 | 0.31 | 0.03 | 0.94 |
| 14 | 0.86 | 0.84 | 0.31 | 0.02 | 0.99 |
| 18 | 0.95 | 0.93 | 0.33 | 0.02 | 1.00 |
| 21 | 0.97 | 0.96 | 0.30 | 0.03 | 1.00 |

At two thirds of the measured effect, 18 pieces: articulation 0.51
(shared) and 0.39 (per channel); timing 0.14; velocity 0.14 (its star
rests on the skew of the per-piece distribution, which a shift removes).

## Reading (2026-11-17)

1. **H2 holds in one half and fails in the other.** The per-channel form
   removes the velocity penalty and turns it into a gain: velocity log
   score −0.086 [−0.200, −0.018], coverage 0.878 → 0.944, RMSE −0.0057*.
   It does not keep the timing log-score gain: −0.005 [−0.076, +0.054].
   By the rule written before the run, the family's timing value came
   from the shared coupling, not from the edges.
2. **The pilot-one timing star was seed-fragile.** On four seeds the
   shared weight gives −0.071 [−0.230, +0.097]; seeds 2 and 3 alone give
   −0.026 [−0.247, +0.259]. The effect is three pieces against one, with
   a median near zero. Timing calibration is not the family's robust
   effect, in either form.
3. **The robust effect is articulation accuracy**, present in both forms
   and on every seed pair: RMSE −0.0345* (shared) and −0.0189* (per
   channel), 0.732 → 0.698 and 0.713. The evidence says the same thing
   directly: it keeps the family on the articulation channel for ten of
   21 pieces and switches it off for timing on 19.
4. **The two forms trade.** The shared weight gives the larger
   articulation gain and a small timing RMSE gain (−0.0027*) at the cost
   of velocity coverage (0.835); the per-channel block gives a smaller
   articulation gain, velocity on both axes and coverage at nominal, and
   nothing on timing. The pitch term adds nothing in either form.
5. **Under random masking** the per-channel block is slightly worse on
   timing calibration over all hidden notes (+0.087, interval touching
   zero) while gaining on velocity; with 60 percent of every occurrence
   observed the family has little to add, as in pilot one.
6. **Scope.** Development side only, 21 pieces, four seeds, 400-note
   excerpts, score features without the music-model embeddings, annotated
   occurrences. Nothing registered was touched and no thesis claim
   follows from this record.

## What this changes

The registration draft (`docs/motif_prereg_design.md`) takes its primary
from this pilot's power table: M1 = articulation RMSE on hidden motif
notes with occurrences hidden (0.95 at 18 pieces at the measured effect,
0.51 at two thirds), M2 = the timing log score (0.33), M3 = velocity log
score and coverage reported. The motif arm named for M1 is the shared
weight `time_motif` (0.95 against 0.93 for the per-channel block at 18
pieces; the larger and more even articulation effect); `time_motif_pc`
runs beside it and is reported. The thesis chapter states the family's
measured value as articulation accuracy, with the timing effect of the
first study reported as what it turned out to be. Stage two (predicted
occurrences) runs next on the same cells.


## Erratum (2026-11-19, Codex audit findings 2 and 3)

- The articulation RMSE levels in reading 3 ("0.732 → 0.698 and 0.713") were
  computed as the time-only level minus the mean paired difference. The
  report table's arm levels are 0.7317 (time), 0.7045 (shared) and 0.7145
  (per-channel). The paired differences (−0.0345*, −0.0189*) are unchanged.
- Movements below zero on articulation RMSE: 14 (shared) and 12
  (per-channel) in the report table's per-movement definition (square root of
  the seed-mean MSE); 15 and 13 in the power script's (mean of per-seed RMSE).
  The thesis uses the table's definition.
