# Tracker calibration — URMP development side (pyin vs GT F0)

Reproduce: `OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/eval_tracker_calibration.py`
(pointer added 2026-08-19; the script existed but was not named here).

| instrument | tracks | med \|dev\| (c) | 90th pct | >50c | voicing recall |
|---|---|---|---|---|---|
| bn | 2 | 2.5 | 6.4 | 7.2% | 0.99 |
| cl | 7 | 3.3 | 9.1 | 0.8% | 1.00 |
| db | 3 | 3.1 | 14.8 | 4.7% | 0.99 |
| fl | 11 | 5.2 | 14.7 | 1.8% | 0.98 |
| hn | 2 | 2.8 | 7.9 | 0.4% | 1.00 |
| ob | 3 | 2.8 | 6.2 | 0.6% | 0.99 |
| sax | 8 | 3.9 | 10.2 | 0.7% | 0.99 |
| tba | 4 | 4.8 | 26.6 | 4.2% | 0.99 |
| tbn | 5 | 2.5 | 6.0 | 0.5% | 0.99 |
| tpt | 10 | 2.3 | 5.4 | 0.6% | 0.99 |
| va | 10 | 3.5 | 340.7 | 10.8% | 0.95 |
| vc | 9 | 3.4 | 23.3 | 5.2% | 0.95 |
| vn | 27 | 2.4 | 6.4 | 1.7% | 0.99 |

## Does pyin confidence predict error?

| prob quintile | med \|dev\| (c) | >50c |
|---|---|---|
| [0.01, 0.25] | 5.2 | 9.6% |
| [0.25, 0.48] | 3.0 | 3.2% |
| [0.48, 0.64] | 2.8 | 1.0% |
| [0.64, 0.76] | 2.7 | 0.3% |
| [0.76, 1.00] | 2.3 | 0.4% |

Median error monotonically decreasing with confidence: **True**; Spearman(prob, |dev|) = -0.262.

## Verdict (feeds docs/phase2_prereg_design.md)

1. **The tracker is accurate**: median |dev| 2–5 cents on every instrument —
   an order of magnitude below the vibrato extents of interest (20–40 c).
2. **Confidence is informative**: median error decreases monotonically with
   pyin's voicing probability and the gross-error rate falls 9.6% → 0.4%
   from the lowest to the highest quintile. Rule adopted for the estimator
   chain: **discard frames in the lowest confidence quintile (prob < ~0.25)
   before the per-note NLLS fit**; keep estimator variances as-given.
3. **Gross errors are octave/subharmonic slips in low registers**: the va
   tail is one track (K515 second viola, med 8.1 c, recall 0.84); bn/tba/db
   carry small tails. The robust-loss option in fit_vibrato_note and the
   confidence filter are the mitigations; flag low-register tracks in eval.
4. **Discovery**: arrangements of one composition share IDENTICAL track
   recordings (e.g. Rondeau 35/36/37 track 3 has byte-identical stats) —
   the composition-level split is literally required, not just prudent.
