# Downstream tasks — does the score-graph prior earn its keep beyond imputation?

Three downstream demonstrations chosen (from the candidate list: rendering, anomaly
detection, performer classification, denoising, masked completion) for having an
objective metric, an honest baseline, and direct reuse of the Phase-1 machinery:

1. **Performance completion / expressive rendering** — predict unheard notes from a
   small observed excerpt (prefix = pure extrapolation, the rendering use-case; block =
   gap extrapolation; random = the Phase-1 interpolation setting as reference).
2. **Performance-error (anomaly) detection** — inject controlled errors, rank notes by
   leave-one-out predictive surprise; the task that directly *cashes in* calibration.
3. **Transcription denoising** — observe every note through synthetic noise (the
   ATEPP/Aria-MIDI scaling scenario) and recover clean values by posterior shrinkage.

Rejected: performer/style classification (ASAP performer labels are sparse and
confounded with piece identity — no honest eval); masked-note completion (that *is*
Phase 1).

**Common setup.** 30 held-out ASAP eval pieces (≤400 notes each), piece-disjoint from
the 40-piece head split (used only to fit the LM head / calibration scales),
contamination-filtered against MAESTRO Phase-0 pretraining, provenance recorded in the
array cache (`scripts/extract_asap_arrays.py`). All LM means use **score-only
embeddings** (constant placeholder velocity — see the leakage correction in
`phase1_calibration_results.md`). Targets `y = [tau, log r, v]`. Chance/na "off" rows are
the same baselines as Phase 1 (zero / ridge / LM mean without the graph).

---

## Task 2 — performance-error (anomaly) detection

Corrupt 5% of notes with a ±`scale`·(channel std) shift; rank all notes by surprise:
graph **on** = EB-fit GMRF (noise-floored) + leave-one-out predictive NLL; graph
**off** = homoscedastic z-score around the same mean. AUROC / average precision over
30 pieces × 2 seeds, bootstrap 95% CIs over piece×seed runs.
Reproduce: `scripts/eval_asap_anomaly.py` (logs `logs/anomaly_s3.log`, `logs/anomaly_s2.log`).

### scale = 3σ errors (`logs/anomaly_s3.log`)

```
channel  mean  graph            AUROC [95% CI]       AP
tau      zero  off      0.974 [0.969,0.978]      0.536
tau      zero  on       0.979 [0.976,0.982]      0.582
tau      LM    off      0.943 [0.924,0.957]      0.487
tau      LM    on       0.958 [0.938,0.973]      0.547
log r    zero  off      0.906 [0.878,0.928]      0.573
log r    zero  on       0.981 [0.977,0.984]      0.763
log r    LM    off      0.962 [0.951,0.971]      0.704
log r    LM    on       0.982 [0.978,0.985]      0.775
v        zero  off      0.950 [0.938,0.960]      0.727
v        zero  on       0.991 [0.988,0.993]      0.882
v        LM    off      0.950 [0.932,0.964]      0.738
v        LM    on       0.990 [0.987,0.993]      0.876

pooled   zero  off      0.943 [0.933,0.953]
pooled   zero  on       0.984 [0.982,0.986]
pooled   LM    off      0.952 [0.943,0.960]
pooled   LM    on       0.977 [0.969,0.982]
```

**Verdict: the graph helped, on every channel and both means** — pooled AUROC
0.943 → 0.984 (zero mean; non-overlapping CIs) and 0.952 → 0.977 (LM mean); AP gains are
larger still (v: 0.73 → 0.88). Honest note: `zero + graph` matches `LM + graph` here —
for error detection the structured residual, not the learned mean, does the work.

<!-- scale=2σ table to be appended when the run completes -->

---

## Task 1 — performance completion / rendering

<!-- pending: logs/completion.log -->

---

## Task 3 — transcription denoising

Every note observed through i.i.d. Gaussian noise (std = `level` × per-channel std,
simulating a noisy AMT transcription); recover the clean values. Metrics vs the clean
targets, latent posterior std as the calibration object; `identity` (the noisy data
itself, oracle noise std) is calibrated by construction and is the RMSE floor to beat.
`independent` (scalar Wiener shrinkage) and `graph-oracle` are told the true noise;
`graph` is fully blind (EB noise estimate, 5% floor). 30 pieces, pooled over channels,
bootstrap 95% CIs over pieces. Reproduce: `scripts/eval_asap_denoise.py`
(log `logs/denoise.log`).

```
 level mean  method                RMSE [95% CI]                NLL              cov@.9
  0.50 -     identity       0.2200                       -0.945 [-1.16,-0.74]    0.902
  0.50 LM    graph          0.2759                       2.5e6  (collapse)       0.638
  0.50 LM    graph-oracle   0.1894                       -1.091 [-1.30,-0.87]    0.901
  0.50 LM    independent    0.1983                       -1.039 [-1.25,-0.83]    0.897
  0.50 zero  graph          0.2908                       1.7e4  (collapse)       0.636
  0.50 zero  graph-oracle   0.1915                       -1.099 [-1.31,-0.87]    0.903
  0.50 zero  independent    0.2106                       -1.020 [-1.26,-0.79]    0.891
  1.00 -     identity       0.4432                       -0.250 [-0.47,-0.03]    0.901
  1.00 LM    graph          0.3559                       7.5e10 (collapse)       0.564
  1.00 LM    graph-oracle   0.2927                       -0.642 [-0.85,-0.42]    0.902
  1.00 LM    independent    0.3153                       -0.550 [-0.76,-0.34]    0.897
  1.00 zero  graph          0.3685                       4.0e11 (collapse)       0.536
  1.00 zero  graph-oracle   0.2991                       -0.653 [-0.88,-0.43]    0.903
  1.00 zero  independent    0.3601                       -0.502 [-0.74,-0.27]    0.886
```

**Verdict: with a known noise level the graph helped on BOTH axes** — `graph-oracle`
beats the independent-shrinkage oracle baseline on RMSE at every level and mean (e.g.
0.293 vs 0.315 at level 1.0 with the LM mean; 0.299 vs 0.360 with the zero mean) *and*
on NLL, at matched ~0.90 coverage. The structured prior extracts more signal from the
same noisy observations than per-note shrinkage, exactly as the thesis argues.

**Honest negative: fully-blind denoising does not work yet.** Estimating the noise level
per piece by EB (even with the 5% noise floor) systematically underestimates it on real
performance data: intervals collapse (coverage 0.54–0.64, NLL explodes) and at low noise
the over-trusting posterior is even worse than identity on RMSE (0.276–0.291 vs 0.220).
The `graph-calib` calibration-split variant collapses the same way. Denoising
applications should treat the transcription noise scale as (approximately) known — e.g.
calibrated once per AMT system, or estimated jointly across many pieces (future work) —
rather than per piece.
