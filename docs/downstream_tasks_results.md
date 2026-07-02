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

<!-- pending: logs/denoise.log -->
