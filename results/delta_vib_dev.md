# delta_vib channel-candidate study (URMP development side)

Gated estimator = draft eq:vibrato exactly; criterion C-A..C-D stated in scripts/eval_delta_vib.py BEFORE the numbers were computed.

- C-A coverage: delta identifiable on 94.8% of vibrato-identifiable notes on GT curves (7176/7569), 96.6% on pyin curves (6652/6884) — PASS (>= 25% both)
- C-B agreement: median |delta_pyin - delta_GT| = 18 ms on 5444 jointly identifiable notes — PASS (<= 40 ms)
- C-C non-degradation: gated intonation median |err| 2.87 c vs ungated 3.02 c vs GT quasi-truth — PASS (<= 1.10x)
- C-D signal: median per-track spread(delta)/median-SE = 8.08 — PASS (> 1)

**VERDICT: delta_vib IN** (all four must pass).
