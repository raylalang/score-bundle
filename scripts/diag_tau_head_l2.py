#!/usr/bin/env python
"""Diagnostic: does the head l2 (10 vs 100) flip the tau graph-on EB blowups?

The feature-baseline run (CV-selected l2=100 for the LM head) showed LM+graph tau
cells blowing up (pooled tau RMSE 1.55 / NLL 13.9) while every published l2=10 run
had the same cell stable (~0.158). Same protocol otherwise. This isolates l2:
identical masks, two heads on the SAME emb_leakfree embeddings, graph on, count
tau cells with RMSE > 0.5 and report pooled tau metrics per head.

Result (2026-07-03, logs/diag_tau_l2.log): the blowup is a single catastrophic EB
fit collapse (seed 2, piece 28, tau RMSE 17.0) that exists only under the l2=100
head; see docs/phase1_calibration_results.md.
"""
import sys
import time

import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.downstream import load_piece_arrays, piece_score
from score_bundle.lm import features as lmfeat
from score_bundle.metrics import evaluate

GUARD = "--guard" in sys.argv
print(f"guard={'on' if GUARD else 'off'}", flush=True)

head, ev, meta = load_piece_arrays(".cache/asap_arrays_named.pkl")
ev = ev[:30]
Yh = np.concatenate([p["y"] for p in head])
H = np.concatenate([p["emb_leakfree"] for p in head])
W = {l2: lmfeat.fit_prior_mean_head(H, Yh, l2=l2) for l2 in (10.0, 100.0)}
print("heads fit", flush=True)

pool = {l2: [[], [], []] for l2 in W}          # y, pred, std (tau, graph on)
blow = {l2: [] for l2 in W}
for s in range(4):
    seed_rng = np.random.default_rng(1000 + s)
    for pi, p in enumerate(ev):
        t0 = time.time()
        score, y = piece_score(p), p["y"]
        mask = ie.random_mask(len(y), seed_rng, observed_frac=0.6)
        means = {f"LM{int(l2)}": lmfeat.apply_prior_mean(p["emb_leakfree"], w)
                 for l2, w in W.items()}
        cells = ie.impute_methods(score, y, means, mask, fit_hyper=True,
                                  rng=seed_rng, noise_floor_frac=0.05, guard=GUARD)
        for l2 in W:
            cell = cells[(f"LM{int(l2)}", True)]
            t = cell.channel == 0
            yy, pp_, ss_ = cell.y[t], cell.pred[t], cell.std[t]
            pool[l2][0].append(yy); pool[l2][1].append(pp_); pool[l2][2].append(ss_)
            r = float(np.sqrt(np.mean((yy - pp_) ** 2)))
            if r > 0.5:
                blow[l2].append((s, pi, r))
        print(f"  s{s} piece {pi:2d} n={len(y):5d} {time.time() - t0:6.1f}s", flush=True)
    print(f"seed {s + 1}/4 done", flush=True)

np.savez(".cache/diag_tau_l2_pool.npz",
         **{f"{int(l2)}_{name}": np.concatenate(v)
            for l2, vs in pool.items() for name, v in zip(("y", "pred", "std"), vs)},
         **{f"{int(l2)}_blow": np.asarray(blow[l2], dtype=float).reshape(-1, 3)
            for l2 in blow})
for l2 in W:
    m = evaluate(*[np.concatenate(v) for v in pool[l2]], level=0.9)
    print(f"\nl2={l2:g}  pooled tau (graph on): RMSE {m['rmse']:.4f}  NLL {m['nll']:.4f} "
          f"cov {m['coverage@0.90']:.3f}  | blowup cells (RMSE>0.5): {len(blow[l2])}/120")
    for s, pi, r in sorted(blow[l2], key=lambda t: -t[2])[:8]:
        print(f"    seed {s} piece {pi:2d}  tau RMSE {r:.3f}")
