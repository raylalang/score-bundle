#!/usr/bin/env python
"""Guard A/B on the published protocol: does the EB guard change the headline?

Same piece x seed x mask, LM mean (emb_leakfree, l2=10 head), graph on, EB with the
published noise floor — once with guard=False (published behavior) and once with
guard=True (`fit_laplacian_field_guarded`). The guard must (a) leave the pooled
headline unchanged within noise on healthy fits, (b) tame the piece-28 collapse when
it strikes. Numpy-only.

    python scripts/eval_guard_ab.py
"""
import sys
import time

import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.downstream import load_piece_arrays, piece_score
from score_bundle.lm import features as lmfeat
from score_bundle.metrics import evaluate

head, ev, meta = load_piece_arrays(".cache/asap_arrays_named.pkl")
ev = ev[:30]
Yh = np.concatenate([p["y"] for p in head])
H = np.concatenate([p["emb_leakfree"] for p in head])
W = lmfeat.fit_prior_mean_head(H, Yh, l2=10.0)
print("head fit (emb_leakfree, l2=10)", flush=True)

pool = {g: [[], [], [], []] for g in (False, True)}   # y, pred, std, channel
tau_cells = {g: [] for g in (False, True)}
for s in range(4):
    seed_rng_a = np.random.default_rng(1000 + s)
    seed_rng_b = np.random.default_rng(1000 + s)
    for pi, p in enumerate(ev):
        t0 = time.time()
        score, y = piece_score(p), p["y"]
        mask = ie.random_mask(len(y), seed_rng_a, observed_frac=0.6)
        mask_b = ie.random_mask(len(y), seed_rng_b, observed_frac=0.6)
        assert (mask == mask_b).all()
        mu = lmfeat.apply_prior_mean(p["emb_leakfree"], W)
        for g, rng in ((False, seed_rng_a), (True, seed_rng_b)):
            cell = ie.impute_methods(score, y, {"LM": mu}, mask, fit_hyper=True,
                                     rng=rng, noise_floor_frac=0.05,
                                     guard=g)[("LM", True)]
            for i, arr in enumerate((cell.y, cell.pred, cell.std, cell.channel)):
                pool[g][i].append(arr)
            t = cell.channel == 0
            r = float(np.sqrt(np.mean((cell.y[t] - cell.pred[t]) ** 2)))
            tau_cells[g].append((s, pi, r))
        print(f"  s{s} p{pi:2d}  {time.time() - t0:5.1f}s", flush=True)
    print(f"seed {s + 1}/4 done", flush=True)

print(f"\n{'guard':6s} {'RMSE':>8s} {'NLL':>9s} {'cov@.9':>8s}   (pooled, graph on)")
for g in (False, True):
    y, pr, sd, _ = (np.concatenate(v) for v in pool[g])
    m = evaluate(y, pr, sd, level=0.9)
    print(f"{('on' if g else 'off'):6s} {m['rmse']:8.4f} {m['nll']:9.4f} "
          f"{m['coverage@0.90']:8.3f}")
for g in (False, True):
    bad = [c for c in tau_cells[g] if c[2] > 0.5]
    print(f"guard={'on' if g else 'off'}: tau cells > 0.5: {len(bad)}/120 "
          + " ".join(f"(s{s},p{pi}:{r:.2f})" for s, pi, r in bad))
