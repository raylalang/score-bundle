#!/usr/bin/env python
"""Downstream task 4 — selective prediction (risk--coverage) on held-out ASAP.

The purest cash-in of the calibration claim: if the per-note predictive std is
*informative*, abstaining on the least-confident notes should drop the error on the
rest. We run the Phase-1 imputation grid, then for each cell sort held-out predictions
by predictive std and trace the risk--coverage curve.

Two numbers per cell:
  * ``AURC`` — area under the risk--coverage curve (lower = confident predictions really
    are the accurate ones);
  * ``excess`` = AURC(random abstention) − AURC — the part that isolates *uncertainty
    ranking quality* from raw accuracy (a model can have low AURC just by being accurate;
    high excess means its own uncertainty tells you where it is wrong).

Reported per channel and pooled, mean {zero, LM} × graph {off, on}. The claim: the graph
posterior's std yields a larger ``excess`` than the homoscedastic no-graph std (whose
per-note std is constant, so it cannot triage at all → excess ≈ 0). Numpy-only:

    python scripts/eval_asap_selective.py --arrays-cache .cache/asap_arrays_named.pkl
"""
from __future__ import annotations

import argparse

import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.downstream import load_piece_arrays, piece_score, selective_report
from score_bundle.lm import features as lmfeat


def bootstrap_ci(vals, B=2000, rng=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--embeddings", default="emb_leakfree",
                    choices=["emb", "emb_scoreonly", "emb_leakfree"])
    ap.add_argument("--noise-floor-frac", type=float, default=0.05)
    args = ap.parse_args()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    channels = ["tau", "log r", "v"]
    print(f"{len(head)} head + {len(ev)} eval pieces | embeddings={args.embeddings} "
          f"| noise_floor_frac={args.noise_floor_frac} | seeds={args.seeds}")

    H = np.concatenate([p[args.embeddings] for p in head])
    W = lmfeat.fit_prior_mean_head(H, np.concatenate([p["y"] for p in head]), l2=args.l2)

    means_list = ["zero", "LM"]
    # (mean, graph) -> per-(piece x seed x channel) pooled arrays for global curve,
    #                  and per-run selective summaries for CIs
    pool = {}
    per_run = {}
    for s in range(args.seeds):
        rng = np.random.default_rng(2000 + s)
        for p in ev:
            score = piece_score(p); y = p["y"]
            mask = ie.random_mask(len(y), rng, observed_frac=args.observed_frac)
            mu_lm = lmfeat.apply_prior_mean(p[args.embeddings], W)
            means = {"zero": np.zeros_like(y), "LM": mu_lm}
            cells = ie.impute_methods(score, y, means, mask, fit_hyper=True, rng=rng,
                                      noise_floor_frac=args.noise_floor_frac)
            for (mn, graph), cell in cells.items():
                if mn not in means_list:
                    continue
                key = (mn, graph)
                buf = pool.setdefault(key, [[], [], [], []])
                buf[0].append(cell.y); buf[1].append(cell.pred)
                buf[2].append(cell.std); buf[3].append(cell.channel)
                # per-run excess (pooled over channels within the run) for CIs
                rep = selective_report(cell.y, cell.pred, cell.std)
                per_run.setdefault(key, []).append(rep["excess"])
        print(f"seed {s + 1}/{args.seeds} done", flush=True)

    boot_rng = np.random.default_rng(7)
    print("\nSelective prediction (pooled over pieces x seeds)")
    print(f"{'mean':5s} {'graph':6s} {'chan':6s} {'RMSE':>7s} {'AURC':>7s} "
          f"{'excess':>8s} {'rmse@50%':>9s}")
    for mn in means_list:
        for graph in (False, True):
            key = (mn, graph)
            if key not in pool:
                continue
            y = np.concatenate(pool[key][0]); pr = np.concatenate(pool[key][1])
            sd = np.concatenate(pool[key][2]); ch = np.concatenate(pool[key][3])
            for ci, cname in enumerate(channels):
                sel = ch == ci
                rep = selective_report(y[sel], pr[sel], sd[sel])
                print(f"{mn:5s} {('on' if graph else 'off'):6s} {cname:6s} "
                      f"{rep['rmse']:7.3f} {rep['aurc']:7.3f} {rep['excess']:8.3f} "
                      f"{rep['rmse_at_50']:9.3f}")
            rep = selective_report(y, pr, sd)
            e_m, e_lo, e_hi = bootstrap_ci(per_run[key], rng=boot_rng)
            print(f"{mn:5s} {('on' if graph else 'off'):6s} {'POOL':6s} "
                  f"{rep['rmse']:7.3f} {rep['aurc']:7.3f} {rep['excess']:8.3f} "
                  f"{rep['rmse_at_50']:9.3f}   excess 95% CI [{e_lo:.3f},{e_hi:.3f}]")
    print("\nReading: excess > 0 means the model's own uncertainty ranks its errors "
          "(so abstaining helps).\nNo-graph rows use a single homoscedastic std per piece "
          "-> no within-piece ranking -> excess ~ 0.\nThe graph gives per-note std, so it "
          "can triage: larger excess = more useful uncertainty.")


if __name__ == "__main__":
    main()
