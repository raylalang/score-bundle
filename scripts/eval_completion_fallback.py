#!/usr/bin/env python
"""Extrapolation-safe completion: the Mahalanobis fallback (DEV ONLY).

The measured adaptation boundary (draft sec:boundary): per-piece Bayesian
feature weights, fit on an opening excerpt alone, can extrapolate wildly
(one prefix-25% cell at RMSE 1.6e4); the cross-piece read-out head is the
honest tool there.  Future Work names the fix: fall back when the
observed excerpt does not cover the feature space.  This study measures
exactly that rule:

  flag note i  iff  d_i^2 = (x_i - xbar_o)' (S_o + lam I)^{-1} (x_i - xbar_o)
                    > chi2_{q}(r)        (r = effective rank of S_o),

with x the per-piece GP's own feature columns (score features + mask-aware
embeddings), moments from OBSERVED notes only (deploy-legal), lam a small
ridge.  Flagged notes take the cross-piece head's prediction (mu_LM, with
the head pieces' per-channel residual SD as its honest scale); unflagged
notes keep the per-piece GP posterior.  q in {0.99, 0.999} and the pure
systems are all reported (exploratory, no tuning claim); random-mask
interpolation is included as the control where the rule should stay
silent.  Development pieces only; no registered artifact touched.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_completion_fallback.py [--fracs 0.25 0.5 0.75]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    from scipy import stats
except ImportError as exc:
    raise SystemExit("this study script needs scipy (chi2 quantiles)") from exc


def mahal_flags(X, mask, q, r_max=10):
    """Held-out flags from observed-excerpt feature moments (deploy-legal).

    A raw Mahalanobis test in ~90 feature dimensions with tens of observed
    notes degenerates (every point is off the rank-deficient subspace and
    everything flags).  The working rule tests coverage in the observed
    excerpt's own low-rank PCA frame: (i) whitened in-subspace distance
    against chi2_r (r = components covering 95% of observed variance,
    capped at ``r_max``), OR (ii) off-subspace energy beyond the observed
    notes' own q-quantile.  Both moments come from observed notes only.
    """
    Xo = X[mask]
    mu = Xo.mean(0)
    Z = Xo - mu
    U, sv, Vt = np.linalg.svd(Z / np.sqrt(max(len(Xo) - 1, 1)),
                              full_matrices=False)
    var = sv ** 2
    tot = var.sum() + 1e-12
    r = int(np.searchsorted(np.cumsum(var) / tot, 0.95) + 1)
    r = max(1, min(r, r_max, len(sv)))
    V = Vt[:r].T
    s = np.maximum(sv[:r], 1e-9)
    T = (X - mu) @ V / s                      # whitened in-subspace coords
    d2_in = np.sum(T ** 2, axis=1)
    P = (X - mu) @ V @ V.T
    e_off = np.sum(((X - mu) - P) ** 2, axis=1)
    thr_in = stats.chi2.ppf(q, r)
    thr_off = np.quantile(e_off[mask], q) * 1.5
    return (d2_in > thr_in) | (e_off > thr_off), d2_in


def main() -> None:
    from eval_downstream_gpfirst import gp_for
    from score_bundle.downstream import (block_mask, load_piece_arrays,
                                         piece_score, prefix_mask)
    from score_bundle.baselines import rich_score_features
    from score_bundle.lm import features as lmfeat
    from score_bundle.metrics import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named50.pkl")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args()

    head, ev, _ = load_piece_arrays(args.arrays_cache)
    # DEV PIECES ONLY -- the cache's eval list continues into the
    # confirmation pieces; the published convention (eval_downstream_gpfirst,
    # eval_asap_completion) slices to the first n_eval_pieces = 30.
    ev = ev[: args.n_eval_pieces]
    W = lmfeat.fit_prior_mean_head(
        np.concatenate([p["emb_leakfree"] for p in head]),
        np.concatenate([p["y"] for p in head]), l2=10.0)
    # the cross-piece head's honest per-channel scale, from head pieces only
    res = np.concatenate([p["y"] - lmfeat.apply_prior_mean(
        p["emb_leakfree"], W) for p in head])
    head_sd = res.std(axis=0, ddof=1)
    print(f"head scale per channel: {np.round(head_sd, 3)}")

    rows = {}
    for kind in ("prefix", "random"):
        for frac in args.fracs:
            rng = np.random.default_rng(0)
            flag_fracs = {0.99: [], 0.999: []}
            for pi, p in enumerate(ev):
                score = piece_score(p)
                Y = np.asarray(p["y"], dtype=float)
                n = len(Y)
                mask = (prefix_mask(n, frac) if kind == "prefix"
                        else rng.random(n) < frac)
                held = ~mask
                if held.sum() < 5 or mask.sum() < 10:
                    continue
                gp = gp_for(p, "GP-featlm")
                floor = 0.05 * np.array([max(float(np.var(Y[mask, c])), 1e-10)
                                         for c in range(3)])
                x_hat, _ = gp.fit(Y, mask, noise_floor=floor,
                                  maxiter=args.maxiter)
                M, S = gp.posterior(Y, mask, x_hat)
                nv = gp.unpack(x_hat)["noise"]
                mu_lm = lmfeat.apply_prior_mean(p["emb_leakfree"], W)

                Xf = rich_score_features(score, rff_dim=0)
                Xf = (Xf - Xf.mean(0)) / np.maximum(Xf.std(0), 1e-9)
                X = np.concatenate([Xf, np.asarray(p["emb_leakfree"])], axis=1)

                yt = np.concatenate([Y[held, c] for c in range(3)])
                pr_gp = np.concatenate([M[held, c] for c in range(3)])
                sd_gp = np.concatenate([np.sqrt(S[held, c] ** 2 + nv[c])
                                        for c in range(3)])
                pr_hd = np.concatenate([mu_lm[held, c] for c in range(3)])
                sd_hd = np.concatenate([np.full(int(held.sum()), head_sd[c])
                                        for c in range(3)])
                rows.setdefault((kind, frac, "GP-featlm"), []).append(
                    evaluate(yt, pr_gp, sd_gp))
                rows.setdefault((kind, frac, "head only"), []).append(
                    evaluate(yt, pr_hd, sd_hd))
                # the disagreement guard: distrust the per-piece prediction
                # where it strays implausibly far from the cross-piece head
                # (per (note, channel); deploy-legal -- both predictions and
                # the head scale are available at deploy time)
                fg = np.abs(pr_gp - pr_hd) > 3.0 * sd_hd
                for q in (0.99, 0.999):
                    flags, _ = mahal_flags(X, mask, q)
                    fh = np.concatenate([flags[held]] * 3)
                    for name, ff in ((f"mahal q={q}", fh),
                                     ("guard 3sd", fg) if q == 0.99 else
                                     (None, None),
                                     (f"either q={q}", fh | fg)):
                        if name is None:
                            continue
                        pr = np.where(ff, pr_hd, pr_gp)
                        sd = np.where(ff, sd_hd, sd_gp)
                        rows.setdefault((kind, frac, name),
                                        []).append(evaluate(yt, pr, sd))
                    flag_fracs[q].append(float(flags[held].mean()))
                rows.setdefault((kind, frac, "guardfrac"), []).append(
                    float(fg.mean()))
                print(f"[{kind} {frac}] piece {pi} done", flush=True)
            for q in (0.99, 0.999):
                if flag_fracs[q]:
                    rows[(kind, frac, f"flagfrac q={q}")] = flag_fracs[q]

    import pickle
    os.makedirs("results/completion_fallback", exist_ok=True)
    tag = "-".join(f"{f:g}" for f in args.fracs)
    with open(f"results/completion_fallback/rows_{tag}.pkl", "wb") as fh:
        pickle.dump(rows, fh)                  # persist BEFORE any printing

    print("\n=== completion with the Mahalanobis fallback (dev, "
          f"{len(ev)} pieces) ===")
    for (kind, frac, name), vals in sorted(rows.items()):
        if name.startswith(("flagfrac", "guardfrac")):
            print(f"{kind:6s} {frac:.2f} {name:16s} "
                  f"mean {np.mean(vals):.3f} max {np.max(vals):.3f}")
            continue
        rm = np.array([v["rmse"] for v in vals])
        nl = np.array([v["nll"] for v in vals])
        cv = np.array([v["coverage@0.90"] for v in vals])
        print(f"{kind:6s} {frac:.2f} {name:16s} RMSE mean {rm.mean():10.3f} "
              f"median {np.median(rm):7.3f} worst {rm.max():10.1f} | "
              f"NLL med {np.median(nl):7.2f} | cov {cv.mean():.2f}")


if __name__ == "__main__":
    main()
