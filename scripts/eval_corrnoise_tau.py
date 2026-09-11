#!/usr/bin/env python
"""Correlated tau noise row: the failed C4 claim's named follow-up (DEV ONLY).

Phase-2's one failed registered claim (timing calibration) names its own
remedy in the thesis: alignment error is correlated along score time, and
the diagonal noise row provably cannot represent it.  This study measures
an AR(1)-correlated tau-noise block on the development split:

  Sigma_tau(rho)_ij = sqrt(d_i d_j) * rho^{|i-j|}   (i, j in note order),

with d_i the as-given tau cell variances (the LOO-warp predictive
variances) and rho profiled per (track, seed) on a grid by the OBSERVED
block's marginal likelihood at the fitted hyperparameters -- deploy-legal,
no held-out peeking; rho = 0 reproduces the published diagonal path and
is the paired baseline.  Predictions for held-out tau observations then
use the full covariance T = C + Sigma_n (noise cross-covariance included:
a neighbour's warp error informs mine, which is the point of modelling
the correlation).

Everything else mirrors eval_phase2_real.stage_run: same cache, same
masks and seeds, same as-given system, same eval rules.  The registered
pipeline and the spent confirmation are untouched.  Exploratory, no
claims.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_corrnoise_tau.py run [--shard K/N]
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_corrnoise_tau.py report
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_phase2_real import (C_MAX, CACHE, CH, HOLD_FRAC, MIN_NOTES,  # noqa: E402
                              SEEDS, _Z90)

TAU = 4                          # channel index of tau in the 6-channel bundle
RHO_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
OUT_DIR = "results/corrnoise_tau"
# --joint: rho chosen by the JOINT evidence -- all hyperparameters
# re-optimized (warm-started) at each rho instead of profiled at the
# rho=0 fit; --seeds overrides the mask seeds (fresh-seed robustness).


def cell_indices(mask):
    """Flattened channel-major indices (c*N + i) of True cells."""
    N, k = mask.shape
    idx = []
    for c in range(k):
        idx.append(c * N + np.where(mask[:, c])[0])
    return np.concatenate(idx)


def run(shard_k: int, shard_n: int, joint: bool = False,
        seeds=None, out_tag: str = "") -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.gp import MultiOutputGraphGP
    from score_bundle.optimize import nelder_mead
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.score import Score

    use_seeds = tuple(seeds) if seeds else SEEDS
    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    n_ch = len(CH)
    recs = []
    keys = sorted(data)
    for ti, key in enumerate(keys):
        seeds_here = [s for si, s in enumerate(use_seeds)
                      if (ti * len(use_seeds) + si) % shard_n == shard_k]
        if not seeds_here:
            continue
        d = data[key]
        est = np.concatenate([d["est"], d["ell"][:, None],
                              d["tau"][:, None], d["dvib"][:, None]], axis=1)
        var = np.concatenate([d["var"], d["var_ell"][:, None],
                              d["var_tau"][:, None],
                              d["var_dvib"][:, None]], axis=1)
        ident = d["ident"]
        n = est.shape[0]
        usable = np.isfinite(est[:, 0]) & (np.abs(est[:, 0]) <= C_MAX)
        if usable.sum() < MIN_NOTES:
            continue
        score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                                  np.zeros(n, dtype=int))
        eig = np.linalg.eigh(laplacian(build_adjacency(score)))
        X = rich_score_features(score, rff_dim=0)
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
        scale = np.ones((n, n_ch))
        for c in range(n_ch):
            v = var[:, c]
            obs_c = np.isfinite(v)
            med = np.median(v[obs_c]) if obs_c.any() else 1.0
            scale[:, c] = np.where(np.isfinite(v),
                                   np.clip(v / max(med, 1e-12), 1e-2, 1e3),
                                   1.0)
        med_var = np.array([np.median(var[:, c][np.isfinite(var[:, c])])
                            if np.isfinite(var[:, c]).any() else 1.0
                            for c in range(n_ch)])
        for seed in seeds_here:
            rng = np.random.default_rng(1000 + 7 * key[0] + key[1] + seed)
            held = (rng.random(n) < HOLD_FRAC) & usable
            mask = np.zeros((n, n_ch), dtype=bool)
            mask[:, 0] = usable & ~held
            mask[:, 1] = mask[:, 2] = usable & ~held & ident
            mask[:, 3] = usable & ~held & np.isfinite(est[:, 3])
            mask[:, 4] = ~held & np.isfinite(est[:, 4])
            mask[:, 5] = ~held & np.isfinite(est[:, 5])
            if mask[:, 0].sum() < 15 or held.sum() < 5:
                continue
            Yobs = np.where(mask, np.nan_to_num(est), 0.0)
            t0 = time.time()

            floor = 0.05 * np.array([float(np.var(Yobs[mask[:, c], c]))
                                     if mask[:, c].sum() > 2 else 1.0
                                     for c in range(n_ch)])
            g = MultiOutputGraphGP(eig[0], eig[1], kernel="additive",
                                   features=feats, n_channels=n_ch)
            g.noise_scale = scale
            x_hat, _ = g.fit(Yobs, mask, noise_floor=floor, maxiter=200,
                             noise_fixed=med_var)
            p = g.unpack(x_hat)

            # full-cell covariance T(rho) = C + Sigma_n(rho), channel-major
            allidx = np.arange(n)
            C = g._blocks(p, allidx, allidx)
            noise_d = g._cell_noise(p)              # (k N,) diagonal
            obs = cell_indices(mask)
            y_o = np.concatenate([Yobs[mask[:, c], c] for c in range(n_ch)])
            # held-out tau observation cells (the est-target eval rule)
            h_tau = (~mask[:, TAU]) & held & np.isfinite(est[:, TAU])
            h_idx = TAU * n + np.where(h_tau)[0]
            if h_idx.size < 3:
                continue
            d_tau = noise_d[TAU * n:(TAU + 1) * n]
            order = np.arange(n)
            lag = np.abs(order[:, None] - order[None, :])
            root = np.sqrt(np.outer(d_tau, d_tau))

            def T_of(rho):
                Sig = np.diag(noise_d.copy())
                blk = root * (rho ** lag) if rho > 0 else np.diag(d_tau)
                Sig[TAU * n:(TAU + 1) * n, TAU * n:(TAU + 1) * n] = blk
                return C + Sig

            def obs_lml(T):
                To = T[np.ix_(obs, obs)]
                sign, logdet = np.linalg.slogdet(To)
                if sign <= 0:
                    return -np.inf
                a = np.linalg.solve(To, y_o)
                return float(-0.5 * (y_o @ a + logdet
                                     + y_o.size * np.log(2 * np.pi)))

            def score_at(T):
                To = T[np.ix_(obs, obs)]
                Tho = T[np.ix_(h_idx, obs)]
                m_h = Tho @ np.linalg.solve(To, y_o)
                v_h = (T[h_idx, h_idx]
                       - np.einsum("ij,ji->i", Tho,
                                   np.linalg.solve(To, Tho.T)))
                v_h = np.clip(v_h, 1e-12, None)
                err = est[h_tau, TAU] - m_h
                s = np.sqrt(v_h)
                return {"rmse": float(np.sqrt(np.mean(err ** 2))),
                        "nll": float(np.mean(0.5 * np.log(2 * np.pi * v_h)
                                             + 0.5 * (err / s) ** 2)),
                        "cov": float(np.mean(np.abs(err) <= _Z90 * s)),
                        "n": int(h_idx.size)}

            if not joint:
                lmls = [obs_lml(T_of(r)) for r in RHO_GRID]
                i_hat = int(np.argmax(lmls))
                rho_hat = RHO_GRID[i_hat]
                base = score_at(T_of(0.0))
                corr = score_at(T_of(rho_hat))
            else:
                # JOINT: warm-started re-optimization of every
                # hyperparameter at each rho; rho by the joint evidence.
                floor_log = np.log(np.maximum(floor, 1e-12))

                def joint_lml(xv, rho):
                    pv = g.unpack(xv)
                    Cv = g._blocks(pv, allidx, allidx)
                    nd = g._cell_noise(pv)
                    Sig = np.diag(nd.copy())
                    dt = nd[TAU * n:(TAU + 1) * n]
                    if rho > 0:
                        Sig[TAU * n:(TAU + 1) * n, TAU * n:(TAU + 1) * n] = \
                            np.sqrt(np.outer(dt, dt)) * (rho ** lag)
                    To = (Cv + Sig)[np.ix_(obs, obs)]
                    sign, logdet = np.linalg.slogdet(To)
                    if sign <= 0:
                        return -np.inf
                    a = np.linalg.solve(To, y_o)
                    return float(-0.5 * (y_o @ a + logdet
                                         + y_o.size * np.log(2 * np.pi)))

                # profile first (cheap) to locate the neighbourhood, then
                # a short warm-started joint refit at the winner and its
                # grid neighbours only -- the full 7-point joint sweep is
                # needlessly expensive on large tracks.
                prof = [obs_lml(T_of(r)) for r in RHO_GRID]
                i0 = int(np.argmax(prof))
                cand = sorted({RHO_GRID[j] for j in
                               (max(i0 - 1, 0), i0,
                                min(i0 + 1, len(RHO_GRID) - 1))})
                best = (-np.inf, 0.0, x_hat)
                x_warm = x_hat
                for rho in cand:
                    def neg(xv, rho=rho):
                        xc = xv.copy()
                        xc[-n_ch:] = np.maximum(xc[-n_ch:], floor_log)
                        try:
                            v = -joint_lml(xc, rho)
                        except (np.linalg.LinAlgError, ValueError):
                            return 1e12
                        return v if np.isfinite(v) else 1e12

                    x_r = nelder_mead(neg, x_warm, step=0.1, max_iter=40)
                    x_r[-n_ch:] = np.maximum(x_r[-n_ch:], floor_log)
                    lml_r = joint_lml(x_r, rho)
                    if lml_r > best[0]:
                        best = (lml_r, rho, x_r.copy())
                    x_warm = x_r
                _, rho_hat, x_j = best
                base = score_at(T_of(0.0))        # at the rho=0 fit x_hat
                p = g.unpack(x_j)                 # rebuild T pieces at x_j
                C = g._blocks(p, allidx, allidx)
                noise_d = g._cell_noise(p)
                d_tau = noise_d[TAU * n:(TAU + 1) * n]
                root = np.sqrt(np.outer(d_tau, d_tau))
                lmls = []
                corr = score_at(T_of(rho_hat))    # at the joint fit x_j
            recs.append({"key": key, "seed": seed, "rho": rho_hat,
                         "lmls": lmls, "base": base, "corr": corr,
                         "instr": d["instrument"]})
            print(f"  {key} seed {seed}: rho={rho_hat:.2f} "
                  f"dNLL={corr['nll'] - base['nll']:+.3f} "
                  f"dRMSE={corr['rmse'] - base['rmse']:+.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR,
                       f"cells{out_tag}.shard{shard_k}of{shard_n}.pkl")
    pickle.dump(recs, open(out, "wb"))
    print(f"wrote {out} ({len(recs)} cells)")


def report(tag: str = "") -> None:
    recs = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR,
                                           f"cells{tag}.shard*.pkl"))):
        recs += pickle.load(open(f, "rb"))
    print(f"{len(recs)} (track, seed) cells")
    rhos = np.array([r["rho"] for r in recs])
    print(f"\nchosen rho: median {np.median(rhos):.2f}, "
          f"rho>0 on {np.mean(rhos > 0):.0%} of cells, "
          f"histogram: " + " ".join(
              f"{r:.2f}:{int((rhos == r).sum())}" for r in RHO_GRID))
    keys = [r["key"] for r in recs]
    rng = np.random.default_rng(0)
    for met in ("nll", "cov", "rmse"):
        a = np.array([r["corr"][met] for r in recs])
        b = np.array([r["base"][met] for r in recs])
        dlt = a - b
        uk = sorted(set(keys))
        per = {k: [] for k in uk}
        for v, k in zip(dlt, keys):
            per[k].append(v)
        means = []
        for _ in range(2000):
            pick = rng.choice(len(uk), len(uk), replace=True)
            means.append(np.concatenate([per[uk[j]] for j in pick]).mean())
        lo, hi = np.percentile(means, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"tau {met:4s}: corr {np.mean(a):7.4f} vs diag "
              f"{np.mean(b):7.4f}  paired dmean {dlt.mean():+7.4f} "
              f"[{lo:+.4f},{hi:+.4f}]{star} (median d {np.median(dlt):+.4f})")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        sys.exit(__doc__)
    tag = ""
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
    if sys.argv[1] == "report":
        report(tag)
        return
    shard_k, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_k, shard_n = map(int, sys.argv[
            sys.argv.index("--shard") + 1].split("/"))
    seeds = None
    if "--seeds" in sys.argv:
        i = sys.argv.index("--seeds") + 1
        seeds = []
        while i < len(sys.argv) and sys.argv[i].isdigit():
            seeds.append(int(sys.argv[i]))
            i += 1
    run(shard_k, shard_n, joint="--joint" in sys.argv, seeds=seeds,
        out_tag=tag)


if __name__ == "__main__":
    main()
