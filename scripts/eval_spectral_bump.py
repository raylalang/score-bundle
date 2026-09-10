#!/usr/bin/env python
"""A learned spectral filter on the graph spectrum: closing the kernel question (DEV).

The kernel comparison measured fixed classical profiles (Matern, heat,
additive) as statistical ties, and the spectral-overlay appendix says why:
the evidence is insensitive where the filters differ.  This study closes
the remaining "maybe a LEARNED filter helps" question: the additive
profile plus one free evidence-fitted Gaussian bump,

    g(nu) = [ 1/(1 + s nu) + a * exp(-(nu - m)^2 / (2 w^2)) ] / g(0),

with (a, m, w) optimized jointly with every other hyperparameter by the
per-piece evidence (a -> 0 recovers the additive baseline exactly: the
family is nested).  If the evidence uses the bump nowhere and the scores
tie, the filter-inertness conclusion holds against a learned filter, not
only against fixed alternatives.  Config b_feat (score features, plain
graph), same masks as every published number; the paired baseline is the
published b_feat cells (results/graphgp_v2).  Development only,
exploratory, no claims.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_spectral_bump.py run [--shard K/N]
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_spectral_bump.py report
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

INPUTS = ".cache/kernel_sweep_inputs.pkl"
OUT_DIR = "results/spectral_bump"
N_EVAL = 30


def bump_filter(nu, s, a, m, w):
    g = 1.0 / (1.0 + s * nu) + a * np.exp(-(nu - m) ** 2 / (2.0 * w ** 2))
    g0 = 1.0 / (1.0 + s * 0.0) + a * np.exp(-m ** 2 / (2.0 * w ** 2))
    return g / g0


def run(shard_k: int, shard_n: int) -> None:
    from score_bundle import gp as gpmod
    from score_bundle.downstream import load_piece_arrays, piece_score
    from score_bundle.baselines import rich_score_features
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.gp import MultiOutputGraphGP
    from score_bundle.optimize import nelder_mead

    with open(INPUTS, "rb") as fh:
        inputs = pickle.load(fh)
    masks, imeta = inputs["masks"], inputs["meta"]
    head, ev, _ = load_piece_arrays(".cache/asap_arrays_named50.pkl")
    ev = ev[:N_EVAL]                          # DEV pieces only
    seeds = imeta["seeds"]

    cells = {}
    t00 = time.time()
    for s_ in range(seeds):
        for pi, p in enumerate(ev):
            if (s_ * len(ev) + pi) % shard_n != shard_k:
                continue
            t0 = time.time()
            Y = np.asarray(p["y"], dtype=float)
            mask = masks[(pi, s_)]
            held = ~mask
            score = piece_score(p)
            X = rich_score_features(score, rff_dim=0)
            X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
            feats = [np.concatenate([X, np.ones((len(X), 1))], axis=1)]
            nu, U = np.linalg.eigh(laplacian(build_adjacency(score)))
            nu_max = float(nu.max())
            floor = 0.05 * np.array([float(np.var(Y[mask, c]))
                                     for c in range(3)])

            # bump params z = [log a, logit-frac position, log w]
            def set_kernel(z):
                a = float(np.exp(z[0]))
                m = float(nu_max / (1.0 + np.exp(-z[1])))
                w = float(np.exp(z[2]) * nu_max)
                gpmod.SHAPE_KERNELS["bump_tmp"] = (
                    lambda nu_, s_p, a=a, m=m, w=w:
                    bump_filter(nu_, s_p, a, m, w))
                return a, m, w

            g = None

            def neg(z):
                nonlocal g
                set_kernel(z[:3])
                g = MultiOutputGraphGP(nu, U, kernel="bump_tmp",
                                       features=feats, n_channels=3)
                xg = z[3:].copy()
                xg[-3:] = np.maximum(xg[-3:], np.log(np.maximum(floor, 1e-12)))
                try:
                    v = -g.log_marginal_likelihood(Y, mask, xg)
                except (np.linalg.LinAlgError, ValueError):
                    return 1e12
                return v if np.isfinite(v) else 1e12

            g0 = MultiOutputGraphGP(nu, U, kernel="additive",
                                    features=feats, n_channels=3)
            z0 = np.concatenate([[np.log(0.05), 0.0, np.log(0.25)], g0.x0()])
            best = nelder_mead(neg, z0, step=0.4, max_iter=1200)
            best = nelder_mead(neg, best, step=0.1, max_iter=400)
            a_hat, m_hat, w_hat = set_kernel(best[:3])
            x_hat = best[3:].copy()
            x_hat[-3:] = np.maximum(x_hat[-3:],
                                    np.log(np.maximum(floor, 1e-12)))
            M, S = g.posterior(Y, mask, x_hat)
            nv = g.unpack(x_hat)["noise"]
            yt, pr, sd, ch = [], [], [], []
            for c in range(3):
                yt.append(Y[held, c]); pr.append(M[held, c])
                sd.append(np.sqrt(S[held, c] ** 2 + nv[c]))
                ch.append(np.full(int(held.sum()), c, dtype=int))
            cells[("GP", pi, s_)] = {
                "cell": (np.concatenate(yt), np.concatenate(pr),
                         np.concatenate(sd), np.concatenate(ch)),
                "bump": {"a": a_hat, "m": m_hat / nu_max,
                         "w": w_hat / nu_max}}
            print(f"  piece {pi} seed {s_}: a={a_hat:.4f} "
                  f"m={m_hat / nu_max:.2f} w={w_hat / nu_max:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"cells.shard{shard_k}of{shard_n}.pkl")
    pickle.dump(cells, open(out, "wb"))
    print(f"wrote {out} ({len(cells)} cells, {time.time() - t00:.0f}s)")


def report() -> None:
    cells = {}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "cells.shard*.pkl"))):
        cells.update(pickle.load(open(f, "rb")))
    base = {}
    for f in sorted(glob.glob("results/graphgp_v2/b_feat.shard*.pkl")):
        base.update(pickle.load(open(f, "rb"))["cells"])
    common = sorted(set(cells) & set(base))
    print(f"{len(cells)} bump cells; paired against published b_feat "
          f"on {len(common)}")

    def scores(cell):
        yt, pr, sd, ch = cell
        z = (yt - pr) / sd
        return (float(np.sqrt(np.mean((yt - pr) ** 2))),
                float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2)
                              + 0.5 * z ** 2)))

    d_r, d_n, amps = [], [], []
    for k in common:
        r1, n1 = scores(cells[k]["cell"])
        r0, n0 = scores(base[k])
        d_r.append(r1 - r0)
        d_n.append(n1 - n0)
        amps.append(cells[k]["bump"]["a"])
    rng = np.random.default_rng(0)
    for name, d in (("RMSE", np.array(d_r)), ("NLL", np.array(d_n))):
        idx = rng.integers(0, d.size, size=(2000, d.size))
        m = d[idx].mean(1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"bump - additive {name}: dmean {d.mean():+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]{star} (median {np.median(d):+.4f})")
    amps = np.array(amps)
    print(f"fitted bump amplitude a: median {np.median(amps):.4f}, "
          f"a < 0.01 (off) on {np.mean(amps < 0.01):.0%} of cells, "
          f"q90 {np.quantile(amps, 0.9):.3f}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        sys.exit(__doc__)
    if sys.argv[1] == "report":
        report()
        return
    shard_k, shard_n = 0, 1
    if "--shard" in sys.argv:
        shard_k, shard_n = map(int, sys.argv[
            sys.argv.index("--shard") + 1].split("/"))
    run(shard_k, shard_n)


if __name__ == "__main__":
    main()
