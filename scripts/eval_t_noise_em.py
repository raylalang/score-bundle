#!/usr/bin/env python
"""Inference-level Student-t timing noise via scale-mixture EM (DEV ONLY).

The deploy-time study (results/tail_predictive_dev.md) rescued the log
score without touching the fit.  This study measures the inference-level
version: Student-t observation noise on the tau channel, implemented
through its Gaussian scale-mixture representation -- per-note latent
variance, EM reweighting

    w_i = (nu + 1) / (nu + z_i^2),   z_i = (y_i - m_i) / sd_noise_i,

over OBSERVED tau notes, with all hyperparameters refit under the
weighted noise (two EM rounds; nu = 5, the deploy-time study's
recommendation).  Unlike the rescoring, this can change the MEANS:
outliers among observed notes currently distort the Gaussian fit, and the
weights down-weight them.  Config b_featlm on the published dev masks
(30 pieces x 4 anchor seeds); the plain Gaussian fit recomputed in-script
is the paired baseline.  Held-out scoring under both the Gaussian and the
variance-matched t5 predictive, so fit effects and scoring effects stay
separable.  Exploratory, no claims; nothing published is touched.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_t_noise_em.py run [--shard K/N]
    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_t_noise_em.py report
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

from eval_graphgp import piece_setup  # noqa: E402

INPUTS = ".cache/kernel_sweep_inputs.pkl"
EMB_DUMP = ".cache/kernel_sweep_emb_ma.pkl"
ARRAYS = ".cache/asap_arrays_named50.pkl"
OUT_DIR = "results/t_noise_em"
NU = 5.0
TAU = 0
Z90 = 1.6448536269514722
N_EVAL = 30


def fit_once(gp, Y, mask2d, floor, x0=None):
    x_hat, _ = gp.fit(Y, mask2d, noise_floor=floor, maxiter=200, x0=x0)
    M, S = gp.posterior(Y, mask2d, x_hat)
    nv = gp.unpack(x_hat)["noise"]
    return x_hat, M, S, nv


def run(shard_k: int, shard_n: int) -> None:
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.gp import MultiOutputGraphGP

    with open(INPUTS, "rb") as fh:
        inputs = pickle.load(fh)
    masks, imeta = inputs["masks"], inputs["meta"]
    with open(EMB_DUMP, "rb") as fh:
        emb_dump = pickle.load(fh)["emb_ma"]
    head, ev, _ = load_piece_arrays(ARRAYS)
    ev = ev[:N_EVAL]                          # DEV pieces only
    seeds = imeta["seeds"]
    recs = []
    for s in range(seeds):
        for pi, p in enumerate(ev):
            if (s * len(ev) + pi) % shard_n != shard_k:
                continue
            t0 = time.time()
            Y = np.asarray(p["y"], dtype=float)
            mask = masks[(pi, s)]             # per-note (1-D) mask
            held = ~mask
            n = len(Y)
            mask2d = np.repeat(mask[:, None], 3, axis=1)
            emb = emb_dump[(pi, s)]
            feats, graph_eig, _, g0 = piece_setup(p, "b_featlm", emb=emb)
            nu_e, U = graph_eig(g0)
            floor = 0.05 * np.array([float(np.var(Y[mask, c]))
                                     for c in range(3)])

            gp = MultiOutputGraphGP(nu_e, U, kernel="additive",
                                    features=feats, n_channels=3)
            x_g, M_g, S_g, nv_g = fit_once(gp, Y, mask2d, floor)

            # EM: weights from observed-tau standardized residuals, refit
            scale = np.ones((n, 3))
            x_warm = x_g
            for _ in range(2):
                gp_t = MultiOutputGraphGP(nu_e, U, kernel="additive",
                                          features=feats, n_channels=3)
                gp_t.noise_scale = scale
                x_t, M_t, S_t, nv_t = fit_once(gp_t, Y, mask2d, floor,
                                               x0=x_warm)
                r = Y[mask, TAU] - M_t[mask, TAU]
                z2 = r ** 2 / np.maximum(nv_t[TAU] * scale[mask, TAU], 1e-12)
                w = (NU + 1.0) / (NU + z2)
                scale = np.ones((n, 3))
                scale[mask, TAU] = 1.0 / np.maximum(w, 1e-3)
                x_warm = x_t

            def held_scores(M, S, nv, sc):
                out = {}
                for c in range(3):
                    yt = Y[held, c]
                    m = M[held, c]
                    sd = np.sqrt(S[held, c] ** 2 + nv[c])
                    z = (yt - m) / sd
                    g_nll = float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2)
                                          + 0.5 * z ** 2))
                    st = sd * np.sqrt((NU - 2) / NU)
                    zt = (yt - m) / st
                    t_nll = float(np.mean(
                        -_t_logpdf(zt, NU) + np.log(st)))
                    out[c] = {"rmse": float(np.sqrt(np.mean((yt - m) ** 2))),
                              "gnll": g_nll, "tnll": t_nll,
                              "cov": float(np.mean(np.abs(z) <= Z90))}
                return out

            rec = {"pi": pi, "seed": s,
                   "gauss": held_scores(M_g, S_g, nv_g, None),
                   "tem": held_scores(M_t, S_t, nv_t, None),
                   "w_min": float((1.0 / scale[mask, TAU]).min()),
                   "w_frac_low": float(np.mean(
                       1.0 / scale[mask, TAU] < 0.5))}
            recs.append(rec)
            print(f"piece {pi} seed {s}: dRMSE_tau "
                  f"{rec['tem'][TAU]['rmse'] - rec['gauss'][TAU]['rmse']:+.4f} "
                  f"w_min {rec['w_min']:.2f} downwt "
                  f"{rec['w_frac_low']:.1%} ({time.time() - t0:.0f}s)",
                  flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"cells.shard{shard_k}of{shard_n}.pkl")
    pickle.dump(recs, open(out, "wb"))
    print(f"wrote {out} ({len(recs)} cells)")


def _t_logpdf(z, nu):
    from math import lgamma
    c = (lgamma((nu + 1) / 2) - lgamma(nu / 2)
         - 0.5 * np.log(nu * np.pi))
    return c - (nu + 1) / 2 * np.log1p(z ** 2 / nu)


def report() -> None:
    recs = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "cells.shard*.pkl"))):
        recs += pickle.load(open(f, "rb"))
    print(f"{len(recs)} cells; observed-tau notes down-weighted (w<0.5): "
          f"mean {np.mean([r['w_frac_low'] for r in recs]):.1%}, "
          f"max {np.max([r['w_frac_low'] for r in recs]):.1%}")
    rng = np.random.default_rng(0)
    for c, name in ((0, "tau"), (1, "log r"), (2, "v")):
        for met in ("rmse", "gnll", "tnll", "cov"):
            a = np.array([r["tem"][c][met] for r in recs])
            b = np.array([r["gauss"][c][met] for r in recs])
            d = a - b
            idx = rng.integers(0, d.size, size=(2000, d.size))
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            if c == 0 or met == "rmse":
                print(f"{name:6s} {met:5s}: t-EM {a.mean():7.4f} vs gauss "
                      f"{b.mean():7.4f}  d {d.mean():+7.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]{star}")


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
