#!/usr/bin/env python
"""B-diagonal attribution check (DEV): does severing cross-channel coupling
remove the embeddings' articulation gain?

Fits ``b_featlm`` with ``fit(b_diagonal=True)`` on the published anchor cells
(30 dev pieces x 4 seeds) and compares per channel against the full-B cells
(results/graphgp_masksweep/obs0.60_anchor/b_featlm). Hypotheses for the
log r gain from adding embeddings (0.615 -> 0.601 in the dev ladder) given
that embeddings carry no within-piece log r signal at the median piece:

  (a) information flows v -> log r through the coregionalization B: then the
      diagonal-B fit loses the log r gain;
  (b) it is carried by the minority of pieces with nonzero log r embedding
      scales: then the gain survives B-diagonal.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_bdiag.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import INPUTS, bootstrap_ci, piece_setup  # noqa: E402
from report_theoryfeat_rates import load_cells  # noqa: E402

OUT = "results/graphgp_bdiag/b_featlm_bdiag.pkl"
BASE_DIR = "results/graphgp_masksweep"
BASE_TAG = "obs0.60_anchor"
CHANNELS = ["tau", "log r", "v"]


def run_cells() -> dict:
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.gp import MultiOutputGraphGP

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    with open(INPUTS, "rb") as fh:
        inp = pickle.load(fh)
    masks, meta = inp["masks"], inp["meta"]
    with open(".cache/kernel_sweep_emb_ma.pkl", "rb") as fh:
        embs = pickle.load(fh)["emb_ma"]
    cells = {}
    for pi in range(meta["n_eval_pieces"]):
        p = ev[pi]
        Y = np.asarray(p["y"], dtype=float)
        for s in range(meta["seeds"]):
            mask = masks[(pi, s)]
            feats, graph_eig, _, g0 = piece_setup(p, "b_featlm",
                                                  emb=embs[(pi, s)])
            nu, U = graph_eig(g0)
            gp = MultiOutputGraphGP(nu, U, kernel="additive", features=feats,
                                    n_channels=3)
            floor = 0.05 * np.array([float(np.var(Y[mask, c]))
                                     for c in range(3)])
            x_hat, _ = gp.fit(Y, mask, noise_floor=floor, maxiter=200,
                              b_diagonal=True)
            M, S = gp.posterior(Y, mask, x_hat)
            nv = gp.unpack(x_hat)["noise"]
            held = ~mask
            yt = Y[held].T.ravel()
            pr = M[held].T.ravel()
            sd = np.sqrt(S[held] ** 2 + nv[None, :]).T.ravel()
            ch = np.repeat(np.arange(3), int(held.sum()))
            cells[("bdiag", pi, s)] = (yt, pr, sd, ch)
        print(f"  piece {pi} done", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump({"cells": cells, "meta": {"config": "b_featlm b_diagonal",
                                              "inputs": INPUTS}}, fh)
    return cells


def per_piece_channel(cells, c, field):
    out = {}
    for (_, pi, s), (yt, pr, sd, ch) in cells.items():
        m = ch == c
        r = yt[m] - pr[m]
        if field == "rmse":
            v = float(np.sqrt(np.mean(r ** 2)))
        else:
            v = float(np.mean(0.5 * np.log(2 * np.pi * sd[m] ** 2)
                              + r ** 2 / (2 * sd[m] ** 2)))
        out.setdefault(pi, []).append(v)
    return {pi: float(np.mean(v)) for pi, v in out.items()}


def main() -> None:
    if os.path.exists(OUT) and "--report-only" in sys.argv:
        with open(OUT, "rb") as fh:
            cells = pickle.load(fh)["cells"]
    else:
        cells = run_cells()
    base = load_cells(BASE_DIR, BASE_TAG, "b_featlm")
    rng = np.random.default_rng(23)
    print("\n== b_featlm: full B vs diagonal B, per channel "
          "(paired, 30 dev pieces, anchor masks) ==")
    print(f"{'channel':>8} {'full-B':>16} {'diag-B':>16} "
          f"{'d(full-diag) RMSE':>26} {'dNLL':>26}")
    for c in range(3):
        rows = []
        for f in ("rmse", "nll"):
            a = per_piece_channel(base, c, f)
            b = per_piece_channel(cells, c, f)
            common = sorted(set(a) & set(b))
            d = np.array([a[pi] - b[pi] for pi in common])
            mu, lo, hi = bootstrap_ci(d, B=2000, rng=rng)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            rows.append((float(np.mean([a[pi] for pi in common])),
                         float(np.mean([b[pi] for pi in common])),
                         f"{mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}"))
        print(f"{CHANNELS[c]:>8} {rows[0][0]:16.4f} {rows[0][1]:16.4f} "
              f"{rows[0][2]:>26} {rows[1][2]:>26}")


if __name__ == "__main__":
    main()
