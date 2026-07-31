#!/usr/bin/env python
"""Posterior cross-covariance between prior components: redundancy (DEV).

Components independent a priori become correlated a posteriori — negatively
where they can explain the same variation (explaining-away).  For components
c != c' the exact posterior cross-covariance is

    Cov(f_c, f_c' | y) = - K_ao^(c) (K_oo + noise)^{-1} (K_ao^(c'))^T

and its diagonal, normalized by the component posterior variances, gives a
per-(note, channel) correlation in [-1, 1].  The graph x LM correlation is a
direct measure of how much the two carry the same information — the quantity
behind the "harmonic edges redundant once the music model is in the kernel"
question.

Dev-only diagnostic (30 dev pieces, anchor-mask seed 0, config b_featlm);
uses gp._blocks(only=...) — promote to a tested gp.py method before any
thesis-facing use.

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/eval_component_redundancy.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import INPUTS, piece_setup  # noqa: E402

OUT = "results/component_redundancy_dev.md"
SEED = 0
COMPS = [("graph", "graph"), (("feat", 0), "score features"),
         (("feat", 1), "LM features")]
PAIRS = [(0, 2), (1, 2), (0, 1)]   # (graph, LM), (feat, LM), (graph, feat)
CHANNELS = ["tau", "log r", "v"]


def piece_redundancy(p, mask, emb):
    """(n_pairs, 3) mean held-out posterior correlation per channel."""
    from score_bundle.gp import MultiOutputGraphGP

    Y = np.asarray(p["y"], dtype=float)
    feats, graph_eig, _, g0 = piece_setup(p, "b_featlm", emb=emb)
    nu, U = graph_eig(g0)
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=feats,
                            n_channels=3)
    floor = 0.05 * np.array([float(np.var(Y[mask, c])) for c in range(3)])
    x_hat, _ = gp.fit(Y, mask, noise_floor=floor, maxiter=200)

    par = gp.unpack(x_hat)
    obs = np.where(mask)[0]
    allidx = np.arange(gp.N)
    n_o = obs.size
    K_oo = gp._blocks(par, obs, obs)
    for c in range(3):
        K_oo[c * n_o:(c + 1) * n_o, c * n_o:(c + 1) * n_o] += \
            par["noise"][c] * np.eye(n_o)
    ao = [gp._blocks(par, allidx, obs, only=sel) for sel, _ in COMPS]
    solves = [np.linalg.solve(K_oo, A.T) for A in ao]          # (k n_o, k N)
    # component posterior variances (diagonal)
    var = []
    for (sel, _), A, S in zip(COMPS, ao, solves):
        prior_diag = np.diag(gp._blocks(par, allidx, allidx, only=sel))
        var.append(np.clip(prior_diag - np.einsum("ij,ji->i", A, S),
                           1e-12, None))
    held = ~mask
    hstack = np.concatenate([held] * 3)          # channel-major over notes
    out = np.zeros((len(PAIRS), 3))
    for j, (a, b) in enumerate(PAIRS):
        cross = -np.einsum("ij,ji->i", ao[a], solves[b])
        corr = cross / np.sqrt(var[a] * var[b])
        corr = corr.reshape(3, gp.N).T[held]     # (n_held, 3)
        out[j] = corr.mean(axis=0)
    return out


def main() -> None:
    from score_bundle.downstream import load_piece_arrays

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    with open(INPUTS, "rb") as fh:
        inp = pickle.load(fh)
    masks, meta = inp["masks"], inp["meta"]
    with open(".cache/kernel_sweep_emb_ma.pkl", "rb") as fh:
        embs = pickle.load(fh)["emb_ma"]

    rows = []
    for pi in range(meta["n_eval_pieces"]):
        rows.append(piece_redundancy(ev[pi], masks[(pi, SEED)],
                                     embs[(pi, SEED)]))
        print(f"  piece {pi} done", flush=True)
    A = np.array(rows)                            # (pieces, pair, channel)
    lines = ["# Posterior component correlations at held-out notes (DEV)",
             "",
             f"{meta['n_eval_pieces']} dev pieces, anchor-mask seed {SEED}, "
             "config b_featlm. Mean per-note posterior correlation between "
             "component pairs (negative = explaining-away / redundancy).",
             "",
             "| pair | " + " | ".join(CHANNELS) + " |",
             "|---|---|---|---|"]
    for j, (a, b) in enumerate(PAIRS):
        label = f"{COMPS[a][1]} x {COMPS[b][1]}"
        cells = [f"{A[:, j, c].mean():+.3f} ± {A[:, j, c].std():.3f}"
                 for c in range(3)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
