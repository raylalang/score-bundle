#!/usr/bin/env python
"""Posterior example figure: what the model actually does on one piece (DEV).

One dev piece under the published 40%-hidden anchor mask (seed 0): the proposed
model's posterior mean and 90% predictive band per channel, with observed notes
as filled dots and the *true values of hidden notes* as open circles — the
reader can see the hidden truth landing inside the calibrated band.

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/make_posterior_example.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import INPUTS, piece_setup  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "docs/thesis/figures/posterior_example_dev.png"
PIECE, SEED = 0, 0
WINDOW = slice(0, 140)          # first 140 notes in score order, for legibility
_Z90 = 1.6448536269514722
INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
NAMES = [r"timing $\tau_i$ (s)", r"articulation $\log r_i$", r"dynamics $v_i$"]


def main() -> None:
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.gp import MultiOutputGraphGP

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    p = ev[PIECE]
    Y = np.asarray(p["y"], dtype=float)
    with open(INPUTS, "rb") as fh:
        mask = pickle.load(fh)["masks"][(PIECE, SEED)]
    with open(".cache/kernel_sweep_emb_ma.pkl", "rb") as fh:
        emb = pickle.load(fh)["emb_ma"][(PIECE, SEED)]

    feats, graph_eig, n_graph, g0 = piece_setup(p, "b_featlm", emb=emb)
    nu, U = graph_eig(g0)
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=feats,
                            n_channels=3)
    floor = 0.05 * np.array([float(np.var(Y[mask, c])) for c in range(3)])
    x_hat, _ = gp.fit(Y, mask, noise_floor=floor, maxiter=200)
    M, S = gp.posterior(Y, mask, x_hat)
    nv = gp.unpack(x_hat)["noise"]

    order = np.argsort(np.asarray(p["onset"], dtype=float), kind="stable")
    idx = order[WINDOW]
    xs = np.arange(len(idx))
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 6.2), dpi=200, sharex=True)
    for c, ax in enumerate(axes):
        m = M[idx, c]
        sd = np.sqrt(S[idx, c] ** 2 + nv[c])
        obs = mask[idx]
        ax.fill_between(xs, m - _Z90 * sd, m + _Z90 * sd, color=BLUE,
                        alpha=0.16, linewidth=0, label="90% predictive band")
        ax.plot(xs, m, color=BLUE, lw=1.4, label="posterior mean")
        ax.plot(xs[obs], Y[idx, c][obs], ".", color=MUTED, ms=4,
                label="observed note")
        ax.plot(xs[~obs], Y[idx, c][~obs], "o", color=VERM, ms=4.5,
                markerfacecolor="white", markeredgewidth=1.3,
                label="hidden note (truth)")
        inside = np.abs(Y[idx, c][~obs] - m[~obs]) <= _Z90 * sd[~obs]
        ax.set_ylabel(NAMES[c], fontsize=9, color=INK)
        ax.annotate(f"hidden notes inside band: {inside.sum()}/{inside.size}",
                    xy=(0.99, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=8, color=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left",
                   bbox_to_anchor=(0.0, 1.22))
    axes[0].set_title(
        f"Proposed model posterior, validation piece {PIECE} "
        f"({p.get('composer', '?')}), 40% hidden, first {len(idx)} notes",
        fontsize=10, color=INK, loc="left", pad=28)
    axes[-1].set_xlabel("note (score order)", fontsize=9, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
