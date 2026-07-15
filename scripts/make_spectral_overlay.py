#!/usr/bin/env python
"""Fitted spectral filters g(nu; s-hat) per kernel family, overlaid (DEV).

Why the kernel families tie (docs/graphgp_theory_alignment.md §7), shown
directly: for a few development pieces, fit the full proposed-model covariance
(b_feat config, published 40%-hidden anchor mask, seed 0) once per shape family
— graph Matérn α=1 (≡ the additive kernel), α=2, α=3, and the heat kernel —
and plot the evidence-fitted filters over the piece's Laplacian spectrum.
The families nearly reparameterize one another: the evidence picks essentially
the same filter whatever the parametric family.

DEV pieces only; deterministic.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_spectral_overlay.py
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

OUT = "docs/thesis/figures/spectral_overlay_dev.png"
PIECES = [0, 5, 12]
KERNELS = [  # (key, label, colour, linestyle)
    ("additive", r"additive $\equiv$ Matérn $\alpha{=}1$", "#0072B2", "-"),
    ("matern2", r"Matérn $\alpha{=}2$", "#E69F00", "--"),
    ("matern3", r"Matérn $\alpha{=}3$", "#009E73", "-."),
    ("diffusion", "heat / diffusion", "#CC79A7", ":"),
]
INK, MUTED = "#1A1A1A", "#6B7280"


def main() -> None:
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.gp import SHAPE_KERNELS, MultiOutputGraphGP

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    with open(INPUTS, "rb") as fh:
        masks = pickle.load(fh)["masks"]

    fig, axes = plt.subplots(1, len(PIECES), figsize=(10.2, 3.4), dpi=200,
                             sharey=True)
    for ax, pi in zip(axes, PIECES):
        p = ev[pi]
        Y = np.asarray(p["y"], dtype=float)
        mask = masks[(pi, 0)]
        feats, graph_eig, n_graph, g0 = piece_setup(p, "b_feat")
        nu, U = graph_eig(g0)
        grid = np.linspace(0.0, float(nu.max()), 400)
        for key, label, col, ls in KERNELS:
            gp = MultiOutputGraphGP(nu, U, kernel=key, features=feats,
                                    n_channels=Y.shape[1])
            floor = 0.05 * np.array([float(np.var(Y[mask, c]))
                                     for c in range(Y.shape[1])])
            x_hat, info = gp.fit(Y, mask, noise_floor=floor, maxiter=200)
            s_hat = float(gp.unpack(x_hat)["s"])
            ax.plot(grid, SHAPE_KERNELS[key](grid, s_hat), ls, color=col,
                    lw=2, label=label)
            print(f"piece {pi} {key:10s} s_hat={s_hat:.4g}", flush=True)
        ax.plot(nu, np.full_like(nu, -0.04), "|", color=MUTED, ms=4, alpha=0.35)
        ax.set_ylim(-0.08, 1.03)
        ax.set_xlabel(r"graph frequency $\nu$", fontsize=9, color=INK)
        ax.set_title(f"dev piece {pi}  ($N={len(Y)}$)", fontsize=9.5,
                     color=INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel(r"fitted filter $g(\nu;\hat s)$", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].annotate("ticks along the base: the piece's Laplacian eigenvalues",
                     xy=(0.97, 0.90), xycoords="axes fraction",
                     fontsize=7.5, color=MUTED, ha="right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
