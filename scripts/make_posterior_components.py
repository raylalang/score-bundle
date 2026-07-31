#!/usr/bin/env python
"""Posterior-mean component decomposition: figure + attribution table (DEV).

The GP-first posterior mean splits exactly by prior component (gp.posterior_
components): graph term B (x) K_G, score-feature kernel, LM-embedding kernel.
This script shows the split on one dev piece (same piece/mask as the posterior
example figure) and aggregates an attribution table over the 30 dev pieces x 4
anchor-mask seeds of the published protocol.

Attribution metric per (piece, seed, channel): the covariance share
    share_c = Cov(m_c, m) / Var(m)   over HELD-OUT notes,
which sums to 1 across components exactly (cross terms land where they belong).
Shares can be negative (a component pulling against the total).

Dev-only diagnostic; nothing here touches the confirmation set.

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/make_posterior_components.py
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

OUT_FIG = "docs/thesis/figures/posterior_components_dev.png"
OUT_TAB = "results/posterior_components_dev.md"
PIECE, SEED = 0, 0
WINDOW = slice(0, 140)
INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
NAMES = [r"timing $\tau_i$ (s)", r"articulation $\log r_i$", r"velocity $v_i$"]
COMPS = [("graph", "graph GP", BLUE),
         ("feat_0", "score features", GREEN),
         ("feat_1", "LM features", VERM)]
CHANNELS = ["tau", "log r", "v"]


def fit_components(p, mask, emb):
    from score_bundle.gp import MultiOutputGraphGP

    Y = np.asarray(p["y"], dtype=float)
    feats, graph_eig, _, g0 = piece_setup(p, "b_featlm", emb=emb)
    nu, U = graph_eig(g0)
    gp = MultiOutputGraphGP(nu, U, kernel="additive", features=feats,
                            n_channels=3)
    floor = 0.05 * np.array([float(np.var(Y[mask, c])) for c in range(3)])
    x_hat, _ = gp.fit(Y, mask, noise_floor=floor, maxiter=200)
    return Y, gp.posterior_components(Y, mask, x_hat)


def cov_shares(comps, mask):
    """(n_comp, 3) covariance shares of the posterior mean at held-out notes."""
    held = ~mask
    m = comps["total"][held]                     # (n_held, 3)
    out = np.zeros((len(COMPS), 3))
    for j, (key, _, _) in enumerate(COMPS):
        mc = comps[key][held]
        for c in range(3):
            var = float(np.var(m[:, c]))
            cov = float(np.mean((mc[:, c] - mc[:, c].mean())
                                * (m[:, c] - m[:, c].mean())))
            out[j, c] = cov / var if var > 1e-12 else np.nan
    return out


def main() -> None:
    from score_bundle.downstream import load_piece_arrays

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    with open(INPUTS, "rb") as fh:
        inp = pickle.load(fh)
    masks, meta = inp["masks"], inp["meta"]
    with open(".cache/kernel_sweep_emb_ma.pkl", "rb") as fh:
        embs = pickle.load(fh)["emb_ma"]

    # ---- figure: one piece, the mean split into component curves ----------
    p = ev[PIECE]
    mask = masks[(PIECE, SEED)]
    Y, comps = fit_components(p, mask, embs[(PIECE, SEED)])
    order = np.argsort(np.asarray(p["onset"], dtype=float), kind="stable")
    idx = order[WINDOW]
    xs = np.arange(len(idx))
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 6.2), dpi=200, sharex=True)
    for c, ax in enumerate(axes):
        obs = mask[idx]
        ax.axhline(0.0, color=MUTED, lw=0.6, alpha=0.5)
        # total drawn first so a component that coincides with it stays visible
        ax.plot(xs, comps["total"][idx, c], color=INK, lw=1.8, alpha=0.35,
                label="posterior mean (sum)")
        for key, label, color in COMPS:
            ax.plot(xs, comps[key][idx, c], color=color, lw=1.1, label=label)
        ax.plot(xs[obs], Y[idx, c][obs], ".", color=MUTED, ms=4,
                label="observed note")
        ax.set_ylabel(NAMES[c], fontsize=9, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=5, loc="upper left",
                   bbox_to_anchor=(0.0, 1.24))
    axes[0].set_title(
        f"Posterior mean by prior component, validation piece {PIECE} "
        f"({p.get('composer', '?')}), 40% hidden, first {len(idx)} notes",
        fontsize=10, color=INK, loc="left", pad=28)
    axes[-1].set_xlabel("note (score order)", fontsize=9, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")

    # ---- table: covariance shares over all dev cells ----------------------
    if "--fig-only" in sys.argv:
        return
    n_pieces, n_seeds = meta["n_eval_pieces"], meta["seeds"]
    shares = []
    for pi in range(n_pieces):
        for s in range(n_seeds):
            mk = masks[(pi, s)]
            _, cps = fit_components(ev[pi], mk, embs[(pi, s)])
            shares.append(cov_shares(cps, mk))
            print(f"  piece {pi} seed {s} done", flush=True)
    A = np.array(shares)                          # (cells, comp, channel)
    lines = ["# Posterior-mean covariance shares at held-out notes (DEV)",
             "",
             f"{n_pieces} dev pieces x {n_seeds} anchor-mask seeds, config "
             "b_featlm (additive kernel). Shares sum to 1 per channel.",
             "",
             "| component | " + " | ".join(CHANNELS) + " |",
             "|---|---|---|---|"]
    for j, (_, label, _) in enumerate(COMPS):
        cells = []
        for c in range(3):
            v = A[:, j, c]
            v = v[np.isfinite(v)]
            cells.append(f"{v.mean():+.3f} ± {v.std():.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TAB), exist_ok=True)
    with open(OUT_TAB, "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(f"wrote {OUT_TAB}")


if __name__ == "__main__":
    main()
