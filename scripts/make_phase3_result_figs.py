#!/usr/bin/env python
"""Deck figures for the two newest quantitative waveform results.

A) curve_scale_deck.png -- the 376-note batch (results/phase3_curve_dev.md)
   as a per-family dumbbell: median GT-frame error of the curve inferred
   from the waveform (fitted-prior arm) vs the tracker's own frames.
B) boundary_deploy_deck.png -- the deployment arm of the chain study
   (results/phase3_chain_dev.md) as a paired scatter: per boundary, the
   score-only (tempo-line) starting error vs the inferred error.

Reads the result cells directly (results/phase3_cells/), so every number
is the record's own. House style.

    PYTHONPATH=src:scripts python scripts/make_phase3_result_figs.py
"""
from __future__ import annotations

import glob
import pickle

import numpy as np

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
FAMS = (("strings", ("vn", "va", "vc", "db")),
        ("winds", ("fl", "cl", "ob", "sax", "bn")),
        ("brass", ("tpt", "hn", "tbn", "tba")))
OUT_A = "docs/thesis/figures/curve_scale_deck.png"
OUT_B = "docs/thesis/figures/boundary_deploy_deck.png"


def style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })


def fig_curve_scale(plt):
    rows = []
    for f in sorted(glob.glob("results/phase3_cells/curve.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    groups = [("all notes", rows)]
    for fam, mem in FAMS:
        groups.append((fam, [r for r in rows if r["instr"] in mem]))
    # order: all, winds, brass, strings (the story's order: wins first)
    groups = [groups[0], groups[2], groups[3], groups[1]]

    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    for y, (name, sub) in enumerate(reversed(groups)):
        wave = float(np.median([r["fitted"]["rmse"] for r in sub]))
        pyin = float(np.median([r["pyin_rmse"] for r in sub
                                if np.isfinite(r["pyin_rmse"])]))
        ax.plot([wave, pyin], [y, y], color=MUTED, lw=1.2, zorder=1)
        ax.plot(wave, y, "o", ms=9, color=BLUE, zorder=3)
        ax.plot(pyin, y, "o", ms=9, color=GREEN, zorder=3)
        off = 0.32 if abs(wave - pyin) > 0.6 else 0.40
        ax.annotate(f"{wave:.2f}", (wave, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=BLUE, fontweight="bold")
        ax.annotate(f"{pyin:.2f}", (pyin, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=GREEN)
        ax.annotate(f"{name} (n={len(sub)})", (0.12, y),
                    ha="left", va="center", fontsize=9, color=INK)
    ax.plot([], [], "o", color=BLUE, label="curve from the waveform")
    ax.plot([], [], "o", color=GREEN, label="tracker frames (pYIN)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04), fontsize=8.5,
              ncol=1)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 4.1)
    ax.set_xlim(0, 6.2)
    ax.set_xlabel("median error vs annotation frames (cents)")
    fig.tight_layout()
    fig.savefig(OUT_A, bbox_inches="tight")
    print("wrote", OUT_A)


def fig_boundary_deploy(plt):
    rows = []
    for f in sorted(glob.glob("results/phase3_cells/chain.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    init, fin = [], []
    for r in rows:
        for j, b in enumerate(r["nominal"]["b"]):
            if abs(b["interval"]) < 0.5:
                continue
            init.append(r["init_err_nom_ms"][j])
            fin.append(b["edge_ms"])
    init, fin = np.asarray(init), np.asarray(fin)
    mi, mf = np.median(init), np.median(fin)

    fig, ax = plt.subplots(figsize=(4.3, 3.9))
    lim = 130.0
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color=MUTED, zorder=1)
    ax.annotate("no change", (lim * 0.78, lim * 0.84), fontsize=8,
                color=MUTED, rotation=45)
    ax.scatter(np.minimum(init, lim), np.minimum(fin, lim), s=16,
               color=BLUE, alpha=0.55, linewidths=0, zorder=3)
    ax.axvline(mi, color=MUTED, lw=1.0, ls=":")
    ax.axhline(mf, color=BLUE, lw=1.0, ls=":")
    ax.annotate(f"median start {mi:.0f} ms", (mi, lim * 0.97),
                rotation=90, va="top", ha="right", fontsize=8, color=MUTED)
    ax.annotate(f"median inferred {mf:.0f} ms", (lim * 0.98, mf),
                va="bottom", ha="right", fontsize=8, color=BLUE)
    n_out = int(np.sum((init > lim) | (fin > lim)))
    if n_out:
        ax.annotate(f"{n_out} boundaries clipped at {lim:.0f} ms",
                    (lim * 0.98, 4), ha="right", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("starting error: score-only tempo line (ms)")
    ax.set_ylabel("inferred boundary error (ms)")
    frac = float(np.mean(fin < init))
    ax.set_title(f"one dot per boundary (n={init.size}); below the "
                 f"diagonal = improved ({frac:.0%})", loc="left",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_B, bbox_inches="tight")
    print(f"medians {mi:.1f} -> {mf:.1f} ms; improved on {frac:.0%}")
    print("wrote", OUT_B)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    fig_curve_scale(plt)
    fig_boundary_deploy(plt)


if __name__ == "__main__":
    main()
