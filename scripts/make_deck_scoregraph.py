#!/usr/bin/env python
"""Slide-6 figure: the score graph on a real excerpt.

Nodes are the notes of the opening of validation piece 0, placed at
(score time, pitch). Edges: chord (same onset, green), voice-leading
(stepwise within 2 beats, vermillion), and time/pitch proximity (blue,
opacity proportional to the Gaussian weight W_ij).

    PYTHONPATH=src python scripts/make_deck_scoregraph.py
"""
from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "docs/thesis/figures/deck_scoregraph.png"
N_NOTES = 26
INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"


def main() -> None:
    from score_bundle.downstream import load_piece_arrays, piece_score
    from score_bundle.graph import build_adjacency
    from score_bundle.score import Score

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    p = ev[0]
    order = np.argsort(np.asarray(p["onset"], dtype=float), kind="stable")[:N_NOTES]
    b = np.asarray(p["onset"], dtype=float)[order]
    pit = np.asarray(p["pitch"], dtype=float)[order]
    d = np.asarray(p["duration"], dtype=float)[order]
    b = b - b.min()
    sub = Score.from_arrays(pit, b, d, np.zeros(len(order), dtype=int))
    W = build_adjacency(sub)

    fig, ax = plt.subplots(figsize=(10.2, 3.6), dpi=200)
    n = len(order)
    for i in range(n):
        for j in range(i + 1, n):
            same = abs(b[i] - b[j]) < 1e-9
            step = (not same) and abs(b[i] - b[j]) <= 2.0 \
                and 1 <= abs(pit[i] - pit[j]) <= 2
            if same:
                ax.plot([b[i], b[j]], [pit[i], pit[j]], color=GREEN, lw=1.8,
                        zorder=1)
            elif step:
                ax.plot([b[i], b[j]], [pit[i], pit[j]], color=VERM, lw=1.4,
                        zorder=1, alpha=0.9)
            elif W[i, j] > 0.15:
                ax.plot([b[i], b[j]], [pit[i], pit[j]], color=BLUE,
                        lw=1.0, alpha=min(0.85, float(W[i, j])), zorder=0)
    ax.scatter(b, pit, s=110, facecolor="white", edgecolor=INK, lw=1.2,
               zorder=2)
    ax.set_xlabel("score time (beats)", fontsize=9, color=INK)
    ax.set_ylabel("pitch", fontsize=9, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    handles = [Line2D([], [], color=BLUE, lw=1.6,
                      label=r"time/pitch proximity (weight $W_{ij}$)"),
               Line2D([], [], color=GREEN, lw=1.8, label="same chord"),
               Line2D([], [], color=VERM, lw=1.6, label="voice-leading step")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left",
              ncol=3, bbox_to_anchor=(0.0, 1.14))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
