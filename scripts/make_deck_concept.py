#!/usr/bin/env python
"""Slide-2 concept figure: a real excerpt as a piano-roll, written vs played,
with the three per-note deviation channels underneath.

Data: the opening of validation piece 0 (real ASAP score + performance
deviations). Written notes are outlined; played notes are filled, shifted by
tau_i and stretched by r_i. Three aligned strips show tau, log r, v per note.

    PYTHONPATH=src python scripts/make_deck_concept.py
"""
from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "docs/thesis/figures/deck_concept.png"
MAX_BEAT = 18.0        # the first engraved system = bars 1-6 in 3/4
INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"


def main() -> None:
    from score_bundle.downstream import load_piece_arrays
    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    p = ev[0]
    onset_all = np.asarray(p["onset"], dtype=float)
    o0 = onset_all.min()
    keep = np.where(onset_all - o0 < MAX_BEAT)[0]
    order = keep[np.argsort(onset_all[keep], kind="stable")]
    b = onset_all[order]
    d = np.asarray(p["duration"], dtype=float)[order]
    pit = np.asarray(p["pitch"], dtype=float)[order]
    y = np.asarray(p["y"], dtype=float)[order]          # tau, log r, v
    b = b - b.min()

    fig = plt.figure(figsize=(10.4, 6.6), dpi=200)
    gs = fig.add_gridspec(5, 1, height_ratios=[2.1, 2.1, 0.62, 0.62, 0.62],
                          hspace=0.42)

    axn = fig.add_subplot(gs[0])
    notation = plt.imread("docs/thesis/figures/deck_notation.png")
    axn.imshow(notation)
    axn.axis("off")
    axn.set_title("the score (Schumann, Kreisleriana No. 2, first system)",
                  fontsize=10, color=INK, loc="left")

    ax = fig.add_subplot(gs[1])
    for i in range(len(order)):
        ax.add_patch(Rectangle((b[i], pit[i] - 0.38), d[i], 0.76, fill=False,
                               edgecolor=MUTED, lw=1.1))
        played_on = b[i] + y[i, 0]
        played_dur = d[i] * np.exp(y[i, 1])
        ax.add_patch(Rectangle((played_on, pit[i] - 0.30), played_dur, 0.60,
                               facecolor=BLUE, edgecolor="none", alpha=0.75))
    ax.set_xlim(-0.3, (b + d).max() + 0.4)
    ax.set_ylim(pit.min() - 1.5, pit.max() + 1.5)
    ax.set_ylabel("pitch", fontsize=9, color=INK)
    ax.set_title("the same bars as a piano-roll: written (outline) vs played (filled)",
                 fontsize=10, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.set_xticklabels([])

    names = [(r"timing $\tau_i$ (s)", VERM), (r"articulation $\log r_i$", GREEN),
             (r"dynamics $v_i$", AMBER)]
    for c in range(3):
        axc = fig.add_subplot(gs[c + 2], sharex=ax)
        axc.axhline(0.0, color=MUTED, lw=0.7)
        axc.vlines(b, 0, y[:, c], color=names[c][1], lw=1.6)
        axc.plot(b, y[:, c], "o", ms=3.5, color=names[c][1])
        axc.set_ylabel(names[c][0], fontsize=8, color=INK, rotation=0,
                       ha="right", va="center", labelpad=6)
        axc.tick_params(colors=MUTED, labelsize=7)
        for side in ("top", "right"):
            axc.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axc.spines[side].set_color(MUTED)
        if c < 2:
            axc.set_xticklabels([])
    axc.set_xlabel("score time (beats)", fontsize=9, color=INK)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
