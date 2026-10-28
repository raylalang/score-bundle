#!/usr/bin/env python
"""Figures for the end-goal proposal deck (docs/slides/proposal_endgoal.tex).

Two concept figures. One of them shows MOCK numbers, generated here from
hand-set values, and carry an "illustrative, not measured" banner. Nothing
in this script reads an evaluation record.

  proposal_motif_graph.png   the motif edge family on a score excerpt
  proposal_mock_results.png  what the motif study's result would look like

    PYTHONPATH=src python scripts/make_proposal_figs.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"
OUT = "docs/thesis/figures/"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 200,
})


def banner(fig, text="illustrative mock result, not measured"):
    fig.text(0.5, 0.995, text.upper(), ha="center", va="top", fontsize=9,
             color=VERM, fontweight="bold")


def motif_graph():
    """Three occurrences of one motif on a time-pitch plane, time edges in
    grey, motif correspondence edges in blue, one hidden note hollow."""
    rng = np.random.default_rng(3)
    contour = np.array([0, 2, 4, 2, 0, -1])            # the motif, semitones
    starts = [0.0, 8.0, 17.0]                          # beats
    transp = [60, 67, 62]
    durs = np.array([1, 1, 0.5, 0.5, 1, 2])
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    occ = []
    for s, p0 in zip(starts, transp):
        t = s + np.concatenate([[0], np.cumsum(durs[:-1])])
        p = p0 + contour
        occ.append((t, p))
    # filler notes between occurrences (not in any motif)
    fill_t = np.array([6.0, 6.5, 7.0, 14.0, 15.0, 15.5, 16.0, 23.5, 24.5])
    fill_p = np.array([55, 57, 59, 64, 62, 60, 59, 57, 55])
    all_t = np.concatenate([o[0] for o in occ] + [fill_t])
    all_p = np.concatenate([o[1] for o in occ] + [fill_p])
    order = np.argsort(all_t)
    all_t, all_p = all_t[order], all_p[order]
    # time edges: consecutive notes in score time
    for i in range(len(all_t) - 1):
        ax.plot([all_t[i], all_t[i + 1]], [all_p[i], all_p[i + 1]],
                color="#D1D5DB", lw=1.2, zorder=1)
    # motif correspondence edges
    for k in range(len(contour)):
        for a in range(3):
            for b in range(a + 1, 3):
                ta, pa = occ[a][0][k], occ[a][1][k]
                tb, pb = occ[b][0][k], occ[b][1][k]
                arc = FancyArrowPatch((ta, pa), (tb, pb), arrowstyle="-",
                                      connectionstyle="arc3,rad=-0.25",
                                      color=BLUE, lw=0.8, alpha=0.55,
                                      zorder=2)
                ax.add_patch(arc)
    # occurrence boxes
    for (t, p), lab in zip(occ, ["occurrence A", "occurrence B", "occurrence C"]):
        ax.add_patch(Rectangle((t[0] - 0.4, p.min() - 1.2), t[-1] - t[0] + 2.6,
                               p.max() - p.min() + 2.4, fill=False,
                               edgecolor=AMBER, lw=1.4, zorder=0))
        ax.text(t[0] - 0.3, p.max() + 1.6, lab, color=AMBER, fontsize=9)
    ax.scatter(fill_t, fill_p, s=38, color=MUTED, zorder=3)
    for j, (t, p) in enumerate(occ):
        ax.scatter(t, p, s=46, color=INK, zorder=4)
    # hidden note: occurrence C, 3rd note
    th, ph = occ[2][0][2], occ[2][1][2]
    ax.scatter([th], [ph], s=130, facecolor="white", edgecolor=VERM, lw=2.0,
               zorder=5)
    ax.annotate("hidden note:\nits counterparts in A and B\nare observed",
                (th, ph), xytext=(th + 2.2, ph + 4.5), fontsize=9,
                color=VERM, arrowprops=dict(arrowstyle="->", color=VERM))
    ax.plot([], [], color="#D1D5DB", lw=1.6, label="time edge (score neighbours)")
    ax.plot([], [], color=BLUE, lw=1.2, label="motif edge (same position in a repeated motif)")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.set_xlabel("score time (beats)")
    ax.set_ylabel("MIDI pitch")
    ax.set_xlim(-1.5, 27)
    ax.set_ylim(52, 76)
    ax.set_title("The score graph with one more edge family: notes that repeat, connected", loc="left")
    fig.savefig(OUT + "proposal_motif_graph.png", bbox_inches="tight")


def mock_results():
    """Panel A: paired timing log-score difference of each arm against the
    time-only graph (mock). Panel B: timing deviation across three
    occurrences of one motif, with bands (mock)."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 3.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.3})
    banner(fig)
    arms = ["time + pitch\n(the current model)", "time + annotated\nmotif edges",
            "time + predicted\nmotif edges"]
    est = np.array([0.003, -0.041, -0.032])
    lo = np.array([-0.012, -0.068, -0.059])
    hi = np.array([0.018, -0.014, -0.006])
    y = np.arange(len(arms))[::-1]
    axa.axvline(0, color=MUTED, lw=0.8)
    axa.hlines(y, lo, hi, color=BLUE, lw=2.2)
    axa.scatter(est, y, color=BLUE, s=40, zorder=3)
    for yi, e, l, h in zip(y, est, lo, hi):
        axa.text(h + 0.004, yi, f"{e:+.3f} [{l:+.3f}, {h:+.3f}]", va="center",
                 fontsize=8.5, color=INK)
    axa.set_yticks(y)
    axa.set_yticklabels(arms, fontsize=9)
    axa.set_xlim(-0.085, 0.07)
    axa.set_xlabel("paired timing NLL difference vs the time-only graph\n(negative = better calibrated)")
    axa.set_title("A    hidden notes inside motif occurrences (mock)", loc="left")

    tq = np.linspace(0, 6, 120)
    shape = 0.018 * np.sin(np.pi * tq / 6) + 0.006 * np.sin(2 * np.pi * tq / 6)
    rng = np.random.default_rng(7)
    cols = [BLUE, GREEN, AMBER]
    for k, (c, off) in enumerate(zip(cols, [0.0, 0.004, -0.003])):
        m = shape + off + rng.normal(0, 0.0025, tq.size).cumsum() * 0.08
        band = 0.006 + 0.002 * k
        axb.fill_between(tq, m - band, m + band, color=c, alpha=0.15, lw=0)
        axb.plot(tq, m, color=c, lw=1.6, label=f"occurrence {'ABC'[k]}")
    axb.axhline(0, color=MUTED, lw=0.8)
    axb.set_xlabel("position in the motif (beats)")
    axb.set_ylabel("onset deviation from the tempo line (s)")
    axb.legend(loc="upper right", fontsize=9, ncol=3)
    axb.set_title("B    one motif, three returns: the same timing shape, with bands (mock)", loc="left")
    fig.savefig(OUT + "proposal_mock_results.png", bbox_inches="tight")


if __name__ == "__main__":
    motif_graph()
    mock_results()
    print("wrote", OUT + "proposal_{motif_graph,mock_results}.png")
