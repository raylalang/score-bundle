#!/usr/bin/env python
"""Render the three meeting-digest figures into docs/figures/.

Reads numbers from the eval logs (never hard-code a metric that exists in a log):

  A  digest_headline.png   dumbbell, RMSE graph off -> on per prior mean
                           (logs/feature_baseline_l2_10.log pooled table)
  B  digest_channels.png   diverging bars, LM vs feat per channel, mean-only RMSE
                           (same log, per-channel tables)
  C  digest_collapse.png   strip of 120 per-cell tau RMSE, unguarded vs guarded
                           (logs/diag_tau_s2x3.log, logs/diag_tau_s2x3_guarded.log)

Palette: dataviz reference instance (validated); light mode, matplotlib.
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; SURFACE = "#fcfcfb"
BLUE = "#2a78d6"; BLUE_LT = "#86b6ef"; BLUE_DK = "#1c5cab"
AQUA = "#1baf7a"; RED = "#d03b3b"; DIV_RED = "#e34948"
OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 150,
})


def _style(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def read_pooled(path):
    """{(mean, on|off): (rmse, nll)} from the pooled table of the grid eval log."""
    txt = open(path).read()
    block = txt.split("Pooled (")[1].split("\n\n")[0]
    out = {}
    for line in block.splitlines():
        m = re.match(r"(\S+)\s+(on|off)\s+([-\d.]+)\s+([-\d.]+)", line.strip())
        if m:
            out[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return out


def read_channel(path, ch):
    txt = open(path).read()
    block = txt.split(f"[{ch}]")[1].split("\n\n")[0]
    out = {}
    for line in block.splitlines():
        m = re.match(r"(\S+)\s+(on|off)\s+([-\d.]+)", line.strip())
        if m:
            out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def read_cells(path):
    """[(seed, piece, tau_rmse)] from a diag log's per-cell lines."""
    cells = []
    for line in open(path):
        m = re.match(r"\s+s(\d) p\s*(\d+) tauRMSE\s+([\d.]+)", line)
        if m:
            cells.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    return cells


def fig_headline(grid_log):
    pooled = read_pooled(grid_log)
    means = ["zero", "feat", "LM", "feat+LM"]
    labels = {"zero": "no prior mean", "feat": "score features",
              "LM": "network (LM)", "feat+LM": "features + network"}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ys = np.arange(len(means))[::-1]
    for y, mn in zip(ys, means):
        off, on = pooled[(mn, "off")][0], pooled[(mn, "on")][0]
        ax.plot([off, on], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([off], [y], s=70, color=BLUE_LT, zorder=2)
        ax.scatter([on], [y], s=70, color=BLUE_DK, zorder=3)
        ax.annotate(f"{on:.3f}", (on, y), textcoords="offset points",
                    xytext=(-6, -14), color=INK2, fontsize=9, ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[m] for m in means], color=INK)
    ax.set_xlabel("held-out error (pooled RMSE, lower is better)")
    ax.scatter([], [], s=70, color=BLUE_LT, label="graph off")
    ax.scatter([], [], s=70, color=BLUE_DK, label="graph on")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("The graph helps every prior mean; features + network is best",
                 color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/digest_headline.png")
    plt.close(fig)


def fig_channels(grid_log):
    chans = ["tau", "log r", "v"]
    names = {"tau": "timing (τ)", "log r": "articulation (log r)", "v": "loudness (v)"}
    rel = []
    for ch in chans:
        t = read_channel(grid_log, ch)
        rel.append(100.0 * (t[("LM", "off")] - t[("feat", "off")]) / t[("feat", "off")])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ys = np.arange(len(chans))[::-1]
    colors = [BLUE if v < 0 else DIV_RED for v in rel]
    ax.barh(ys, rel, height=0.45, color=colors)
    ax.axvline(0, color=BASE, lw=1)
    for y, v in zip(ys, rel):
        ax.annotate(f"{v:+.0f}%", (v, y), textcoords="offset points",
                    xytext=(6 if v >= 0 else -6, -3),
                    ha="left" if v >= 0 else "right", color=INK2, fontsize=10)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[c] for c in chans], color=INK)
    ax.set_xlabel("network error vs feature error (mean-only, % — negative = network better)")
    lim = max(abs(v) for v in rel) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_title("Where the network earns its place: loudness",
                 color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/digest_channels.png")
    plt.close(fig)


def fig_collapse(unguarded_log, guarded_log):
    rows = [("guard off", read_cells(unguarded_log)),
            ("guard on", read_cells(guarded_log))]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    rng = np.random.default_rng(0)
    for yi, (label, cells) in enumerate(rows):
        vals = np.array([c[2] for c in cells])
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        normal = vals <= 1.0
        ax.scatter(vals[normal], np.full(normal.sum(), yi) + jitter[normal],
                   s=22, color=BLUE, alpha=0.55, edgecolors="none")
        for v, j in zip(vals[~normal], jitter[~normal]):
            ax.scatter([v], [yi + j], s=60, color=RED, zorder=3)
            ax.annotate(f"one collapsed fit: {v:.1f}", (v, yi + j),
                        textcoords="offset points", xytext=(-8, 10),
                        ha="right", color=RED, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("per-evaluation-cell timing error (τ RMSE), 120 cells each")
    ax.set_title("The 1-in-120 fit collapse, and the guard that removes it",
                 color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/digest_collapse.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    grid_log = "logs/feature_baseline_l2_10.log"
    if "--collapse-only" not in sys.argv:
        fig_headline(grid_log)
        fig_channels(grid_log)
        print("headline + channels written")
    fig_collapse("logs/diag_tau_s2x3.log", "logs/diag_tau_s2x3_guarded.log")
    print("collapse written")
