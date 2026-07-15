#!/usr/bin/env python
"""Per-channel masking-rate figure (DEV): RMSE vs hidden fraction, one panel
per channel, three systems each, LOO as detached points.

    PYTHONPATH=src:scripts python scripts/make_masksweep_perchannel.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from report_mask_sweep import ROOT, load_cells  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "docs/thesis/figures/masksweep_perchannel_dev.png"
TAGS = [("obs0.50", 50), ("obs0.60_anchor", 40), ("obs0.70", 30),
        ("obs0.80", 20), ("obs0.90", 10)]
SYSTEMS = [("b_featlm", "proposed", "#0072B2", "o", "-"),
           ("b_featlm_nograph", "no graph", "#E69F00", "s", "--"),
           ("b_feat", "no music model", "#009E73", "^", ":")]
CH = [r"timing $\tau$", r"articulation $\log r$", r"dynamics $v$"]
INK, MUTED = "#1A1A1A", "#6B7280"


def perch_rmse(cells, c):
    yt = np.concatenate([v[0] for v in cells.values()])
    pr = np.concatenate([v[1] for v in cells.values()])
    ch = np.concatenate([v[3] for v in cells.values()])
    m = ch == c
    return float(np.sqrt(np.mean((yt[m] - pr[m]) ** 2)))


def loo_perch(c):
    with open(os.path.join(ROOT, "loo.pkl"), "rb") as fh:
        loo = pickle.load(fh)["results"]
    out = {}
    for cfg, r in loo.items():
        yt = np.concatenate([v[0][:, c] for v in r["cells"].values()])
        pr = np.concatenate([v[1][:, c] for v in r["cells"].values()])
        out[cfg] = float(np.sqrt(np.mean((yt - pr) ** 2)))
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), dpi=200)
    xs = [50, 40, 30, 20, 10]
    x_loo = 3.0
    for c, ax in enumerate(axes):
        loo = loo_perch(c)
        for cfg, label, col, mark, ls in SYSTEMS:
            ys = [perch_rmse(load_cells(tag, cfg), c) for tag, _ in TAGS]
            ax.plot(xs, ys, ls, color=col, lw=2.2, marker=mark, ms=6.0,
                    markerfacecolor=col, markeredgecolor="white",
                    markeredgewidth=0.7, label=label)
            ax.plot([x_loo], [loo[cfg]], marker=mark, ms=5.5, color=col,
                    markerfacecolor="white", markeredgecolor=col,
                    markeredgewidth=1.4, linestyle="none")
        ax.axvline(6.5, color=MUTED, lw=0.6, ls=(0, (2, 3)))
        ax.set_xlim(54, 1.2)
        ax.set_xticks(xs + [x_loo])
        ax.set_xticklabels(["50", "40", "30", "20", "10", "LOO"], fontsize=10)
        ax.set_title(CH[c], fontsize=13, color=INK, loc="left")
        ax.set_xlabel("% of notes hidden", fontsize=11, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
    axes[0].set_ylabel("held-out RMSE", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
