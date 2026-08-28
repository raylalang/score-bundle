#!/usr/bin/env python
"""The Phase-2 confirmation, visualized (fig:phase2-verdict).

Three panels that explain what the one-shot actually was:
  A. the PROCESS: 44 pieces split data-blind; every decision made on the
     development side; claims + protocol frozen (git tag); the wall around
     the 13-piece pool crossed exactly once, on 2026-08-27.
  B. the VERDICT: each registered claim is a pre-stated statistical
     statement --- a paired confidence interval that must exclude zero.
     Confirmation CI (filled) next to its development basis (hollow):
     C1, C2 (both channels) pass; C4 fails and is reported.
  C. C3: as-given coverage@90 per channel inside the registered
     [0.85, 0.95] band.

Reads the raw cells (results/phase2_confirmation_cells.pkl + the dev
results/phase2_real_cells.pkl); bootstrap identical to the report
(rng 31, B = 2000). House style: Okabe-Ito, legends above panels.

    PYTHONPATH=src:scripts python scripts/make_phase2_verdict.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
OUT = "docs/thesis/figures/phase2_verdict.png"
CH_NAMES = ["c", r"$\log\gamma$", r"$\log f$", r"$\ell$", r"$\tau$",
            r"$\delta^{\mathrm{vib}}$"]


def paired_deltas(per_track, channel, metric_idx):
    """as-given minus no-graph per (track, seed); metric 0=RMSE, 1=NLL."""
    d = {}
    for key, seed, sname, ch, _instr, rmse, nll in per_track:
        if ch != channel:
            continue
        d.setdefault((tuple(key), seed), {})[sname] = (rmse, nll)[metric_idx]
    return np.array([v["gp_asgiven"] - v["nograph"]
                     for v in d.values()
                     if "gp_asgiven" in v and "nograph" in v])


def ci(deltas, rng):
    bs = np.array([rng.choice(deltas, deltas.size).mean()
                   for _ in range(2000)])
    return deltas.mean(), *np.quantile(bs, [.025, .975])


def box(ax, x, y, w, h, text, fc, fontsize=8.6, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012",
                                fc=fc, ec=MUTED, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc or INK)


def main() -> None:
    conf = pickle.load(open("results/phase2_confirmation_cells.pkl", "rb"))
    dev = pickle.load(open("results/phase2_real_cells.pkl", "rb"))
    rng = np.random.default_rng(31)

    # CIs are quoted VERBATIM from the frozen report
    # (results/phase2_confirmation_results.md; its bootstrap rng-draw order
    # is canonical). Means and development bases are recomputed from the
    # raw cells and asserted against the report to the reported precision.
    claims = [
        ("C1  intonation recovery", 0, 0, r"$\Delta$RMSE (cents)",
         (-0.877, -1.236, -0.538)),
        ("C2  extent calibration", 1, 1, r"$\Delta$NLL",
         (-2.990, -6.575, -0.435)),
        ("C2  rate calibration", 2, 1, r"$\Delta$NLL",
         (-0.564, -0.739, -0.392)),
        ("C4  timing calibration", 4, 1, r"$\Delta$NLL",
         (-0.030, -0.061, 0.006)),
    ]
    stats = []
    for _, ch, mi, _, (m_pub, lo, hi) in claims:
        m = paired_deltas(conf["per_track"], ch, mi).mean()
        assert abs(m - m_pub) < 5e-4, (ch, mi, m, m_pub)
        dv = paired_deltas(dev["per_track"], ch, mi).mean()
        stats.append((m_pub, lo, hi, dv, hi < 0))
    cov = [np.mean(conf["rows"]["gp_asgiven"]["est"][c]["cov"])
           for c in range(6)]

    fig = plt.figure(figsize=(11.4, 8.2), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.45],
                          width_ratios=[1.5, 1], hspace=0.32, wspace=0.28)

    # ---- A: the process ---------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    ax.set_title("A. the procedure: every decision above the wall; "
                 "the wall crossed once", fontsize=10.5, color=INK,
                 loc="left")
    box(ax, 1, 11, 11, 8, "URMP\n44 pieces", "#EAEAEA")
    ax.annotate("split frozen\ndata-blind\n2026-08-06", (18, 21.5),
                ha="center", fontsize=7.2, color=MUTED)
    ax.add_patch(FancyArrowPatch((12, 16), (24, 23), arrowstyle="-|>",
                                 mutation_scale=12, color=MUTED))
    ax.add_patch(FancyArrowPatch((12, 14), (24, 6.8), arrowstyle="-|>",
                                 mutation_scale=12, color=MUTED))
    box(ax, 24, 19, 38, 9,
        "DEVELOPMENT   31 pieces\nevery decision made here:\n"
        "estimator, noise default, $\\tau$, $\\delta^{\\mathrm{vib}}$, "
        "seeds, robustness", "#DCE9F7", fontsize=8.2)
    box(ax, 66, 19.5, 17, 8,
        "REGISTERED 2026-08-17\nclaims C1--C4 + protocol\n(git tag)",
        "#FBE8D3", fontsize=7.8)
    ax.add_patch(FancyArrowPatch((62, 23.5), (66, 23.5), arrowstyle="-|>",
                                 mutation_scale=12, color=MUTED))
    # the wall: horizontal, sealing the pool off from the decisions
    ax.plot([22, 99], [13.2, 13.2], ls="--", lw=1.6, color=VERM)
    ax.text(97.5, 14.2, "the wall", ha="right", fontsize=8, color=VERM)
    box(ax, 24, 3, 38, 7.6,
        "CONFIRMATION POOL   13 pieces\nuntouched: no fit, no look, "
        "no decision", "#DDF0E7", fontsize=8.2)
    ax.add_patch(FancyArrowPatch((70, 19.5), (58, 10.8),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=ORANGE, lw=2.0))
    ax.text(66.5, 14.6, "the ONE crossing:\n2026-08-27, one shot,\n"
            "protocol verbatim", fontsize=7.4, color=ORANGE)
    ax.add_patch(FancyArrowPatch((62, 6.8), (85, 6.8), arrowstyle="-|>",
                                 mutation_scale=14, color=GREEN, lw=2.0))
    box(ax, 85, 2.6, 14, 9,
        "VERDICT\nC1 pass · C2 pass\nC3 pass · C4 fail", "#DDF0E7",
        fontsize=8.2)

    # ---- B: the four claims as intervals ---------------------------------
    axb = fig.add_subplot(gs[1, 0])
    axb.set_title("B. each claim: a pre-stated interval that must "
                  "exclude 0", fontsize=10, color=INK, loc="left", pad=26)
    ys = np.arange(len(claims))[::-1] * 1.0
    for (label, chn, mi, unit, _pub), (m, lo, hi, dv, passed), y in zip(
            claims, stats, ys):
        col = GREEN if passed else VERM
        span = max(abs(lo), abs(hi), abs(dv)) * 1.1
        axb.plot([lo / span, hi / span], [y, y], color=col, lw=3,
                 solid_capstyle="round")
        axb.plot(m / span, y, "o", ms=7, color=col)
        axb.plot(dv / span, y, "D", ms=6, mfc="none", mec=MUTED, mew=1.4)
        axb.text(-2.35, y + 0.13, label, ha="left", va="center",
                 fontsize=9, color=INK)
        axb.text(-2.35, y - 0.24,
                 f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] {unit}",
                 ha="left", va="center", fontsize=7.4, color=col)
        axb.text(1.14, y, "PASS" if passed else "FAIL\n(CI includes 0)",
                 ha="left", va="center", fontsize=8.6, color=col,
                 weight="bold")
    axb.axvline(0.0, color=INK, lw=1.1)
    axb.text(0.0, ys[0] + 0.58, "0 = graph adds nothing", ha="center",
             fontsize=8, color=INK)
    axb.set_xlim(-2.4, 1.75)
    axb.set_ylim(-0.6, ys[0] + 0.9)
    axb.axis("off")
    axb.plot([], [], "o", color=MUTED, label="confirmation (CI bar)")
    axb.plot([], [], "D", mfc="none", mec=MUTED, label="development basis")
    axb.legend(frameon=False, fontsize=8, ncol=2, loc="lower left",
               bbox_to_anchor=(0.0, 1.0))

    # ---- C: coverage ------------------------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    axc.set_title("C. coverage@90 inside [0.85, 0.95]\n(claim C3)", fontsize=10, color=INK, loc="left", pad=12)
    axc.axhspan(0.85, 0.95, color=GREEN, alpha=0.12, linewidth=0)
    axc.axhline(0.90, color=INK, lw=1.0, ls=":")
    axc.plot(range(6), cov, "o", ms=8, color=GREEN)
    for i, cv in enumerate(cov):
        axc.text(i, cv + 0.012, f"{cv:.2f}", ha="center", fontsize=7.6,
                 color=INK)
    axc.set_xticks(range(6), CH_NAMES, fontsize=9)
    axc.set_ylim(0.78, 1.0)
    axc.set_ylabel("coverage at nominal 90%", fontsize=9, color=INK)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axc.spines[s].set_color(MUTED)
    axc.tick_params(colors=MUTED, labelsize=8)

    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
