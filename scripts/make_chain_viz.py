#!/usr/bin/env python
"""Deck figure: one chain, score + audio in, boundaries + curves out.

Runs the deployment arm (score-nominal tempo-line init, no annotation
access) of curve.fit_chain on the first eligible dev chain and draws the
first genuine Phase-3 output: A the waveform with annotated onsets,
tempo-line starting points, and inferred boundaries; B the inferred
per-note cents curves against the corpus annotation frames (never seen).

Writes docs/thesis/figures/chain_boundary_deck.png (house style).

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_chain_viz.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_phase3_chain_dev import iter_chains
from eval_phase3_curve_dev import load_hypers

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
OUT = "docs/thesis/figures/chain_boundary_deck.png"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from score_bundle.phase3.curve import chain_reference, fit_chain

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })

    hypers = load_hypers()
    ch = next(c for _, c in iter_chains())
    x, t, midis = ch["x"], ch["t"], ch["midis"]
    c_nom, nv_nom = chain_reference(x, t, midis, ch["b_nom"],
                                    guard_frac=0.2)
    fit = fit_chain(x, t, midis, ch["b_nom"], hypers=hypers, c_init=c_nom,
                    noise_var=nv_nom, free_width=True)

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(10.6, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.35], "hspace": 0.12})

    # A: waveform + the three boundary sets
    dec = max(1, x.size // 6000)
    axa.plot(t[::dec], x[::dec], color=VERM, lw=0.5, alpha=0.8)
    for j in range(len(midis) - 1):
        axa.axvline(ch["b_annot"][j], color=INK, lw=1.2)
        axa.axvline(ch["b_nom"][j], color=MUTED, lw=1.0, ls="--")
        axa.axvline(fit.b_lo[j], color=BLUE, lw=1.6)
    axa.plot([], [], color=INK, lw=1.2, label="annotated onset")
    axa.plot([], [], color=MUTED, lw=1.0, ls="--",
             label="tempo-line start (score only)")
    axa.plot([], [], color=BLUE, lw=1.6, label="inferred boundary")
    axa.set_ylabel("audio")
    axa.set_yticks([])
    axa.legend(fontsize=7.5, loc="upper right", ncol=3)
    axa.set_title(
        f"A    violin, {len(midis)} consecutive notes: boundaries from the "
        "score and the waveform alone", loc="left")

    # B: inferred curves vs the annotation frames
    cuts = ch["cuts"]
    for k in range(len(midis)):
        tq = np.linspace(cuts[k], cuts[k + 1], 150)
        axb.plot(tq, fit.curve(k, tq), color=BLUE, lw=1.5,
                 label="inferred pitch curve" if k == 0 else None)
        if ch["gts"][k] is not None:
            gt_t, gt_x = ch["gts"][k]
            sel = (gt_t >= cuts[k]) & (gt_t <= cuts[k + 1])
            axb.plot(gt_t[sel], gt_x[sel], ".", ms=2.8, color=VERM,
                     label="annotation frames (never seen)"
                     if k == 0 else None)
        axb.axvline(cuts[k + 1] if k < len(midis) - 1 else np.nan,
                    color="#E5E7EB", lw=0.8)
    for j in range(len(midis) - 1):
        axb.axvline(fit.b_lo[j], color=BLUE, lw=1.0, alpha=0.5)
    axb.set_ylim(-60, 60)
    axb.set_xlabel("time in the segment (s)")
    axb.set_ylabel("pitch deviation (cents)")
    axb.legend(fontsize=7.5, loc="upper right", ncol=2)
    axb.set_title("B    per-note pitch curves from the same joint fit",
                  loc="left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    err = np.abs(fit.b_lo - ch["b_annot"]) * 1000
    ini = np.abs(ch["b_nom"] - ch["b_annot"]) * 1000
    print(f"chain {ch['key']} notes {ch['idx']}: init err "
          + ",".join(f"{e:.0f}" for e in ini) + " ms -> inferred "
          + ",".join(f"{e:.0f}" for e in err) + " ms")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
