#!/usr/bin/env python
"""Contrast figure + disagreement statistics for the component decomposition (DEV).

Two exhibits for "when", rather than "how much on average":
  1. posterior_components_contrast_dev.png — the SAME channel (velocity) on
     two development pieces at the dominance extremes: one graph-dominated,
     one embedding-dominated. Dominance is a property of the piece, assigned
     by the per-piece evidence. (Timing is unsuitable for this exhibit: its
     graph-dominated pieces are the steady ones whose mean shrinks to noise.)
  2. Disagreement statistics — per channel and component pair, the fraction of
     held-out notes where two components pull in OPPOSITE directions with both
     pulls substantial (|m_c| above a quarter of the channel's posterior-mean
     spread): the per-note face of the negative posterior cross-correlations.

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/make_component_contrast.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import INPUTS  # noqa: E402
from make_posterior_components import (COMPS, INK, MUTED, NAMES,  # noqa: E402
                                       fit_components)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_FIG = "docs/thesis/figures/posterior_components_contrast_dev.png"
SEED = 0
CHAN = 2                       # velocity: the channel where both extremes are active
WINDOW = slice(0, 60)
PANELS = [(14, "graph-dominated"), (5, "embedding-dominated")]
CHANNELS = ["tau", "log r", "v"]
PAIR_NAMES = [(0, 1, "graph x feat"), (0, 2, "graph x emb"), (1, 2, "feat x emb")]


def main() -> None:
    from score_bundle.downstream import load_piece_arrays

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    with open(INPUTS, "rb") as fh:
        inp = pickle.load(fh)
    masks, meta = inp["masks"], inp["meta"]
    with open(".cache/kernel_sweep_emb_ma.pkl", "rb") as fh:
        embs = pickle.load(fh)["emb_ma"]

    # ---- contrast figure --------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 4.8), dpi=200, sharex=True)
    for ax, (pi, tag) in zip(axes, PANELS):
        p = ev[pi]
        mask = masks[(pi, SEED)]
        Y, comps = fit_components(p, mask, embs[(pi, SEED)])
        order = np.argsort(np.asarray(p["onset"], dtype=float), kind="stable")
        idx = order[WINDOW]
        xs = np.arange(len(idx))
        obs = mask[idx]
        ax.axhline(0.0, color=MUTED, lw=0.6, alpha=0.5)
        ax.plot(xs, comps["total"][idx, CHAN], color=INK, lw=1.8, alpha=0.35,
                label="posterior mean (sum)")
        for key, label, color in COMPS:
            ax.plot(xs, comps[key][idx, CHAN], color=color, lw=1.1, label=label)
        ax.plot(xs[obs], Y[idx, CHAN][obs], ".", color=MUTED, ms=5,
                label="observed note")
        ax.plot(xs[~obs], Y[idx, CHAN][~obs], "o", color=INK, ms=4.5,
                markerfacecolor="white", markeredgewidth=1.1,
                label="hidden note (truth)")
        ax.set_ylabel(NAMES[CHAN], fontsize=9, color=INK)
        ax.annotate(f"piece {pi} ({p.get('composer', '?')}): {tag}",
                    xy=(0.01, 1.02), xycoords="axes fraction", ha="left",
                    fontsize=9, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left",
                   bbox_to_anchor=(0.0, 1.60))
    axes[-1].set_xlabel("note (score order)", fontsize=9, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(f"wrote {OUT_FIG}", flush=True)

    # ---- disagreement statistics over the dev set -------------------------
    if "--fig-only" in sys.argv:
        return
    keys = [k for k, _, _ in COMPS]
    frac = np.zeros((len(PAIR_NAMES), 3))
    tot = np.zeros(3)
    for pi in range(meta["n_eval_pieces"]):
        mask = masks[(pi, SEED)]
        _, comps = fit_components(ev[pi], mask, embs[(pi, SEED)])
        held = ~mask
        for c in range(3):
            scale = 0.25 * float(np.std(comps["total"][held, c]))
            for j, (a, b, _) in enumerate(PAIR_NAMES):
                ma = comps[keys[a]][held, c]
                mb = comps[keys[b]][held, c]
                dis = (ma * mb < 0) & (np.abs(ma) > scale) & (np.abs(mb) > scale)
                frac[j, c] += dis.sum()
            tot[c] += held.sum()
        print(f"  piece {pi} done", flush=True)
    print("\nFraction of held-out notes with substantial opposite-sign pulls")
    print("(both |m_c| > 0.25 x channel posterior-mean std; 30 dev pieces):")
    print(f"{'pair':>14} " + " ".join(f"{c:>8}" for c in CHANNELS))
    for j, (_, _, name) in enumerate(PAIR_NAMES):
        print(f"{name:>14} " + " ".join(f"{frac[j, c] / tot[c]:8.3f}"
                                        for c in range(3)))


if __name__ == "__main__":
    main()
