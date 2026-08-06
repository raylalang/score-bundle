#!/usr/bin/env python
"""Phase-2 real-audio example figure: one URMP development track.

Intonation channel of one track, end to end: the confidence-filtered cents
curve from pyin (light), the per-note estimator targets (dots, with their
own +/- sigma), the cell-mask graph GP's posterior band fitted with 30% of
notes hidden, and the hidden notes' estimator values as open circles.  The
vibrato-extent panel below shows the per-(note,channel) missingness case:
notes whose extent the estimator could not identify get a posterior anyway.

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/make_phase2_example.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_phase2_real import CACHE, _fit_systems  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "docs/thesis/figures/phase2_real_example.png"
TRACK = (1, 1)                 # Jupiter, violin — the calibrated smoke track
SEED = 0
_Z90 = 1.6448536269514722
INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"


def main() -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.score import Score

    with open(CACHE, "rb") as fh:
        d = pickle.load(fh)[TRACK]
    est, var, ident = d["est"], d["var"], d["ident"]
    n = est.shape[0]
    usable = np.isfinite(est[:, 0])
    score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                              np.zeros(n, dtype=int))
    eig = np.linalg.eigh(laplacian(build_adjacency(score)))
    X = rich_score_features(score, rff_dim=0)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
    scale = np.ones((n, 3))
    for c in range(3):
        v = var[:, c]
        med = np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 1.0
        scale[:, c] = np.where(np.isfinite(v),
                               np.clip(v / max(med, 1e-12), 1e-2, 1e3), 1.0)
    rng = np.random.default_rng(1000 + 7 * TRACK[0] + TRACK[1] + SEED)
    held = (rng.random(n) < 0.30) & usable
    mask = np.zeros((n, 3), dtype=bool)
    mask[:, 0] = usable & ~held
    mask[:, 1] = mask[:, 2] = usable & ~held & ident
    Yobs = np.where(mask, np.nan_to_num(est), 0.0)
    m, sd, sd_pred = _fit_systems(eig, feats, Yobs, mask, scale, var,
                                  None)["gp_asgiven"]

    x = d["onset"]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.4), dpi=200, sharex=True,
                             height_ratios=[3, 2])
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.5)
    ax.fill_between(x, m[:, 0] - _Z90 * sd_pred[:, 0],
                    m[:, 0] + _Z90 * sd_pred[:, 0], color=BLUE, alpha=0.15,
                    linewidth=0, label="GP 90% predictive band")
    ax.plot(x, m[:, 0], color=BLUE, lw=1.3, label="GP posterior mean")
    o = mask[:, 0]
    ax.errorbar(x[o], est[o, 0], yerr=np.sqrt(var[o, 0]), fmt=".",
                color=MUTED, ms=5, elinewidth=0.7, capsize=0,
                label="observed note (estimator ± σ)")
    h = held & np.isfinite(est[:, 0])
    ax.plot(x[h], est[h, 0], "o", color=INK, ms=4.5, markerfacecolor="white",
            markeredgewidth=1.1, label="hidden note (estimator value)")
    ax.set_ylabel("intonation $c_i$ (cents)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left",
              bbox_to_anchor=(0.0, 1.26))

    ax = axes[1]
    gam = np.exp(m[:, 1])
    lo = np.exp(m[:, 1] - _Z90 * sd[:, 1])
    hi = np.exp(m[:, 1] + _Z90 * sd[:, 1])
    ax.fill_between(x, lo, hi, color=GREEN, alpha=0.15, linewidth=0)
    ax.plot(x, gam, color=GREEN, lw=1.3, label="GP posterior extent")
    o2 = mask[:, 1]
    ax.plot(x[o2], np.exp(est[o2, 1]), ".", color=MUTED, ms=5,
            label="observed extent")
    miss = usable & ~held & ~ident
    ax.plot(x[miss], gam[miss], "s", color=VERM, ms=4,
            markerfacecolor="none", markeredgewidth=1.1,
            label="estimator-missing cell (GP fills it)")
    ax.set_ylabel(r"vibrato extent $\gamma_i$ (cents)", fontsize=9, color=INK)
    ax.set_xlabel("time (s)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, 1.22))
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_title(
        f"Phase 2 on real audio: {d['name']} ({d['instrument']}), "
        f"30% of notes hidden", fontsize=10, color=INK, loc="left", pad=52)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
