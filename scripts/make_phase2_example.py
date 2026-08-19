#!/usr/bin/env python
"""Phase-2 real-audio example figure: one URMP development track.

Four channels of the six-channel bundle on one track (as-given fit, 30% of
notes hidden): intonation (posterior band, observed targets with their own
sigma, hidden notes as open circles), vibrato extent (the per-(note,channel)
missingness case — estimator-missing cells get a posterior anyway), the
onset-anchored timing channel tau (aligner error in the noise row), and the
gated-fit vibrato onset delay delta_vib.  Loudness is omitted for space (its
graph contrast is the bundle's honest ns cell).

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
ORANGE, PURPLE = "#E69F00", "#CC79A7"


def main() -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.score import Score

    with open(CACHE, "rb") as fh:
        d = pickle.load(fh)[TRACK]
    est = np.concatenate([d["est"], d["ell"][:, None],
                          d["tau"][:, None], d["dvib"][:, None]], axis=1)
    var = np.concatenate([d["var"], d["var_ell"][:, None],
                          d["var_tau"][:, None],
                          d["var_dvib"][:, None]], axis=1)
    ident = d["ident"]
    n = est.shape[0]
    usable = np.isfinite(est[:, 0]) & (np.abs(est[:, 0]) <= 150.0)
    score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                              np.zeros(n, dtype=int))
    eig = np.linalg.eigh(laplacian(build_adjacency(score)))
    X = rich_score_features(score, rff_dim=0)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
    k = 6
    scale = np.ones((n, k))
    for c in range(k):
        v = var[:, c]
        med = np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 1.0
        scale[:, c] = np.where(np.isfinite(v),
                               np.clip(v / max(med, 1e-12), 1e-2, 1e3), 1.0)
    rng = np.random.default_rng(1000 + 7 * TRACK[0] + TRACK[1] + SEED)
    held = (rng.random(n) < 0.30) & usable
    mask = np.zeros((n, k), dtype=bool)
    mask[:, 0] = usable & ~held
    mask[:, 1] = mask[:, 2] = usable & ~held & ident
    mask[:, 3] = usable & ~held & np.isfinite(est[:, 3])
    mask[:, 4] = ~held & np.isfinite(est[:, 4])
    mask[:, 5] = ~held & np.isfinite(est[:, 5])
    Yobs = np.where(mask, np.nan_to_num(est), 0.0)
    m, sd, sd_pred = _fit_systems(eig, feats, Yobs, mask, scale, var,
                                  None)["gp_asgiven"]

    x = d["onset"]
    fig, axes = plt.subplots(4, 1, figsize=(10.2, 9.6), dpi=200, sharex=True,
                             height_ratios=[3, 2, 2, 2])
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
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, 1.30))

    ax = axes[2]                                   # tau (ms)
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.5)
    ax.fill_between(x, 1e3 * (m[:, 4] - _Z90 * sd_pred[:, 4]),
                    1e3 * (m[:, 4] + _Z90 * sd_pred[:, 4]), color=ORANGE,
                    alpha=0.18, linewidth=0)
    ax.plot(x, 1e3 * m[:, 4], color=ORANGE, lw=1.3,
            label="GP posterior timing")
    o4 = mask[:, 4]
    ax.errorbar(x[o4], 1e3 * est[o4, 4], yerr=1e3 * np.sqrt(var[o4, 4]),
                fmt=".", color=MUTED, ms=5, elinewidth=0.7, capsize=0,
                label=r"observed $\tau$ (warp $\pm\sigma$)")
    h4 = held & np.isfinite(est[:, 4])
    ax.plot(x[h4], 1e3 * est[h4, 4], "o", color=INK, ms=4.5,
            markerfacecolor="white", markeredgewidth=1.1,
            label="hidden note")
    ax.set_ylabel(r"timing $\tau_i$ (ms)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, 1.30))

    ax = axes[3]                                   # delta_vib (ms)
    ax.fill_between(x, 1e3 * (m[:, 5] - _Z90 * sd[:, 5]),
                    1e3 * (m[:, 5] + _Z90 * sd[:, 5]), color=PURPLE,
                    alpha=0.18, linewidth=0)
    ax.plot(x, 1e3 * m[:, 5], color=PURPLE, lw=1.3,
            label="GP posterior onset delay")
    o5 = mask[:, 5]
    ax.plot(x[o5], 1e3 * est[o5, 5], ".", color=MUTED, ms=5,
            label="observed delay (gated fit)")
    miss5 = ~held & ~np.isfinite(est[:, 5])
    ax.plot(x[miss5], 1e3 * m[miss5, 5], "s", color=VERM, ms=4,
            markerfacecolor="none", markeredgewidth=1.1,
            label="estimator-missing cell (GP fills it)")
    ax.set_ylabel(r"vibrato delay $\delta_i^{\mathrm{vib}}$ (ms)",
                  fontsize=9, color=INK)
    ax.set_xlabel("time (s)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, 1.30))
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_title(
        f"Phase 2 on real audio, four of six channels: {d['name']} "
        f"({d['instrument']}), 30% of notes hidden",
        fontsize=10, color=INK, loc="left", pad=52)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
