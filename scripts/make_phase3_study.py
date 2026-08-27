#!/usr/bin/env python
"""Figure for the Phase-3 waveform study (fig:phase3-study).

Two panels from the study cells (no refits):
  A. accuracy ladder --- median |waveform c - quasi-truth c| per position
     model, against the pyin+NLLS estimator chain's median;
  B. the calibration diagnosis --- coverage@90 vs quasi-truth for every
     variant (flat near zero) next to the model-true self-consistency
     coverage (exactly nominal), which locates the failure in the
     estimand gap rather than the inference.

Reads results/phase3_cells/{wave.shard*, wavedev.shard*, selfcheck_z.pkl}
(produced by eval_phase3_waveform_dev.py run/rundev and
eval_phase3_selfcheck.py). House style: Okabe-Ito, legends above panels.

    PYTHONPATH=src:scripts python scripts/make_phase3_study.py
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
OUT = "docs/thesis/figures/phase3_study_dev.png"
_Z90 = 1.6448536269514722


def main() -> None:
    rows = []
    for f in sorted(glob.glob("results/phase3_cells/wave.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    dev = {}
    for f in sorted(glob.glob("results/phase3_cells/wavedev.shard*.pkl")):
        for r in pickle.load(open(f, "rb")):
            dev[(tuple(r["key"]), r["i"])] = r["dev8"]
    for r in rows:
        r["dev8"] = dev[(tuple(r["key"]), r["i"])]
    z_self = pickle.load(open("results/phase3_cells/selfcheck_z.pkl", "rb"))

    variants = [("flat", "constant $c$"),
                ("ar1", "+ AR(1) noise"),
                ("drift", "+ drift term"),
                ("dev8", "+ deviation prior")]
    med_err, cov = [], []
    for v, _ in variants:
        err = np.array([abs(r[v][0] - r["gt_c"]) for r in rows])
        z = np.array([(r[v][0] - r["gt_c"]) / r[v][1] for r in rows])
        med_err.append(float(np.median(err)))
        cov.append(float(np.mean(np.abs(z) <= _Z90)))
    est_med = float(np.median([abs(r["est_c"] - r["gt_c"]) for r in rows
                               if np.isfinite(r["est_c"])]))
    cov_self = float(np.mean(np.abs(z_self) <= _Z90))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 3.6), dpi=200,
                                   width_ratios=[1.15, 1])
    xs = np.arange(len(variants))
    labels = [lab for _, lab in variants]

    axa.bar(xs, med_err, 0.62, color=BLUE, alpha=0.85, zorder=3)
    axa.axhline(est_med, color=VERM, lw=1.4, ls="--", zorder=4,
                label=f"pyin + NLLS estimator chain ({est_med:.2f})")
    for x, e in zip(xs, med_err):
        axa.text(x, e + 0.05, f"{e:.2f}", ha="center", fontsize=8.5,
                 color=INK)
    axa.set_xticks(xs, labels, fontsize=8.5)
    axa.set_ylabel("median $|c - c^{\\mathrm{quasi\\text{-}truth}}|$ (cents)",
                   fontsize=9, color=INK)
    axa.set_title("A. accuracy: no tracker in the loop "
                  "(376 development notes)", fontsize=10, color=INK,
                  loc="left", pad=24)
    axa.legend(frameon=False, fontsize=8, loc="lower left",
               bbox_to_anchor=(0.0, 1.0))
    axa.set_ylim(0, max(med_err) * 1.25)

    xs_b = np.arange(len(variants) + 1)
    covs = cov + [cov_self]
    cols = [BLUE] * len(variants) + [GREEN]
    axb.bar(xs_b, covs, 0.62, color=cols, alpha=0.9, zorder=3)
    axb.axhline(0.90, color=INK, lw=1.2, ls=":", zorder=4,
                label="nominal 0.90")
    for x, cv in zip(xs_b, covs):
        axb.text(x, cv + 0.02, f"{cv:.2f}", ha="center", fontsize=8.5,
                 color=INK)
    axb.set_xticks(xs_b, labels + ["model-true\n(self-check)"],
                   fontsize=8)
    axb.set_ylabel("coverage@90 vs quasi-truth", fontsize=9, color=INK)
    axb.set_title("B. calibration: the gap is the estimand, "
                  "not the inference", fontsize=10, color=INK,
                  loc="left", pad=24)
    axb.legend(frameon=False, fontsize=8, loc="lower left",
               bbox_to_anchor=(0.0, 1.0))
    axb.set_ylim(0, 1.05)

    for ax in (axa, axb):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
