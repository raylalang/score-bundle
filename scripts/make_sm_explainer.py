#!/usr/bin/env python
"""The SM-estimator teaching figure: what happens on ONE real note.

Not a result -- an explainer. One vibrato-identifiable development note:
(A) the tracked cents frames, the sine fit's single rigid curve, and the
SM-GP's posterior band over the curve; (B) the SM posterior split into
its two components (the vibrato oscillation and the slow drift) plus the
centre -- the 'two damped cosines' made visible; (C) the read-outs of
both estimators side by side with their uncertainties.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_sm_explainer.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
OUT = "docs/thesis/figures/sm_estimator_explainer.png"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def pick_note():
    """A long, clearly vibrato-identifiable dev note with visible drift."""
    from eval_phase2_real import dev_unique_tracks
    from score_bundle.phase2.intonation import cents_from_f0
    from score_bundle.phase2.urmp import read_notes_annotation

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
    best = None
    for key in sorted(data)[:12]:
        d, f0c = data[key], f0s[key]
        notes = read_notes_annotation(tracks[key].notes)
        ok = f0c["voiced"] & np.isfinite(f0c["f0"]) & (f0c["f0"] > 0)
        ok &= f0c["prob"] >= np.quantile(f0c["prob"][ok], 0.2)
        for i in range(d["onset"].size):
            if not (d["ident"][i] and np.isfinite(d["est"][i, 2])):
                continue
            on, du = notes["onset"][i], notes["duration"][i]
            if not (1.2 <= du <= 2.6):
                continue
            sel = ok & (f0c["t"] >= on) & (f0c["t"] < on + du)
            if sel.sum() < 80:
                continue
            tt = f0c["t"][sel] - on
            x = cents_from_f0(f0c["f0"][sel], 440.0, float(d["midi"][i] - 69))
            if np.abs(x).max() > 60:
                continue
            span = x.max() - x.min()
            score = sel.sum() + 10 * span      # long + lively
            if best is None or score > best[0]:
                best = (score, key, i, d["instrument"], tt, x)
    _, key, i, instr, tt, x = best
    return key, i, instr, tt, x


def main() -> None:
    from score_bundle.phase2.intonation import fit_vibrato_note
    from score_bundle.phase2.sm_estimator import (fit_sm_note, sm_kernel,
                                                  sm_predict, _unpack,
                                                  _chol_terms)

    key, i, instr, tt, x = pick_note()
    print(f"note: track {key} #{i} ({instr}), {tt.size} frames, "
          f"{tt.max():.2f}s")

    nl = fit_vibrato_note(tt, x)
    sm = fit_sm_note(tt, x)
    p = sm["params"]
    w1, mu1, v1, w2, v2, s2 = _unpack(p)

    tg = np.linspace(tt.min(), tt.max(), 400)
    m_curve, v_curve = sm_predict(tt, x, p, tg, include_noise=False)
    sine = nl["c"] + nl["gamma"] * np.sin(2 * np.pi * nl["f"]
                                          * (tg - nl["delta"]))

    # component split at the fitted kernel (posterior means on the grid)
    L, a, b, _, _ = _chol_terms(tt, x, p)
    c_hat = b / a
    r = np.linalg.solve(L.T, np.linalg.solve(L, x - c_hat))
    tau_go = tg[:, None] - tt[None, :]
    K1g = w1 * np.exp(-2 * np.pi ** 2 * v1 * tau_go ** 2) \
        * np.cos(2 * np.pi * mu1 * tau_go)
    K2g = w2 * np.exp(-2 * np.pi ** 2 * v2 * tau_go ** 2)
    m1 = K1g @ r                               # vibrato component
    m2 = K2g @ r                               # drift component

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    ax = axes[0]
    ax.plot(tt, x, ".", ms=3, color=MUTED, label="tracked frames")
    ax.plot(tg, sine, color=VERM, lw=1.6,
            label="sine fit (one rigid shape)")
    ax.plot(tg, m_curve, color=BLUE, lw=1.6, label="SM-GP posterior mean")
    ax.fill_between(tg, m_curve - 1.645 * np.sqrt(v_curve),
                    m_curve + 1.645 * np.sqrt(v_curve),
                    color=BLUE, alpha=0.18, linewidth=0)
    ax.set_xlabel("time in the note (s)")
    ax.set_ylabel("pitch deviation (cents)")
    ax.margins(y=0.22)
    ax.set_title(f"A  one real {instr} note: two models of its curve",
                 loc="left")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.axhline(sm["c"], color=INK, lw=1.2, ls=":",
               label=f"centre read-out c = {sm['c']:.1f} cents "
                     "(realized average)")
    ax.plot(tg, c_hat + m2, color=ORANGE, lw=1.6,
            label="+ drift component (near 0 Hz)")
    ax.plot(tg, c_hat + m2 + m1, color=BLUE, lw=1.2, alpha=0.9,
            label=f"+ vibrato component ({sm['f']:.1f} Hz)")
    ax.set_xlabel("time in the note (s)")
    ax.set_ylabel("pitch deviation (cents)")
    ax.margins(y=0.22)
    ax.set_title("B  the same posterior, split into its parts", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[2]
    rows = [("centre (cents)", nl["c"], np.sqrt(nl["var_c"]),
             sm["c"], np.sqrt(sm["var_c"])),
            ("rate (Hz)", nl["f"], np.sqrt(nl["var_f"]),
             sm["f"], np.sqrt(sm["var_f"])),
            ("extent (cents)", nl["gamma"], np.sqrt(nl["var_gamma"]),
             sm["gamma"], np.sqrt(min(sm["var_gamma"], 1e4)))]
    for j, (lab, mv, ms_, sv, ss) in enumerate(rows):
        y = 2 - j
        ax.errorbar(mv, y + 0.12, xerr=1.645 * ms_, fmt="o", ms=5,
                    color=VERM, capsize=2.5)
        ax.errorbar(sv, y - 0.12, xerr=1.645 * ss, fmt="o", ms=5,
                    color=BLUE, capsize=2.5)
        ax.text(ax.get_xlim()[0], y + 0.32, lab, fontsize=8, color=INK)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 2.75)
    ax.set_xlabel("value with 90% interval "
                  "(vermilion = sine fit, blue = SM-GP)")
    ax.set_title("C  the read-outs: same three numbers, two witnesses",
                 loc="left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
