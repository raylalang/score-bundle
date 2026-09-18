#!/usr/bin/env python
"""The SM-estimator failure mode, made visible on ONE real note.

Companion to make_sm_explainer.py: find a development note where the
SM-GP's read-outs disagree badly between the note's two measurements
(tracked curve vs ground-truth curve) while the sine fit's agree, refit
both estimators on both curves, and show why: the flexible decomposition
settles differently on each curve; the rigid fit asks a narrower question
and gets the same answer twice.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_sm_worstcase.py
"""
from __future__ import annotations

import glob
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
BLUE, VERM = "#0072B2", "#D55E00"
OUT = "docs/thesis/figures/sm_estimator_worstcase.png"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def pick_worst():
    """Note with large SM cross-curve rate disagreement, small sine one."""
    rows = []
    for f in sorted(glob.glob(".cache/sm_dev_shard_*_4.pkl")):
        rows += pickle.load(open(f, "rb"))
    cand = []
    for r in rows:
        if not (r.get("ident_tr") and r.get("ident_gt")
                and "sm_tr" in r and "sm_gt" in r
                and np.isfinite(r["nl_tr"]["lf"])
                and np.isfinite(r["nl_gt"]["lf"])):
            continue
        if r["dur"] < 0.8 or r["n_tr"] < 60 or r["n_gt"] < 60:
            continue
        d_sm = abs(r["sm_tr"]["lf"] - r["sm_gt"]["lf"])
        d_nl = abs(r["nl_tr"]["lf"] - r["nl_gt"]["lf"])
        cand.append((d_sm - d_nl, d_sm, d_nl, r))
    cand.sort(key=lambda z: -z[0])
    return cand[0][3]


def curves_for(rec):
    from eval_phase2_real import dev_unique_tracks
    from score_bundle.phase2.intonation import cents_from_f0
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)

    key, i = tuple(rec["key"]), rec["i"]
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))[key]
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))[key]
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
    tr = tracks[key]
    notes = read_notes_annotation(tr.notes)
    t_gt, f0_gt = read_f0_annotation(tr.f0s)
    on, du = notes["onset"][i], notes["duration"][i]
    midi = float(data["midi"][i])
    ok = f0s["voiced"] & np.isfinite(f0s["f0"]) & (f0s["f0"] > 0)
    ok &= f0s["prob"] >= np.quantile(f0s["prob"][ok], 0.2)
    sel = ok & (f0s["t"] >= on) & (f0s["t"] < on + du)
    okg = np.isfinite(f0_gt) & (f0_gt > 0)
    selg = okg & (t_gt >= on) & (t_gt < on + du)
    tt = f0s["t"][sel] - on
    x = cents_from_f0(f0s["f0"][sel], 440.0, midi - 69)
    ttg = t_gt[selg] - on
    xg = cents_from_f0(f0_gt[selg], 440.0, midi - 69)
    return key, i, data["instrument"], (tt, x), (ttg, xg)


def main() -> None:
    from score_bundle.phase2.intonation import fit_vibrato_note
    from score_bundle.phase2.sm_estimator import fit_sm_note, sm_predict

    rec = pick_worst()
    key, i, instr, (tt, x), (ttg, xg) = curves_for(rec)
    print(f"worst note: track {key} #{i} ({instr}), "
          f"{tt.size}/{ttg.size} frames")

    fits = {}
    for name, (t_, x_) in (("tracked", (tt, x)), ("ground truth", (ttg, xg))):
        fits[name] = (fit_vibrato_note(t_, x_), fit_sm_note(t_, x_))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for k, (name, (t_, x_)) in enumerate(
            (("tracked", (tt, x)), ("ground truth", (ttg, xg)))):
        ax = axes[k]
        nl, sm = fits[name]
        tg = np.linspace(t_.min(), t_.max(), 400)
        m, v = sm_predict(t_, x_, sm["params"], tg, include_noise=False)
        sine = nl["c"] + nl["gamma"] * np.sin(2 * np.pi * nl["f"]
                                              * (tg - nl["delta"]))
        ax.plot(t_, x_, ".", ms=3, color=MUTED)
        ax.plot(tg, sine, color=VERM, lw=1.5,
                label=f"sine: rate {nl['f']:.1f} Hz, "
                      f"extent {nl['gamma']:.1f} c")
        ax.plot(tg, m, color=BLUE, lw=1.5,
                label=f"SM-GP: rate {sm['f']:.1f} Hz, "
                      f"extent {sm['gamma']:.1f} c")
        ax.fill_between(tg, m - 1.645 * np.sqrt(v), m + 1.645 * np.sqrt(v),
                        color=BLUE, alpha=0.16, linewidth=0)
        ax.set_xlabel("time in the note (s)")
        ax.set_ylabel("pitch deviation (cents)" if k == 0 else "")
        ax.margins(y=0.24)
        ax.set_title(f"{'AB'[k]}  the {name} curve of the SAME note",
                     loc="left")
        ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[2]
    for j, (q, lab) in enumerate((("f", "rate (Hz)"),
                                  ("gamma", "extent (cents)"))):
        y0 = 2.4 - 2.2 * j
        for name, mk in (("tracked", "o"), ("ground truth", "s")):
            nl, sm = fits[name]
            ax.plot(nl[q], y0 + 0.35, mk, ms=6, color=VERM)
            ax.plot(sm[q], y0 - 0.35, mk, ms=6, color=BLUE)
        nl_a, sm_a = fits["tracked"]
        nl_b, sm_b = fits["ground truth"]
        ax.annotate("", xy=(sm_b[q], y0 - 0.35), xytext=(sm_a[q], y0 - 0.35),
                    arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2))
        ax.annotate("", xy=(nl_b[q], y0 + 0.35), xytext=(nl_a[q], y0 + 0.35),
                    arrowprops=dict(arrowstyle="-", color=VERM, lw=1.2))
        ax.text(ax.get_xlim()[0], y0 + 0.9, lab, fontsize=8.5, color=INK)
    ax.set_yticks([])
    ax.set_ylim(-1.2, 3.6)
    ax.set_xlabel("read-out per curve (circle = tracked, square = ground "
                  "truth); the line is the disagreement")
    ax.set_title("C  two witnesses: sine agrees with itself, SM flips",
                 loc="left")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)
    for name in ("tracked", "ground truth"):
        nl, sm = fits[name]
        print(f"{name:13s} sine: f={nl['f']:.2f} g={nl['gamma']:.2f} | "
              f"SM: f={sm['f']:.2f} g={sm['gamma']:.2f}")


if __name__ == "__main__":
    main()
