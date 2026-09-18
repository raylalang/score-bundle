#!/usr/bin/env python
"""Phase-2 data intro figure: what the raw material looks like.

Three panels of real URMP development data, no fits anywhere:
(A) one note measured twice -- the tracker's cents frames against the
    corpus ground-truth curve of the same note;
(B) a note whose curve supports the vibrato channels (identifiable);
(C) a note the identifiability rule refuses (channels become missing
    cells).

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_phase2_intro.py
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
BLUE, VERM = "#0072B2", "#D55E00"
OUT = "docs/thesis/figures/phase2_frames_intro.png"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def note_curves(key, i, data, f0s, tracks):
    from score_bundle.phase2.intonation import cents_from_f0
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)
    tr = tracks[key]
    notes = read_notes_annotation(tr.notes)
    t_gt, f0_gt = read_f0_annotation(tr.f0s)
    on, du = notes["onset"][i], notes["duration"][i]
    midi = float(data[key]["midi"][i])
    f0c = f0s[key]
    ok = f0c["voiced"] & np.isfinite(f0c["f0"]) & (f0c["f0"] > 0)
    ok &= f0c["prob"] >= np.quantile(f0c["prob"][ok], 0.2)
    sel = ok & (f0c["t"] >= on) & (f0c["t"] < on + du)
    okg = np.isfinite(f0_gt) & (f0_gt > 0)
    selg = okg & (t_gt >= on) & (t_gt < on + du)
    tt = f0c["t"][sel] - on
    x = cents_from_f0(f0c["f0"][sel], 440.0, midi - 69)
    ttg = t_gt[selg] - on
    xg = cents_from_f0(f0_gt[selg], 440.0, midi - 69)
    return tt, x, ttg, xg


def main() -> None:
    from eval_phase2_real import dev_unique_tracks

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    # A: a note with plenty of frames on both curves (two measurements)
    # B: identifiable vibrato with a healthy extent
    # C: refused (fit ran, vibrato unidentifiable)
    pick_a = pick_c = None
    best_b = None
    for key in sorted(data)[:12]:
        d = data[key]
        for i in range(d["onset"].size):
            n, ident = int(d["n_frames"][i]), bool(d["ident"][i])
            if pick_a is None and n >= 90 and ident:
                pick_a = (key, i)
            if ident and 80 <= n <= 300:
                tt, x, _, _ = note_curves(key, i, data, f0s, tracks)
                dur = tt.max() - tt.min()
                if 1.1 <= dur <= 2.6 and np.abs(x).max() <= 60:
                    score = n + 10 * (x.max() - x.min())
                    if best_b is None or score > best_b[0]:
                        best_b = (score, key, i)
            if pick_c is None and (not ident) and 4 <= n <= 20:
                pick_c = (key, i)
    pick_b = (best_b[1], best_b[2])
    print("picked:", pick_a, pick_b, pick_c)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))

    key, i = pick_a
    tt, x, ttg, xg = note_curves(key, i, data, f0s, tracks)
    ax = axes[0]
    ax.plot(ttg, xg, ".", ms=3.5, color=VERM, label=f"ground truth ({ttg.size} frames)")
    ax.plot(tt, x, ".", ms=3.5, color=BLUE, label=f"tracked ({tt.size} frames)")
    ax.set_title("A  one note, measured twice", loc="left")
    ax.set_ylabel("pitch deviation (cents)")
    ax.legend(fontsize=7.5, loc="best")

    key, i = pick_b
    tt, x, _, _ = note_curves(key, i, data, f0s, tracks)
    ax = axes[1]
    ax.plot(tt, x, ".", ms=3.5, color=BLUE)
    ax.set_title("B  vibrato identifiable: channels observed", loc="left")

    key, i = pick_c
    tt, x, _, _ = note_curves(key, i, data, f0s, tracks)
    ax = axes[2]
    ax.plot(tt, x, ".", ms=4.5, color=BLUE)
    ax.text(0.03, 0.94, f"{tt.size} frames, no cycle to fit",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    ax.set_title("C  refused: vibrato channels missing", loc="left")

    for ax in axes:
        ax.set_xlabel("time in the note (s)")
        ax.margins(y=0.25)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
