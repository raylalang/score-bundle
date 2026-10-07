#!/usr/bin/env python
"""The three pitch-curve channels, annotated on one real GT curve."""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, BLUE, VERM, GREEN = "#1A1A1A", "#6B7280", "#0072B2", "#D55E00", "#009E73"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#E5E7EB",
                     "grid.linewidth": 0.6, "figure.dpi": 200})
from make_phase2_intro import note_curves
from eval_phase2_real import dev_unique_tracks
from score_bundle.phase2.intonation import fit_vibrato_note
data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
_, x_, tg, xg = note_curves((1, 2), 6, data, f0s, tracks)
sel = (tg > 0.15) & (tg < tg.max() - 0.15)
tg, xg = tg[sel], xg[sel]
fit = fit_vibrato_note(tg, xg)
c, g, f = fit["c"], fit["gamma"], fit["f"]
fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.plot(tg, xg, ".", ms=3.5, color=MUTED)
ax.axhline(c, color=BLUE, lw=1.6, ls="--")
ax.text(tg[-1], c - 5.5, f"intonation c = {c:.1f} cents", color=BLUE,
        ha="right", fontsize=10)
w = (tg > 0.35) & (tg < 0.35 + 1.5 / f)
pk = float(tg[w][np.argmax(xg[w])])
ax.annotate("", xy=(pk, c + g), xytext=(pk, c),
            arrowprops=dict(arrowstyle="<->", color=VERM, lw=1.8))
ax.text(pk - 0.03, c + g / 2, f"extent γ = {g:.1f} cents",
        color=VERM, fontsize=10, ha="right")
yp = c + g + 7
ax.annotate("", xy=(pk + 1 / f, yp), xytext=(pk, yp),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.8))
ax.text(pk + 0.5 / f, yp + 2.5, f"period 1/f  (f = {f:.1f} Hz)",
        color=GREEN, ha="center", fontsize=10)
ax.set_xlabel("time in the note (s)")
ax.set_ylabel("pitch deviation (cents)")
ax.margins(y=0.3)
fig.tight_layout()
fig.savefig("docs/thesis/figures/channels_def.png", bbox_inches="tight")
print("wrote channels_def.png")
