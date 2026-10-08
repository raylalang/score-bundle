#!/usr/bin/env python
"""Audio chapter visual: reconstruction + the likelihood over c."""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, BLUE, VERM = "#1A1A1A", "#6B7280", "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#E5E7EB",
                     "grid.linewidth": 0.6, "figure.dpi": 200})
import soundfile as sf
from scipy.signal import resample_poly
from eval_phase3_waveform_dev import (SR, chunked_design, f0_curve,
                                      fit_noise, loglik, selected)
from score_bundle.phase2.urmp import read_notes_annotation

# the clarinet note of the prototype figure
for key, d, tr in selected():
    if d["instrument"] == "cl":
        break
notes = read_notes_annotation(tr.notes)
i = 96
audio48, sr48 = sf.read(tr.audio)
audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
on = float(notes["onset"][i]); du = min(float(notes["duration"][i]), 2.0)
a0, b0 = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
x = audio[a0:b0]; t = np.arange(x.size) / SR
midi = float(d["midi"][i])
grid = np.arange(-50.0, 50.0, 0.5)
nv = fit_noise(x, t, midi, 0.0)
lls = np.array([loglik(x, t, midi, c, nv) for c in grid])
c_hat = float(grid[int(np.argmax(lls))])
Phi = chunked_design(f0_curve(t, midi, c_hat), t)
beta, *_ = np.linalg.lstsq(Phi, x, rcond=None)
recon = Phi @ beta
fig, (A, B) = plt.subplots(1, 2, figsize=(9.8, 2.9))
w = slice(int(0.40 * SR), int(0.44 * SR))
A.plot(t[w] * 1000, x[w], color=VERM, lw=1.0, label="recorded audio $x$")
A.plot(t[w] * 1000, recon[w], color=BLUE, lw=1.0, ls=(0, (4, 2)),
       label="model $\\Phi(z)\\,\\hat a$")
A.set_xlabel("time (ms)"); A.set_ylabel("amplitude")
A.set_title("A  40 ms of the note: audio vs model", loc="left")
A.legend(fontsize=8)
B.plot(grid, lls - lls.max(), color=BLUE, lw=1.4,
       label="log p(audio | curve at offset c)")
B.axvline(c_hat, color=VERM, lw=1.0, ls=":",
          label=f"best c = {c_hat:+.1f} cents")
B.set_xlabel("proposed intonation offset c (cents)")
B.set_ylabel("log probability, relative to peak")
B.set_ylim(-3000, 150)
B.legend(fontsize=8, loc="lower right")
B.set_title("B  how sharply the audio picks the pitch", loc="left")
fig.tight_layout()
fig.savefig("docs/thesis/figures/waveform_viz.png", bbox_inches="tight")
print("wrote waveform_viz.png, c_hat", c_hat)
