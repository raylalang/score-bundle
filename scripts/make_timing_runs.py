#!/usr/bin/env python
"""Timing channel along score order: neighbours err together (lag-1)."""
import os, pickle, sys
import numpy as np
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
data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
best = None
for key, d in sorted(data.items()):
    tau = d["tau"][:, 0] if d["tau"].ndim > 1 else d["tau"]
    ok = np.isfinite(tau)
    if ok.sum() < 100:
        continue
    r = np.corrcoef(tau[ok][:-1], tau[ok][1:])[0, 1]
    # representative, not cherry-picked: nearest the study's median rho
    if best is None or abs(r - 0.55) < abs(best[0] - 0.55):
        best = (float(r), key, tau[ok])
r, key, tau = best
tau = tau - tau.mean()
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 2.9),
                           gridspec_kw={"width_ratios": [2.1, 1]})
a.plot(np.arange(tau.size), tau * 1000, "-", color=BLUE, lw=1.0, alpha=0.6)
a.plot(np.arange(tau.size), tau * 1000, ".", color=BLUE, ms=4)
a.set_xlabel("note (score order)")
a.set_ylabel("timing residual (ms)")
a.set_title("A  one track's timing channel: runs, not white noise",
            loc="left")
b.plot(tau[:-1] * 1000, tau[1:] * 1000, ".", color=VERM, ms=4)
b.set_xlabel("note $i$ (ms)")
b.set_ylabel("note $i{+}1$ (ms)")
b.set_title(f"B  neighbours together: r = {r:.2f}", loc="left")
fig.tight_layout()
fig.savefig("docs/thesis/figures/timing_runs.png", bbox_inches="tight")
print(f"wrote timing_runs.png ({key}, r={r:.2f}, n={tau.size})")
