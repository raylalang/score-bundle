#!/usr/bin/env python
"""The two-component kernel, drawn: k(dt), S(xi), prior samples."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, BLUE, VERM, ORANGE = "#1A1A1A", "#6B7280", "#0072B2", "#D55E00", "#E69F00"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
                     "legend.frameon": False, "figure.dpi": 200})
w1, mu1, v1, w2, v2 = 30.0, 5.5, 0.4, 60.0, 0.3
fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.0, 2.9))
dt = np.linspace(0, 1.0, 400)
k1 = w1 * np.exp(-2 * np.pi**2 * v1 * dt**2) * np.cos(2 * np.pi * mu1 * dt)
k2 = w2 * np.exp(-2 * np.pi**2 * v2 * dt**2)
a.plot(dt, k1, color=BLUE, lw=1.6, label="vibrato component")
a.plot(dt, k2, color=ORANGE, lw=1.6, label="drift component")
a.plot(dt, k1 + k2, color=INK, lw=1.0, ls=":", label="sum $k(\\Delta t)$")
a.set_xlabel("lag $\\Delta t$ (s)"); a.set_ylabel("covariance (cents$^2$)")
a.set_title("A  the kernel", loc="left"); a.legend(fontsize=7.5)
xi = np.linspace(0, 10, 400)
S1 = 0.5 * w1 * (np.exp(-(xi - mu1)**2 / (2*v1)) + np.exp(-(xi + mu1)**2 / (2*v1))) / np.sqrt(2*np.pi*v1)
S2 = w2 * np.exp(-xi**2 / (2*v2)) / np.sqrt(2*np.pi*v2)
b.plot(xi, S1, color=BLUE, lw=1.6)
b.plot(xi, S2, color=ORANGE, lw=1.6)
b.annotate("vibrato band at $\\mu_1$", xy=(mu1, S1.max()), xytext=(6.4, S1.max()*2.6),
           color=BLUE, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
b.text(1.1, S2.max()*0.55, "drift, at zero", color=ORANGE, fontsize=8.5)
b.set_yscale("log"); b.set_ylim(1e-2, 300)
b.set_xlabel("frequency $\\xi$ (Hz)"); b.set_ylabel("spectral density")
b.set_title("B  where the energy lives", loc="left")
tt = np.linspace(0, 1.5, 300)
tau = tt[:, None] - tt[None, :]
K = (w1 * np.exp(-2*np.pi**2*v1*tau**2) * np.cos(2*np.pi*mu1*tau)
     + w2 * np.exp(-2*np.pi**2*v2*tau**2))
K[np.diag_indices_from(K)] += 1e-6 * (w1 + w2)
L = np.linalg.cholesky(K)
rng = np.random.default_rng(4)
for i, col in enumerate([BLUE, VERM, MUTED]):
    c.plot(tt, L @ rng.standard_normal(tt.size), color=col, lw=1.1)
c.set_xlabel("time in the note (s)"); c.set_ylabel("cents")
c.set_title("C  three draws from the prior", loc="left")
fig.tight_layout()
fig.savefig("docs/thesis/figures/kernel_viz.png", bbox_inches="tight")
print("wrote kernel_viz.png")
