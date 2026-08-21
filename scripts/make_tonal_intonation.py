import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from eval_graphgp import bootstrap_ci

INK, MUTED, BLUE, VERM = "#1A1A1A", "#6B7280", "#0072B2", "#D55E00"
with open("results/phase2_tonal_cells.pkl", "rb") as fh:
    blob = pickle.load(fh)
by = {}
for key, seed, sname, c, inst, rmse, nll in blob["per_track"]:
    if c == 0:
        by.setdefault((key, seed), {})[sname] = rmse
pairs = [(v["gp_tonal_asgiven"] - v["gp_asgiven"]) for v in by.values()
         if "gp_tonal_asgiven" in v and "gp_asgiven" in v]
d = np.array(pairs)
mu, lo, hi = bootstrap_ci(d, B=2000, rng=np.random.default_rng(31))

fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=200)
rng = np.random.default_rng(0)
ax.axvline(0, color=MUTED, lw=0.8)
ax.scatter(d, rng.uniform(-0.32, 0.32, d.size), s=14, color=BLUE, alpha=0.45,
           linewidths=0, label="per-(track, seed) paired delta")
ax.errorbar([mu], [0.62], xerr=[[mu - lo], [hi - mu]], fmt="o", color=VERM,
            ms=6, elinewidth=2.2, capsize=4,
            label=f"mean {mu:+.2f} cents  [{lo:+.2f}, {hi:+.2f}]")
better = (d < 0).mean()
ax.text(0.985, 0.06, f"{better:.0%} of track-seeds improve",
        transform=ax.transAxes, ha="right", fontsize=9, color=INK)
ax.set_xlabel("intonation RMSE, circle-of-fifths metric − plain graph "
              "(cents; negative = tonal better)", fontsize=9.5, color=INK)
ax.set_yticks([])
ax.set_ylim(-0.55, 0.9)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
out = "docs/thesis/figures/tonal_intonation_dev.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "| n =", d.size, f"| mean {mu:+.3f} [{lo:+.3f},{hi:+.3f}]")
# Thesis figure generator (fig:tonal-intonation); caption carries the title.
# Reads results/phase2_tonal_cells.pkl — no refitting.
