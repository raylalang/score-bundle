#!/usr/bin/env python
"""Hidden notes, one piece: the full model vs the no-graph ablation."""
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
                     "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.dpi": 200})
import eval_graphgp as eg
from score_bundle.downstream import load_piece_arrays

inputs = pickle.load(open(".cache/kernel_sweep_inputs.pkl", "rb"))
masks, imeta = inputs["masks"], inputs["meta"]
emb_dump = pickle.load(open(".cache/kernel_sweep_emb_ma.pkl", "rb"))["emb_ma"]
head, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
ev = ev[:imeta["n_eval_pieces"]]
pi, s = 0, 0
p = ev[pi]
Y = np.asarray(p["y"], float)
mask = masks[(pi, s)]
emb = emb_dump[(pi, s)]
out = {}
for name, kernel in (("with the graph", "additive"), ("no graph", "none")):
    feats, graph_eig, n_graph, g0 = eg.piece_setup(p, "b_featlm", emb=emb)
    cell, info = eg.fit_and_predict(Y, mask, feats, graph_eig, n_graph, g0,
                                    kernel)
    out[name] = cell
    print(name, "done", flush=True)

fig, ax = plt.subplots(figsize=(10.5, 3.1))
CH = 2  # velocity
yt, pr_g, sd_g, ch = out["with the graph"]
_, pr_n, sd_n, _ = out["no graph"]
sel = np.where(ch == CH)[0][:40]
ix = np.arange(sel.size)
rg = float(np.sqrt(np.mean((pr_g[sel] - yt[sel])**2)))
rn = float(np.sqrt(np.mean((pr_n[sel] - yt[sel])**2)))
ax.errorbar(ix - 0.12, pr_n[sel], yerr=1.645*sd_n[sel], fmt="o", ms=3.5,
            color=GREEN, capsize=2, lw=1,
            label=f"no graph (RMSE {rn:.3f})")
ax.errorbar(ix + 0.12, pr_g[sel], yerr=1.645*sd_g[sel], fmt="o", ms=3.5,
            color=BLUE, capsize=2, lw=1,
            label=f"with the graph (RMSE {rg:.3f})")
ax.plot(ix, yt[sel], "o", ms=5.5, mfc="none", mec=VERM, mew=1.4,
        label="hidden truth")
ax.set_xlabel("hidden note (order within the piece)")
ax.set_ylabel("velocity $v_i$")
ax.set_title("The same hidden notes, predicted with and without the graph "
             f"(piece: {p.get('title', 'validation piece 0')})", loc="left")
ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.margins(y=0.3)
fig.tight_layout()
fig.savefig("docs/thesis/figures/variant_compare.png", bbox_inches="tight")
print("wrote variant_compare.png | RMSE graph", rg, "nograph", rn)
