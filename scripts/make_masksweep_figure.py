#!/usr/bin/env python
"""Masking-level sweep figure (DEV) for the thesis/presentation.

Two panels from results/graphgp_masksweep/ (the sweep of
docs/masking_sweep_results.md):
  A - note-pooled RMSE vs hidden fraction for the proposed model, its no-graph
      ablation, and the features-only variant; LOO limit as detached points
      (its protocol differs: hyperparameters fit on the full piece).
  B - the two paired per-piece contrasts with bootstrap 95% CIs at each level:
      graph value (proposed - no-graph) and embedding value (proposed - features).

Everything is read from the committed sweep results; deterministic.

    PYTHONPATH=src:scripts python scripts/make_masksweep_figure.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from eval_graphgp import bootstrap_ci  # noqa: E402
from report_mask_sweep import ROOT, load_cells, per_piece, pooled  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "docs/thesis/figures/masksweep_dev.png"
TAGS = [("obs0.50", 50), ("obs0.60_anchor", 40), ("obs0.70", 30),
        ("obs0.80", 20), ("obs0.90", 10)]
CONFIGS = [  # (key, label, okabe-ito colour, marker, linestyle)
    ("b_featlm", "proposed model", "#0072B2", "o", "-"),
    ("b_featlm_nograph", "no graph", "#E69F00", "s", "--"),
    ("b_feat", "features only", "#009E73", "^", ":"),
]
INK, MUTED = "#1A1A1A", "#6B7280"


def loo_pooled():
    import pickle
    from score_bundle.metrics import evaluate
    with open(os.path.join(ROOT, "loo.pkl"), "rb") as fh:
        loo = pickle.load(fh)["results"]
    out = {}
    for cfg, r in loo.items():
        yt = np.concatenate([c[0].ravel() for c in r["cells"].values()])
        pr = np.concatenate([c[1].ravel() for c in r["cells"].values()])
        sd = np.sqrt(np.concatenate([np.maximum(c[2], 1e-12).ravel()
                                     for c in r["cells"].values()]))
        out[cfg] = evaluate(yt, pr, sd)["rmse"]
        out[(cfg, "pp")] = {row["piece"]: row["rmse"] for row in r["per_piece"]}
    return out


def main() -> None:
    rmse = {}          # (config, hidden%) -> pooled rmse
    cells_at = {}      # (config, hidden%) -> cells (for pairing)
    for tag, hid in TAGS:
        for cfg, *_ in CONFIGS:
            cells = load_cells(tag, cfg)
            rmse[(cfg, hid)] = pooled(cells)["rmse"]
            cells_at[(cfg, hid)] = cells
    loo = loo_pooled()

    rng = np.random.default_rng(11)
    deltas = {"graph": [], "emb": []}   # (hidden, mu, lo, hi)
    for tag, hid in TAGS:
        for name, ca, cb in [("graph", "b_featlm", "b_featlm_nograph"),
                             ("emb", "b_featlm", "b_feat")]:
            pa = per_piece(cells_at[(ca, hid)], "rmse")
            pb = per_piece(cells_at[(cb, hid)], "rmse")
            d = np.array([pa[k] - pb[k] for k in sorted(set(pa) & set(pb))])
            deltas[name].append((hid, *bootstrap_ci(d, B=2000, rng=rng)))
    loo_delta = {}
    for name, ca, cb in [("graph", "b_featlm", "b_featlm_nograph"),
                         ("emb", "b_featlm", "b_feat")]:
        pa, pb = loo[(ca, "pp")], loo[(cb, "pp")]
        d = np.array([pa[k] - pb[k] for k in sorted(set(pa) & set(pb))])
        loo_delta[name] = bootstrap_ci(d, B=2000, rng=rng)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(10.2, 4.0), dpi=200,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28})
    for ax in (axA, axB):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    # ---- panel A ----------------------------------------------------------
    xs = [50, 40, 30, 20, 10]
    x_loo = 3.0                       # detached position past 10%
    for cfg, label, col, mark, ls in CONFIGS:
        ys = [rmse[(cfg, h)] for h in xs]
        axA.plot(xs, ys, ls, color=col, lw=2, marker=mark, ms=5.5,
                 markerfacecolor=col, markeredgecolor="white",
                 markeredgewidth=0.8, label=label)
        axA.plot([x_loo], [loo[cfg]], marker=mark, ms=6.5, color=col,
                 markerfacecolor="white", markeredgecolor=col,
                 markeredgewidth=1.6, linestyle="none")
    labpos = {"b_featlm": (33, 0.3525, "proposed model", "right"),
              "b_featlm_nograph": (37, 0.3790, "no graph", "left"),
              "b_feat": (16, 0.3706, "features only", "center")}
    for cfg, label, col, mark, ls in CONFIGS:
        x, y, txt, ha = labpos[cfg]
        axA.annotate(txt, xy=(x, y), fontsize=8.5, color=col, ha=ha, va="center")
    axA.axvline(6.5, color=MUTED, lw=0.6, ls=(0, (2, 3)))
    axA.set_xlim(54, 1.2)
    axA.set_xticks(xs + [x_loo])
    axA.set_xticklabels(["50%", "40%", "30%", "20%", "10%", "LOO"])
    axA.set_xlabel("fraction of notes hidden", fontsize=9, color=INK)
    axA.set_ylabel("held-out RMSE (pooled)", fontsize=9, color=INK)
    axA.set_title("Recovery vs. masking level (development set)",
                  fontsize=10, color=INK, loc="left")
    axA.annotate("open markers: LOO limit\n(hyperparams fit on full piece)",
                 xy=(x_loo, loo["b_feat"]), xytext=(52, 0.3315),
                 fontsize=7, color=MUTED, ha="left")

    # ---- panel B ----------------------------------------------------------
    off = {"graph": -0.8, "emb": +0.8}
    style = {"graph": ("#0072B2", "o", "graph value"),
             "emb": ("#D55E00", "D", "embedding value")}
    for name in ("graph", "emb"):
        col, mark, lab = style[name]
        hs = [d[0] + off[name] for d in deltas[name]]
        mus = [d[1] for d in deltas[name]]
        los = [d[1] - d[2] for d in deltas[name]]
        his = [d[3] - d[1] for d in deltas[name]]
        axB.errorbar(hs, mus, yerr=[los, his], fmt=mark, color=col, ms=5.5,
                     lw=1.6, capsize=2.5, markerfacecolor=col,
                     markeredgecolor="white", markeredgewidth=0.8, label=lab)
        mu, lo, hi = loo_delta[name]
        axB.errorbar([x_loo + off[name] * 0.5], [mu], yerr=[[mu - lo], [hi - mu]],
                     fmt=mark, color=col, ms=6.0, lw=1.6, capsize=2.5,
                     markerfacecolor="white", markeredgecolor=col,
                     markeredgewidth=1.4)
    axB.axhline(0.0, color=MUTED, lw=0.8)
    axB.axvline(6.5, color=MUTED, lw=0.6, ls=(0, (2, 3)))
    axB.set_xlim(54, 1.2)
    axB.set_xticks([h for _, h in TAGS] + [x_loo])
    axB.set_xticklabels(["50%", "40%", "30%", "20%", "10%", "LOO"])
    axB.set_xlabel("fraction of notes hidden", fontsize=9, color=INK)
    axB.set_ylabel(r"paired $\Delta$RMSE (95% CI)", fontsize=9, color=INK)
    axB.set_title("What each ingredient adds, by level",
                  fontsize=10, color=INK, loc="left")
    axB.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
