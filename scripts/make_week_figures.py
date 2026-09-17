#!/usr/bin/env python
"""Figures for the exploration-week findings (DEV ONLY, recorded numbers).

One figure per finding, all from the committed run pickles; the only
computation is one single-cell refit for the mechanism panel (via the
adopted core path `gp.noise_corr` / `gp.posterior_observations`).
House style: Okabe-Ito, legends above panels, muted grids, ink text.
Color roles are fixed across all five figures: BLUE = the new/robust/
correlated system, VERMILION = the published/baseline system, GREEN and
ORANGE = auxiliary series (runs), MUTED = guides.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_week_figures.py
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
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
FIGDIR = "docs/thesis/figures"
RHO_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def _recs(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += pickle.load(open(f, "rb"))
    return rows


def _cluster_ci(vals, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    uk = sorted(set(groups))
    per = {k: [] for k in uk}
    for v, g in zip(vals, groups):
        per[g].append(v)
    means = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uk), len(uk))
        means[b] = np.concatenate([per[uk[j]] for j in pick]).mean()
    return float(np.mean(vals)), *np.percentile(means, [2.5, 97.5])


# ------------------------------------------------------- fig 1: corrnoise
def fig_corrnoise():
    runs = [("seeds 0/1", "", BLUE, "o"), ("seeds 2/3", "_s23", GREEN, "s"),
            ("joint refit", "_joint", ORANGE, "D")]
    data = {}
    for label, tag, color, mk in runs:
        recs = _recs(f"results/corrnoise_tau/cells{tag}.shard*.pkl")
        data[label] = (recs, color, mk)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    # A: paired per-run deltas with cluster CIs, both metrics
    ax = axes[0]
    for j, met in ((0, "nll"), (1, "rmse")):
        for i, (label, (recs, color, mk)) in enumerate(data.items()):
            d = [r["corr"][met] - r["base"][met] for r in recs]
            g = [r["key"][0] for r in recs]
            m, lo, hi = _cluster_ci(d, g)
            y = j * 4 + (3 - i)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=mk, color=color,
                        ms=5, capsize=2.5, lw=1.4)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_ylim(0.2, 7.8)
    ax.set_yticks([2, 6])
    ax.set_yticklabels(["timing\nNLL", "timing\nRMSE (s)"], fontsize=8.5,
                       color=INK)
    ax.set_xlabel("paired delta, correlated minus diagonal")
    ax.set_title("A  both axes improve, three runs agree", loc="left")

    # B: evidence-chosen rho histogram, three runs
    ax = axes[1]
    w = 0.26
    for i, (label, (recs, color, mk)) in enumerate(data.items()):
        rhos = np.array([r["rho"] for r in recs])
        counts = [np.mean(rhos == r) for r in RHO_GRID]
        ax.bar(np.arange(len(RHO_GRID)) + (i - 1) * w, counts, width=w,
               color=color, label=label, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(RHO_GRID)))
    ax.set_xticklabels([f"{r:g}" for r in RHO_GRID])
    ax.set_xlabel(r"evidence-chosen correlation $\rho$")
    ax.set_ylabel("fraction of cells")
    ax.set_title("B  the correlation is detected, not assumed", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    # C: the mechanism on one real track (single-cell refit, core path)
    ax = axes[2]
    tgt, base_pred, corr_pred = _mechanism_cell()
    idx = np.arange(len(tgt[0]))
    ax.errorbar(idx - 0.15, base_pred[0], yerr=1.645 * base_pred[1],
                fmt="o", ms=3.5, color=VERM, capsize=1.5, lw=1,
                label="diagonal noise")
    ax.errorbar(idx + 0.15, corr_pred[0], yerr=1.645 * corr_pred[1],
                fmt="o", ms=3.5, color=BLUE, capsize=1.5, lw=1,
                label="correlated noise")
    ax.plot(idx, tgt[0], "x", ms=5, color=INK, label="held-out target",
            zorder=5)
    ax.set_xlabel("held-out note (score order)")
    ax.set_ylabel(r"timing $\tau$ (s)")
    ax.margins(y=0.14)
    ax.set_title("C  a neighbour's warp error informs the note", loc="left",
                 pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=7.5,
              ncol=3, columnspacing=1.0, handletextpad=0.4)

    fig.tight_layout()
    out = f"{FIGDIR}/corrnoise_tau_dev.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _mechanism_cell(key=(10, 2), seed=0):
    """One (track, seed) cell refit; held-out tau targets + both predictives."""
    from eval_phase2_real import C_MAX, CACHE, CH, HOLD_FRAC
    from score_bundle.baselines import rich_score_features
    from score_bundle.gp import MultiOutputGraphGP
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.score import Score

    d = pickle.load(open(CACHE, "rb"))[key]
    n_ch = len(CH)
    est = np.concatenate([d["est"], d["ell"][:, None], d["tau"][:, None],
                          d["dvib"][:, None]], axis=1)
    var = np.concatenate([d["var"], d["var_ell"][:, None],
                          d["var_tau"][:, None], d["var_dvib"][:, None]],
                         axis=1)
    n = est.shape[0]
    usable = np.isfinite(est[:, 0]) & (np.abs(est[:, 0]) <= C_MAX)
    score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                              np.zeros(n, dtype=int))
    eig = np.linalg.eigh(laplacian(build_adjacency(score)))
    X = rich_score_features(score, rff_dim=0)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
    scale = np.ones((n, n_ch))
    for c in range(n_ch):
        v = var[:, c]
        ok = np.isfinite(v)
        med = np.median(v[ok]) if ok.any() else 1.0
        scale[:, c] = np.where(np.isfinite(v),
                               np.clip(v / max(med, 1e-12), 1e-2, 1e3), 1.0)
    med_var = np.array([np.median(var[:, c][np.isfinite(var[:, c])])
                        if np.isfinite(var[:, c]).any() else 1.0
                        for c in range(n_ch)])
    rng = np.random.default_rng(1000 + 7 * key[0] + key[1] + seed)
    held = (rng.random(n) < HOLD_FRAC) & usable
    mask = np.zeros((n, n_ch), dtype=bool)
    mask[:, 0] = usable & ~held
    mask[:, 1] = mask[:, 2] = usable & ~held & d["ident"]
    mask[:, 3] = usable & ~held & np.isfinite(est[:, 3])
    mask[:, 4] = ~held & np.isfinite(est[:, 4])
    mask[:, 5] = ~held & np.isfinite(est[:, 5])
    Yobs = np.where(mask, np.nan_to_num(est), 0.0)
    floor = 0.05 * np.array([float(np.var(Yobs[mask[:, c], c]))
                             if mask[:, c].sum() > 2 else 1.0
                             for c in range(n_ch)])
    g = MultiOutputGraphGP(eig[0], eig[1], kernel="additive",
                           features=feats, n_channels=n_ch)
    g.noise_scale = scale
    x_hat, _ = g.fit(Yobs, mask, noise_floor=floor, maxiter=200,
                     noise_fixed=med_var)
    h_tau = (~mask[:, 4]) & held & np.isfinite(est[:, 4])

    def predict(rho):
        g.noise_corr = {"channel": 4, "rho": rho}
        M, S = g.posterior_observations(Yobs, mask, x_hat)
        return M[h_tau, 4], S[h_tau, 4]

    # rho by the observed-block profile (as in the study)
    best = (-np.inf, 0.0)
    for rho in RHO_GRID:
        g.noise_corr = {"channel": 4, "rho": rho}
        lml = g.log_marginal_likelihood(Yobs, mask, x_hat)
        if lml > best[0]:
            best = (lml, rho)
    m0 = predict(0.0)
    m1 = predict(best[1])
    order = np.argsort(d["onset"][h_tau], kind="stable")
    take = order[:18]                          # first 18 held-out notes shown
    return ((est[h_tau, 4][take],), (m0[0][take], m0[1][take]),
            (m1[0][take], m1[1][take]))


# ------------------------------------------------- fig 2: the timing tail
def fig_tail():
    from eval_tail_predictive import load_cells, variants

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    cells = load_cells("results/graphgp_masksweep/obs0.70/b_featlm.shard*.pkl")
    per = {"gauss": [], "t5": []}
    for key, (yt, pr, sd, ch) in sorted(cells.items()):
        tau = ch == 0
        v = variants(yt[tau], pr[tau], sd[tau])
        for name in per:
            per[name].append(float(v[name][0].mean()))
    ax = axes[0]
    for name, color, lab in (("gauss", VERM, "Gaussian predictive"),
                             ("t5", BLUE, "Student-t predictive")):
        ax.plot(np.sort(per[name]), color=color, lw=1.6, label=lab)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("cells, sorted by own NLL (30% hidden)")
    ax.set_ylabel("per-cell timing NLL")
    ax.set_title("A  one cell explodes the Gaussian score", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    sets = [("40%", "results/graphgp_v2/b_featlm.shard*.pkl"),
            ("50%", "results/graphgp_masksweep/obs0.50/b_featlm.shard*.pkl"),
            ("30%", "results/graphgp_masksweep/obs0.70/b_featlm.shard*.pkl"),
            ("20%", "results/graphgp_masksweep/obs0.80/b_featlm.shard*.pkl"),
            ("10%", "results/graphgp_masksweep/obs0.90/b_featlm.shard*.pkl")]
    worst = {"gauss": [], "t5": []}
    labels = []
    for lab, pattern in sets:
        cells = load_cells(pattern)
        w = {"gauss": -np.inf, "t5": -np.inf}
        for key, (yt, pr, sd, ch) in cells.items():
            tau = ch == 0
            v = variants(yt[tau], pr[tau], sd[tau])
            for name in w:
                w[name] = max(w[name], float(v[name][0].mean()))
        labels.append(lab)
        for name in worst:
            worst[name].append(w[name])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, worst["gauss"], width=0.36, color=VERM,
           label="Gaussian predictive")
    ax.bar(x + 0.18, worst["t5"], width=0.36, color=BLUE,
           label="Student-t predictive")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("fraction of notes hidden")
    ax.set_ylabel("worst-cell timing NLL")
    ax.set_title("B  the tail is gone at every masking level", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = f"{FIGDIR}/tail_predictive_dev.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------- fig 3: the t-EM spillover
def fig_tem():
    recs = _recs("results/t_noise_em/cells.shard*.pkl")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))

    ax = axes[0]
    d = np.sort(np.array([r["tem"][1]["rmse"] - r["gauss"][1]["rmse"]
                          for r in recs]))
    ax.bar(np.arange(d.size), d, width=1.0,
           color=np.where(d < 0, BLUE, VERM).tolist())
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(d.mean(), color=INK, lw=1.1, ls="--")
    ax.text(d.size - 2, d.mean() - 0.003, f"mean {d.mean():+.3f}",
            fontsize=8, color=INK, va="top", ha="right")
    ax.set_xlabel("cells, sorted")
    ax.set_ylabel("articulation RMSE delta\n(t-EM minus Gaussian)")
    ax.set_title("A  robust timing noise helps articulation", loc="left")

    ax = axes[1]
    cg = np.array([r["gauss"][0]["cov"] for r in recs])
    ct = np.array([r["tem"][0]["cov"] for r in recs])
    ax.plot([0.7, 1.0], [0.7, 1.0], color=MUTED, lw=0.9, ls="--")
    ax.axhline(0.90, color=GREEN, lw=1.0)
    ax.axvline(0.90, color=GREEN, lw=1.0)
    ax.text(0.705, 0.905, "nominal 0.90", fontsize=7.5, color=GREEN)
    ax.plot(cg, ct, "o", ms=3.5, color=BLUE, alpha=0.7)
    ax.set_xlabel("timing coverage, Gaussian fit")
    ax.set_ylabel("timing coverage, t-EM fit")
    ax.set_xlim(0.7, 1.02)
    ax.set_ylim(0.7, 1.02)
    ax.set_title("B  coverage moves from padded to nominal", loc="left")

    fig.tight_layout()
    out = f"{FIGDIR}/t_noise_em_dev.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------- fig 4: the completion guard
def fig_completion():
    rows = {}
    for f in sorted(glob.glob("results/completion_fallback/rows_*.pkl")):
        rows.update(pickle.load(open(f, "rb")))
    settings = [("prefix", 0.25), ("prefix", 0.5), ("prefix", 0.75),
                ("random", 0.25), ("random", 0.5), ("random", 0.75)]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0))

    ax = axes[0]
    systems = [("GP-featlm", VERM, "per-piece model alone"),
               ("either q=0.99", BLUE, "with the combined guard"),
               ("head only", MUTED, "cross-piece head")]
    x = np.arange(len(settings))
    for i, (name, color, lab) in enumerate(systems):
        w = [max(v["rmse"] for v in rows[(k, f, name)])
             for k, f in settings]
        ax.bar(x + (i - 1) * 0.27, w, width=0.27, color=color, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k[:4]} {f:g}" for k, f in settings], fontsize=8)
    ax.set_ylabel("worst-piece RMSE (log)")
    ax.set_title("A  every blow-up tamed, nothing given away", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[1]
    for i, (name, color, lab) in enumerate(
            (("flagfrac q=0.99", BLUE, "coverage test (per note)"),
             ("guardfrac", ORANGE, "disagreement guard (per cell)"))):
        v = [float(np.mean(rows[(k, f, name)])) for k, f in settings]
        ax.bar(x + (i - 0.5) * 0.36, v, width=0.36, color=color, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k[:4]} {f:g}" for k, f in settings], fontsize=8)
    ax.set_ylabel("mean flag rate")
    ax.set_title("B  the rules fire where they should", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout()
    out = f"{FIGDIR}/completion_fallback_dev.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------- fig 5: the kernel question closed
def fig_kernels():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    # A: kill test, median |err| ratio SM / sine per quantity
    rows = _recs(".cache/sm_dev_shard_*_4.pkl")
    both = [r for r in rows if r.get("ident_tr") and r.get("ident_gt")
            and "sm_tr" in r and "sm_gt" in r
            and np.isfinite(r["nl_tr"]["lf"]) and np.isfinite(r["nl_gt"]["lf"])]
    ax = axes[0]
    ratios = []
    for q in ("c", "lg", "lf"):
        e_sm = np.median([abs(r["sm_tr"][q] - r["sm_gt"][q]) for r in both])
        e_nl = np.median([abs(r["nl_tr"][q] - r["nl_gt"][q]) for r in both])
        ratios.append(e_sm / e_nl)
    ax.bar(range(3), ratios, width=0.55, color=VERM)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.text(-0.4, 1.08, "parity", fontsize=7.5, color=MUTED, ha="left",
            va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["intonation", "extent", "rate"], fontsize=8.5)
    ax.set_ylabel("median error ratio, GP / sine fit")
    ax.set_title("A  slot 1: the estimator swap loses", loc="left")

    # B: learned filter amplitude histogram
    cells = {}
    for f in sorted(glob.glob("results/spectral_bump/cells.shard*.pkl")):
        cells.update(pickle.load(open(f, "rb")))
    amps = np.array([v["bump"]["a"] for v in cells.values()])
    ax = axes[1]
    ax.hist(np.clip(amps, 0, 0.12), bins=24, color=BLUE)
    ax.axvline(0.01, color=MUTED, lw=1.0, ls="--")
    ax.text(0.012, ax.get_ylim()[1] * 0.9,
            f"off (<0.01): {np.mean(amps < 0.01):.0%}", fontsize=8, color=INK)
    ax.set_xlabel("evidence-fitted bump amplitude (clipped at 0.12)")
    ax.set_ylabel("cells")
    ax.set_title("B  a free filter, switched off", loc="left")

    # C: Slot B per-family paired deltas
    rows = _recs("results/phase3_smprior/cells.shard*.pkl")
    fams = [("strings", ("vn", "va", "vc", "db")),
            ("winds", ("fl", "cl", "ob", "sax", "bn")),
            ("brass", ("tpt", "hn", "tbn", "tba")), ("all", None)]
    ax = axes[2]
    rng = np.random.default_rng(0)
    for i, (fam, members) in enumerate(fams):
        d = np.array([abs(r["sm"][0] - r["gt_c"])
                      - abs(r["bump"][0] - r["gt_c"]) for r in rows
                      if members is None or r["instr"] in members])
        idx = rng.integers(0, d.size, size=(2000, d.size))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        ax.errorbar(d.mean(), 3 - i, xerr=[[d.mean() - lo], [hi - d.mean()]],
                    fmt="o", ms=5, color=BLUE if fam != "all" else INK,
                    capsize=2.5, lw=1.4)
    ax.axvline(0, color=MUTED, lw=1.0, ls="--")
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([f[0] for f in fams], fontsize=8.5)
    ax.set_xlabel("error delta, SM minus bumps (cents)")
    ax.set_title("C  slot 2: no gain at the waveform", loc="left")

    fig.tight_layout()
    out = f"{FIGDIR}/kernels_closed_dev.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_corrnoise()
    fig_tail()
    fig_tem()
    fig_completion()
    fig_kernels()
