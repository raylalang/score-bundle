#!/usr/bin/env python
"""House-style figures for the three spans of the score-prior-in-waveform
studies (2026-10-15..20), drawn from the stored cells only:

  span_schematic.png     the experiment as a system: inputs, the three priors,
                         the hidden element, the common-mode scalar, outputs
  span_protocol.png      one chain, end to end: waveform + mask, the three
                         arms' per-note posteriors against the reference
  span_leakage.png       why the first run was withdrawn: the leaked
                         likelihood well along the hidden c, and masked
                         estimates vs truth before/after the fix
  span_commonmode.png    the tuning scalar: per-cell error without/with it
  span_contrasts.png     paired contrasts with bootstrap CIs, three spans
  span_calibration.png   z-scores of the hidden element's band, by span/arm
  span_boundary.png      hidden-joint error: tempo line vs the three priors
  span_phrase_dose.png   phrase scale: graph-vs-temporal gain against the
                         recurrence edge weight (the mechanism test)

Okabe-Ito house style: BLUE = score graph (this model), ORANGE = temporal
chain, VERM = independent (the reference arm), grey = analytic references,
black = truth; dpi 200; one axis per panel; legends always present.

    PYTHONPATH=src:scripts python scripts/make_span_viz.py [--no-profile]
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

BLUE, ORANGE, VERM, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"
GREY, INK = "#8C8C8C", "#222222"
ARM_COL = {"indep": VERM, "temporal": ORANGE, "graph": BLUE}
ARM_NAME = {"indep": "independent", "temporal": "temporal chain",
            "graph": "score graph"}
ARMS = ("indep", "temporal", "graph")
FIGDIR = "docs/thesis/figures"
CELLS = "results/phase3_cells"
Z90 = 1.6448536269514722
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 200, "figure.dpi": 100})


def load(prefix):
    rows = []
    for f in sorted(glob.glob(f"{CELLS}/{prefix}.shard*.pkl")):
        rows.extend(pickle.load(open(f, "rb")))
    return {(r["key"], tuple(r["idx"])): r for r in rows}


def paired_ci(d, seed=31, n=2000):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, d.size).mean() for _ in range(n)])
    return d.mean(), np.quantile(bs, .025), np.quantile(bs, .975)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- schematic
def fig_schematic():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    titles = ["independent", "temporal chain", "score graph"]
    cols = [VERM, ORANGE, BLUE]
    # a 6-note phrase: pitches chosen so note 3 (hidden) recurs at note 5
    pitch = np.array([0, 2, 4, 2, 0, 4])
    xs = np.arange(6)
    hidden = 2
    for ax, title, col, arm in zip(axes, titles, cols, ARMS):
        ax.set_title(f"{title}", color=col, fontsize=10, loc="left")
        ax.set_xlim(-0.7, 5.7)
        ax.set_ylim(-1.6, 5.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"n{k + 1}" for k in xs])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        # edges
        if arm == "temporal":
            for k in range(5):
                ax.plot([k, k + 1], [pitch[k], pitch[k + 1]], color=col, lw=1.6)
        elif arm == "graph":
            for i in range(6):
                for j in range(i + 1, 6):
                    w = np.exp(-((i - j) * 0.5) ** 2 / 8.0 - (pitch[i] - pitch[j]) ** 2 / 32.0)
                    if w > 0.25:
                        ax.plot([i, j], [pitch[i], pitch[j]], color=col,
                                lw=0.6 + 2.2 * w, alpha=0.35 + 0.6 * w,
                                solid_capstyle="round")
        # common-mode scalar: a bar under all notes
        ax.add_patch(Rectangle((-0.45, -1.3), 5.9, 0.5, facecolor="#F2F2F2",
                               edgecolor=GREY, lw=0.8))
        ax.text(2.5, -1.05, "+ shared scalar s ~ N(0, broad)", ha="center",
                va="center", fontsize=7.5, color=INK)
        # nodes
        for k in xs:
            face = "white" if k == hidden else col
            edge = col
            ax.scatter([k], [pitch[k]], s=170, facecolor=face, edgecolor=edge,
                       lw=1.6, zorder=5)
            if k == hidden:
                ax.annotate("hidden: audio removed,\ncolumns dropped",
                            (k, pitch[k]), (k + 0.35, pitch[k] + 1.1),
                            fontsize=7, ha="left", color=INK,
                            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
        ax.text(-0.6, 5.3, "pitch", fontsize=7.5, color=GREY, va="top")
        ax.set_xlabel("score time", fontsize=7.5, color=GREY)
    fig.suptitle("Three cross-note priors over one latent per note (intonation constant c; "
                 "per joint: boundary offset db), matched marginal variance,\nsame likelihood, "
                 "data and optimizer.  Output: posterior mean and sd of the hidden element.",
                 fontsize=8.5, y=1.06)
    save(fig, "span_schematic.png")


# ---------------------------------------------------------------- protocol
def fig_protocol():
    os.environ["PRIORVAL_DROP_HIDDEN"] = "1"
    os.environ["PRIORVAL_TUNING_SD"] = "50"
    import eval_prior_value_dev as pv
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    run = load("priorval_v2_tuning")
    for gidx, ch in pv.iter_cells():
        break
    r = run[(ch["key"], tuple(ch["idx"]))]
    gt_all = data[ch["key"]]["est_gt"][ch["idx"], 0]
    x, t, mask, j = ch["x"], ch["t"], ch["mask"], ch["j"]
    cuts = np.concatenate([[0.0], ch["b_annot"], [t[-1]]])
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 5.8),
                               gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.55})
    # A: waveform with note regions and mask
    step = max(1, x.size // 6000)
    a.plot(t[::step], x[::step], color=GREY, lw=0.5)
    for k in range(len(cuts) - 1):
        a.axvspan(cuts[k], cuts[k + 1], color=BLUE if k != j else "white",
                  alpha=0.07 if k != j else 0.0)
        a.text(0.5 * (cuts[k] + cuts[k + 1]), a.get_ylim()[1] * 0.92 if False else 0,
               "", fontsize=7)
    lo, hi = t[~mask][0], t[~mask][-1]
    a.axvspan(lo, hi, facecolor="none", edgecolor=VERM, hatch="///", lw=0.0, alpha=0.6)
    ymax = np.max(np.abs(x)) * 1.1
    a.set_ylim(-ymax, ymax)
    for k in range(len(cuts) - 1):
        a.text(0.5 * (cuts[k] + cuts[k + 1]), ymax * 0.82,
               f"MIDI {int(ch['midis'][k])}" + ("\n(hidden)" if k == j else ""),
               ha="center", fontsize=7.5, color=VERM if k == j else INK)
    a.set_xlabel("segment time (s)")
    a.set_ylabel("audio")
    a.set_yticks([])
    a.set_title(f"A. One chain, track {ch['key']} ({ch['instr']}): the middle note's samples "
                "masked (hatched), its design columns dropped", loc="left", fontsize=8.5)
    # B: per-note c posteriors
    k_n = len(ch["idx"])
    xs = np.arange(k_n)
    off = {"indep": -0.22, "temporal": 0.0, "graph": 0.22}
    for arm in ARMS:
        c_all = r[f"masked_{arm}"]["c_all"]
        sd_all = r[f"masked_{arm}"]["sd_all"]
        b.errorbar(xs + off[arm], c_all, yerr=Z90 * sd_all, fmt="o", ms=5,
                   color=ARM_COL[arm], ecolor=ARM_COL[arm], elinewidth=1.2,
                   capsize=2.5, label=f"{ARM_NAME[arm]} (masked fit, 90% band)")
    b.plot(xs, gt_all, "k_", ms=18, mew=1.6, label="reference centre", zorder=6)
    b.axvspan(j - 0.5, j + 0.5, color=VERM, alpha=0.06)
    b.set_xticks(xs)
    b.set_xticklabels([f"note {k + 1}\nMIDI {int(m)}" for k, m in enumerate(ch["midis"])])
    b.set_ylabel("intonation constant c (cents)")
    b.set_title("B. Per-note posteriors: observed notes pinned by the audio, the hidden note "
                "= the prior conditional\n(free tuning scalar in every arm)", loc="left", fontsize=8.5)
    b.legend(loc="upper left", fontsize=7.5, ncol=2)
    save(fig, "span_protocol.png")


# ---------------------------------------------------------------- leakage
def fig_leakage(profile=True):
    first = load("priorval")
    v2 = load("priorval_v2")
    keys = [k for k in first if k in v2]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"wspace": 0.32})
    # A: the profile (recomputed on cell 0 if requested)
    a = axes[0]
    if profile:
        prof = _leak_profile()
        if prof is not None:
            grid, nll, pr = prof
            a.plot(grid, nll, color=VERM, lw=1.8, label="leaked −log-likelihood (first run)")
            a.plot(grid, pr, color=GREY, lw=1.8, ls="--", label="prior term (zero-mean, var_c)")
            a.plot(grid, nll + pr, color=INK, lw=1.2, label="total")
            a.axvline(29.5, color="k", lw=0.8, ls=":")
            a.text(29.5, a.get_ylim()[1] * 0.95, " truth", fontsize=7.5, va="top")
            a.set_xlabel("hidden note's c (cents)")
            a.set_ylabel("nats (relative to c = 0)")
            a.set_title("A. Cell 0: the leak carves a 38-nat well,\nthe prior spans 4–35 nats", loc="left")
            a.legend(fontsize=7, loc="upper right")
    else:
        a.text(0.5, 0.5, "profile skipped", ha="center", transform=a.transAxes)
    # B / C: masked estimate vs truth, first run vs v2 (graph arm, all arms light)
    for ax, run, name in ((axes[1], first, "B. First run: samples masked,\ncolumns kept"),
                          (axes[2], v2, "C. Protocol v2: hidden note\ndropped from the design")):
        gt = np.array([run[k]["gt_c"] for k in keys])
        for arm in ARMS:
            c = np.array([run[k][f"masked_{arm}"]["c"] for k in keys])
            r_ = np.corrcoef(c, gt)[0, 1] if np.std(c) > 0 else None
            lab = (f"{ARM_NAME[arm]}  (r = {r_:+.2f})" if r_ is not None
                   else f"{ARM_NAME[arm]}  (exactly the prior mean 0)")
            ax.scatter(gt, c, s=16, color=ARM_COL[arm], alpha=0.75, label=lab)
        lim = 95
        ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.7, ls=":")
        ax.set_xlim(-55, 55)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("reference c of the hidden note (cents)")
        ax.set_ylabel("masked estimate (cents)")
        ax.set_title(name, loc="left")
        ax.legend(fontsize=7, loc="upper left")
    save(fig, "span_leakage.png")


def _leak_profile():
    try:
        import torch
        os.environ.pop("PRIORVAL_DROP_HIDDEN", None)
        os.environ.pop("PRIORVAL_TUNING_SD", None)
        import importlib
        import eval_prior_value_dev as pv
        importlib.reload(pv)
        from eval_phase3_curve_dev import load_hypers
        from score_bundle.phase3.curve import (chain_reference, fit_chain,
                                               collapsed_loglik_torch,
                                               design_support_torch,
                                               interp_knots, knot_gram,
                                               CurvePrior)
    except Exception as e:                      # pragma: no cover
        print("profile skipped:", e)
        return None
    hypers = load_hypers()
    for gidx, ch in pv.iter_cells():
        break
    mask, midis, j = ch["mask"], ch["midis"], ch["j"]
    c0, nv = chain_reference(ch["x"], ch["t"], midis, ch["b_annot"], sample_mask=mask)
    K_c = pv.arm_cov("indep", midis, ch["onsets"], ch["durs"], ch["var_c"])
    fit = fit_chain(ch["x"], ch["t"], midis, ch["b_annot"], hypers=hypers, c_init=c0,
                    noise_var=nv, c_prior_cov=K_c, sample_mask=mask,
                    fix_boundaries=True, n_steps=pv.N_STEPS)
    x_t = torch.as_tensor(np.asarray(ch["x"], float)); t_np = np.asarray(ch["t"], float)
    t_t = torch.as_tensor(t_np); m = t_np.size; k_n = len(midis); prior = CurvePrior()
    b_lo = torch.as_tensor(fit.b_lo); b_hi = torch.as_tensor(fit.b_hi); msk = torch.as_tensor(mask)
    chol_c = torch.linalg.cholesky(torch.as_tensor(K_c))
    knots_t, slices = [], []
    for k in range(k_n):
        kt = torch.as_tensor(fit.knots[k]); knots_t.append(kt)
        lo, hi = fit.knots[k][0], fit.knots[k][-1]
        slices.append((int(np.searchsorted(t_np, max(t_np[0], lo - 0.2))),
                       int(np.searchsorted(t_np, min(t_np[-1], hi + 0.2)))))
    us = [torch.as_tensor(fit.us[k]) for k in range(k_n)]

    def parts(c_vec):
        cols = []
        for k in range(k_n):
            ia, ib = slices[k]; ts = t_t[ia:ib]
            cents = c_vec[k] + interp_knots(us[k], knots_t[k], ts)
            Phi_k = design_support_torch(cents, ts, float(midis[k]), ia, m, 8, 2)
            w = torch.ones(m, dtype=torch.float64)
            if k > 0:
                w = w * torch.sigmoid((t_t - b_lo[k - 1]) / 0.010)
            if k < k_n - 1:
                w = w * torch.sigmoid((b_hi[k] - t_t) / 0.010)
            cols.append(Phi_k * w.unsqueeze(1))
        Phi = torch.cat(cols, dim=1)
        ll = collapsed_loglik_torch(x_t[msk], Phi[msk], fit.noise_var, 10.0)
        pr = 0.5 * (c_vec @ torch.cholesky_solve(c_vec.unsqueeze(1), chol_c).squeeze(1))
        return float(-ll), float(pr)
    base = torch.as_tensor(fit.c.copy())
    grid = np.arange(-100, 101, 5.0)
    nll, pr = [], []
    with torch.no_grad():
        for c in grid:
            cv = base.clone(); cv[j] = c
            a_, b_ = parts(cv); nll.append(a_); pr.append(b_)
    nll = np.array(nll); pr = np.array(pr)
    nll0 = nll[len(grid) // 2]
    return grid, nll - nll0, pr


# ---------------------------------------------------------------- common mode
def fig_commonmode():
    v2 = load("priorval_v2"); v2t = load("priorval_v2_tuning")
    keys = [k for k in v2 if k in v2t]
    err = lambda run, k, arm: abs(run[k][f"masked_{arm}"]["c"] - run[k]["gt_c"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, arm in zip(axes, ("indep", "graph")):
        e0 = np.array([err(v2, k, arm) for k in keys]); e1 = np.array([err(v2t, k, arm) for k in keys])
        order = np.argsort(e0)
        ys = np.arange(len(keys))
        for y, i in zip(ys, order):
            ax.plot([e0[i], e1[i]], [y, y], color="#CCCCCC", lw=0.9, zorder=1)
        ax.scatter(e0[order], ys, s=14, color=GREY, label="without the tuning scalar", zorder=3)
        ax.scatter(e1[order], ys, s=14, color=ARM_COL[arm], label="with the tuning scalar", zorder=4)
        m, lo, hi = paired_ci(e1 - e0)
        ax.set_title(f"{ARM_NAME[arm]}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] cents",
                     loc="left", color=ARM_COL[arm], fontsize=9)
        ax.set_xlabel("hidden-note error (cents)")
        ax.set_yticks([])
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("47 chains, sorted by the no-scalar error")
    fig.suptitle("Adding the free tuning scalar: paired change in hidden-note error (intonation, chain scale).\n"
                 "The independent prior gains the chain mean; the score graph barely moves — its gain was the common offset.",
                 fontsize=9, y=1.06)
    save(fig, "span_commonmode.png")


# ---------------------------------------------------------------- contrasts
def fig_contrasts():
    spans = [("intonation, chain scale\n(47 chains, cents)", load("priorval_v2_tuning"),
              lambda r, a: abs(r[f"masked_{a}"]["c"] - r["gt_c"]), None),
             ("boundaries, chain scale\n(40 non-unison joints, ms)", load("boundprior"),
              lambda r, a: abs(r[f"masked_{a}"]["err"]), lambda r: not r["unison"]),
             ("intonation, phrase scale\n(48 phrases, cents)", load("priorval_phrase"),
              lambda r, a: abs(r[f"masked_{a}"]["c"] - r["gt_c"]), None)]
    contrasts = [("graph", "indep"), ("temporal", "indep"), ("graph", "temporal")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, (title, run, err, sel) in zip(axes, spans):
        rows = [r for r in run.values() if (sel is None or sel(r))]
        ys = np.arange(len(contrasts))[::-1]
        for y, (a, b) in zip(ys, contrasts):
            d = np.array([err(r, a) - err(r, b) for r in rows])
            m, lo, hi = paired_ci(d)
            col = ARM_COL[a]
            star = (hi < 0 or lo > 0)
            ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="butt")
            ax.scatter([m], [y], s=46, color=col, edgecolor="white", lw=1.2, zorder=5)
            ax.text(m, y + 0.2, f"{m:+.2f}{'*' if star else ''}", fontsize=7.5,
                    color=INK, ha="center")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{ARM_NAME[a]} − {ARM_NAME[b]}" for a, b in contrasts], fontsize=8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_ylim(-0.6, len(contrasts) - 0.4)
    fig.supxlabel("paired difference in the hidden element's error (negative = first arm better); "
                  "* = 95% interval excludes zero", fontsize=8.5)
    fig.suptitle("Paired contrasts with 95% bootstrap intervals; every arm carries the free common-mode scalar",
                 fontsize=9, y=1.03)
    save(fig, "span_contrasts.png")


# ---------------------------------------------------------------- calibration
def fig_calibration():
    v2t = load("priorval_v2_tuning"); bp = load("boundprior"); ph = load("priorval_phrase")
    groups = []
    for arm in ARMS:
        groups.append((f"intonation\nchain\n{arm}", ARM_COL[arm],
                       np.array([(r["masked_" + arm]["c"] - r["gt_c"]) / r["masked_" + arm]["sd"]
                                 for r in v2t.values()])))
    for arm in ARMS:
        groups.append((f"intonation\nphrase\n{arm}", ARM_COL[arm],
                       np.array([(r["masked_" + arm]["c"] - r["gt_c"]) / r["masked_" + arm]["sd"]
                                 for r in ph.values()])))
    for arm in ARMS:
        groups.append((f"boundary\nhidden\n{arm}", ARM_COL[arm],
                       np.array([r["masked_" + arm]["err"] / r["masked_" + arm]["sd"]
                                 for r in bp.values() if not r["unison"]])))
    groups.append(("boundary\nclean, observed\n(graph arm)", GREY,
                   np.array([r["clean_graph"]["err"] / r["clean_graph"]["sd"]
                             for r in bp.values() if not r["unison"]])))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    rng = np.random.default_rng(0)
    for i, (name, col, z) in enumerate(groups):
        zc = np.clip(z, -8, 8)
        ax.scatter(i + rng.uniform(-0.18, 0.18, zc.size), zc, s=10, color=col, alpha=0.7)
        cov = np.mean(np.abs(z) <= Z90)
        ax.text(i, 8.6, f"cov {cov:.2f}", ha="center", fontsize=7.5, color=INK)
        n_clip = np.sum(np.abs(z) > 8)
        if n_clip:
            ax.text(i, -8.7, f"{n_clip} beyond ±8", ha="center", fontsize=6.5, color=GREY)
    ax.axhspan(-Z90, Z90, color=GREEN, alpha=0.08, lw=0)
    ax.axhline(Z90, color=GREEN, lw=0.8); ax.axhline(-Z90, color=GREEN, lw=0.8)
    ax.text(len(groups) - 0.5, Z90 + 0.15, "nominal 90% band", fontsize=7.5, color=GREEN, ha="right")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylim(-9.4, 9.4)
    ax.set_ylabel("z = (estimate − truth) / posterior sd")
    ax.set_title("Standardized errors of the hidden element's band by span and prior (masked), and of the "
                 "observed joints on clean audio:\nhonest for intonation at both spans, too wide for hidden "
                 "joints, far too narrow for observed joints (the estimand gap at the boundary)",
                 loc="left", fontsize=8.5)
    save(fig, "span_calibration.png")


# ---------------------------------------------------------------- boundary
def fig_boundary():
    bp = load("boundprior")
    rows = [r for r in bp.values() if not r["unison"]]
    dead = np.array([abs(r["tau_ms"]) for r in rows])
    order = np.argsort(dead)
    ys = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for y, i in zip(ys, order):
        e = [dead[i]] + [abs(rows[i][f"masked_{a}"]["err"]) for a in ARMS]
        ax.plot([min(e), max(e)], [y, y], color="#DDDDDD", lw=0.8, zorder=1)
    ax.scatter(dead[order], ys, s=16, color=GREY, marker="s", label="tempo line (no prior transfer)", zorder=3)
    for arm in ARMS:
        e = np.array([abs(rows[i][f"masked_{arm}"]["err"]) for i in order])
        ax.scatter(e, ys, s=14, color=ARM_COL[arm], label=f"{ARM_NAME[arm]} prior (with scalar)", zorder=4)
    ax.set_xlim(0, 160)
    ax.set_yticks([])
    ax.set_ylabel("40 non-unison hidden joints, sorted by tempo-line error")
    ax.set_xlabel("hidden-joint error against the annotated onset (ms)")
    meds = {a: np.median([abs(r[f"masked_{a}"]["err"]) for r in rows]) for a in ARMS}
    ax.set_title("Boundaries at chain scale: every prior beats the tempo line (median "
                 f"{np.median(dead):.1f} ms) through the common offset\n(medians: independent "
                 f"{meds['indep']:.1f}, temporal chain {meds['temporal']:.1f}, score graph "
                 f"{meds['graph']:.1f} ms), and the three priors tie", loc="left", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "span_boundary.png")


# ---------------------------------------------------------------- phrase dose
def fig_phrase_dose():
    ph = load("priorval_phrase")
    rows = list(ph.values())
    err = lambda r, a: abs(r[f"masked_{a}"]["c"] - r["gt_c"])
    w = np.array([r["edge_w"] for r in rows])
    d_gt = np.array([err(r, "graph") - err(r, "temporal") for r in rows])
    d_gi = np.array([err(r, "graph") - err(r, "indep") for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"wspace": 0.3})
    for ax, d, name, col in ((a, d_gt, "score graph − temporal chain", ORANGE),
                             (b, d_gi, "score graph − independent", VERM)):
        ax.scatter(w, d, s=18, color=BLUE, alpha=0.8, label="one phrase cell")
        ax.axhline(0, color="k", lw=0.8)
        r_ = np.corrcoef(w, d)[0, 1]
        m, lo, hi = paired_ci(d)
        ax.set_title(f"{name}\nmean {m:+.2f} [{lo:+.2f}, {hi:+.2f}] cents;  r with edge weight = {r_:+.2f}",
                     loc="left", fontsize=8.5)
        ax.set_xlabel("edge weight, hidden note to its nearest recurrence")
        ax.set_ylabel("error difference (cents; negative = graph better)")
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Phrase scale: the graph's gain over the temporal chain does not grow with the recurrence edge "
                 "(no dose-response);\nagainst the common-mode baseline there is no gain", fontsize=9, y=1.05)
    save(fig, "span_phrase_dose.png")


if __name__ == "__main__":
    profile = "--no-profile" not in sys.argv
    fig_schematic()
    fig_contrasts()
    fig_commonmode()
    fig_calibration()
    fig_boundary()
    fig_phrase_dose()
    fig_protocol()
    fig_leakage(profile=profile)
