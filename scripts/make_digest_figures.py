#!/usr/bin/env python
"""Render the three meeting-digest figures into docs/thesis/figures/.

Reads numbers from the eval logs (never hard-code a metric that exists in a log):

  A  digest_headline.png   dumbbell, RMSE graph off -> on per prior mean
                           (logs/feature_baseline_l2_10.log pooled table)
  B  digest_channels.png   diverging bars, music model vs features per channel, mean-only RMSE
                           (same log, per-channel tables)
  C  digest_collapse.png   strip of 120 per-cell tau RMSE, unguarded vs guarded
                           (logs/diag_tau_s2x3.log, logs/diag_tau_s2x3_guarded.log)

Palette: dataviz reference instance (validated); light mode, matplotlib.
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; SURFACE = "#fcfcfb"
BLUE = "#2a78d6"; BLUE_LT = "#86b6ef"; BLUE_DK = "#1c5cab"
AQUA = "#1baf7a"; RED = "#d03b3b"; DIV_RED = "#e34948"
OUT = "docs/thesis/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 150,
})


def _style(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def read_pooled(path):
    """{(mean, on|off): (rmse, nll)} from the pooled table of the grid eval log."""
    txt = open(path).read()
    block = txt.split("Pooled (")[1].split("\n\n")[0]
    out = {}
    for line in block.splitlines():
        m = re.match(r"(\S+)\s+(on|off)\s+([-\d.]+)\s+([-\d.]+)", line.strip())
        if m:
            out[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return out


def read_channel(path, ch):
    txt = open(path).read()
    block = txt.split(f"[{ch}]")[1].split("\n\n")[0]
    out = {}
    for line in block.splitlines():
        m = re.match(r"(\S+)\s+(on|off)\s+([-\d.]+)", line.strip())
        if m:
            out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def read_cells(path):
    """[(seed, piece, tau_rmse)] from a diag log's per-cell lines."""
    cells = []
    for line in open(path):
        m = re.match(r"\s+s(\d) p\s*(\d+) tauRMSE\s+([\d.]+)", line)
        if m:
            cells.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    return cells


def read_ci_rows(path):
    """{row name: rmse} from logs/headline_cis.log."""
    out = {}
    for line in open(path):
        m = re.match(r"(.+?)\s{2,}([\d.]+) \[", line.rstrip())
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def read_kernel_block(path, block):
    """{kernel: (rmse, nll, drmse, dlo, dhi, dsig, dnll, nlo, nhi, nsig)} from a
    kernel report log; ``block`` is 'mu_LM' or 'mu = 0'."""
    txt = open(path).read()
    key = "mu = mu_LM" if block == "mu_LM" else "mu = 0"
    seg = txt.split(f"===== {key}")[1]
    seg = seg.split("\n", 1)[1]          # drop the rest of the header line
    seg = seg.split("\n=====")[0]        # stop at the next block
    out = {}
    pat = re.compile(
        r"^(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+"
        r"([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\](\*?)\s+"
        r"([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\](\*?)")
    for line in seg.splitlines():
        m = pat.match(line.strip())
        if m:
            g = m.groups()
            out[g[0]] = (float(g[1]), float(g[2]), float(g[3]), float(g[4]),
                         float(g[5]), g[6] == "*", float(g[7]), float(g[8]),
                         float(g[9]), g[10] == "*")
        elif re.match(r"^(additive)\s+([-\d.]+)\s+([-\d.]+)", line.strip()):
            m2 = re.match(r"^additive\s+([-\d.]+)\s+([-\d.]+)", line.strip())
            out["additive"] = (float(m2.group(1)), float(m2.group(2)),
                               0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, False)
    return out


def fig_headline(ci_log, kernel_log, featlm_log, featlm_strict_log):
    """Dumbbell per prior mean: graph off -> plain graph -> harmonic graph (strict)."""
    ci = read_ci_rows(ci_log)
    k_lm = read_kernel_block(kernel_log, "mu_LM")
    k_zero = read_kernel_block(kernel_log, "mu = 0")
    k_feat = read_kernel_block(featlm_log, "mu_LM")
    # feat+LM graph-off is only in the strict feat log's pooled table
    txt = open(featlm_strict_log).read()
    feat_off = float(re.search(r"feat\+LM-ma\s+off\s+([\d.]+)", txt).group(1))

    rows = [
        ("no prior mean", ci["zero mean, graph off"],
         k_zero["additive"][0], k_zero["harmonic_vl"][0]),
        ("network (music model)", ci["LM mean, graph off"],
         k_lm["additive"][0], k_lm["harmonic_vl"][0]),
        ("features + network", feat_off,
         k_feat["additive"][0], k_feat["harmonic_vl"][0]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ys = np.arange(len(rows))[::-1]
    for y, (label, off, plain, harm) in zip(ys, rows):
        ax.plot([harm, off], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([off], [y], s=70, color=BLUE_LT, zorder=2)
        ax.scatter([plain], [y], s=70, color=BLUE_DK, zorder=3)
        ax.scatter([harm], [y], s=85, color=AQUA, marker="D", zorder=4)
        ax.annotate(f"{harm:.3f}", (harm, y), textcoords="offset points",
                    xytext=(-11, -3), color=INK2, fontsize=9, ha="right",
                    va="center")
    ax.set_xlim(0.335, 0.59)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlabel("held-out error (pooled RMSE, strict protocol — lower is better)")
    ax.scatter([], [], s=70, color=BLUE_LT, label="no graph")
    ax.scatter([], [], s=70, color=BLUE_DK, label="plain score graph")
    ax.scatter([], [], s=85, color=AQUA, marker="D", label="+ chord/voice-leading edges")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Two orthogonal upgrades: a better guess, and a better graph",
                 color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/digest_headline.png")
    plt.close(fig)


def fig_kernels(kernel_log):
    """Dot + CI whisker per kernel: paired per-piece delta vs the plain additive graph."""
    k = read_kernel_block(kernel_log, "mu_LM")
    rows = [  # (key, display), simplest -> experimental; independent off scale (see note)
        ("chain", "time-chain only"),
        ("matern1", "graph Matérn α=1"),
        ("matern2", "graph Matérn α=2"),
        ("matern3", "graph Matérn α=3"),
        ("diffusion", "diffusion / heat"),
        ("norm_additive", "normalized Laplacian"),
        ("tonal", "tonal pitch metric (replace)"),
        ("harmonic", "chord edges (add)"),
        ("harmonic_vl", "chord + voice-leading (add)"),
    ]
    panels = [("Δ error (RMSE) vs plain graph", 2, 3, 4, 5),
              ("Δ confidence quality (NLL) vs plain graph", 6, 7, 8, 9)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, (label, i_d, i_lo, i_hi, i_sig) in zip(axes, panels):
        ax.axvline(0, color=BASE, lw=1, zorder=1)
        for y, (key, _) in zip(ys, rows):
            v = k[key]
            d, lo, hi, sig = v[i_d], v[i_lo], v[i_hi], v[i_sig]
            color = MUTED if not sig else (BLUE_DK if d < 0 else DIV_RED)
            ax.plot([lo, hi], [y, y], color=color, lw=2, alpha=0.75, zorder=2)
            ax.scatter([d], [y], s=48, color=color, zorder=3)
        ax.set_xlabel(label)
        _style(ax)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[1] for r in rows], color=INK)
    h = [axes[0].scatter([], [], s=48, color=BLUE_DK, label="significantly better"),
         axes[0].scatter([], [], s=48, color=MUTED, label="tie (ns)"),
         axes[0].scatter([], [], s=48, color=DIV_RED, label="significantly worse")]
    fig.legend(handles=h, frameon=False, loc="upper right", fontsize=8,
               ncol=3, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Only added music-theory edges beat the plain graph",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.015,
             "paired per-piece 95% CIs vs the additive baseline; independent (no coupling) omitted: "
             "+0.052 / +0.131, off scale",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(f"{OUT}/digest_kernels.png")
    plt.close(fig)


def fig_channels(grid_log):
    chans = ["tau", "log r", "v"]
    names = {"tau": "timing (τ)", "log r": "articulation (log r)", "v": "loudness (v)"}
    rel = []
    for ch in chans:
        t = read_channel(grid_log, ch)
        rel.append(100.0 * (t[("LM", "off")] - t[("feat", "off")]) / t[("feat", "off")])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ys = np.arange(len(chans))[::-1]
    colors = [BLUE if v < 0 else DIV_RED for v in rel]
    ax.barh(ys, rel, height=0.45, color=colors)
    ax.axvline(0, color=BASE, lw=1)
    for y, v in zip(ys, rel):
        ax.annotate(f"{v:+.0f}%", (v, y), textcoords="offset points",
                    xytext=(6 if v >= 0 else -6, -3),
                    ha="left" if v >= 0 else "right", color=INK2, fontsize=10)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[c] for c in chans], color=INK)
    ax.set_xlabel("network error vs feature error (mean-only, % — negative = network better)")
    lim = max(abs(v) for v in rel) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_title("Where the network earns its place: loudness",
                 color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/digest_channels.png")
    plt.close(fig)


def fig_collapse(unguarded_log, guarded_log):
    rows = [("guard off", read_cells(unguarded_log)),
            ("guard on", read_cells(guarded_log))]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    rng = np.random.default_rng(0)
    for yi, (label, cells) in enumerate(rows):
        vals = np.array([c[2] for c in cells])
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        normal = vals <= 1.0
        ax.scatter(vals[normal], np.full(normal.sum(), yi) + jitter[normal],
                   s=22, color=BLUE, alpha=0.55, edgecolors="none")
        for v, j in zip(vals[~normal], jitter[~normal]):
            ax.scatter([v], [yi + j], s=60, color=RED, zorder=3)
            ax.annotate(f"one collapsed fit: {v:.1f}", (v, yi + j),
                        textcoords="offset points", xytext=(-8, 10),
                        ha="right", color=RED, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("per-evaluation-cell timing error (τ RMSE), 120 cells each")
    ax.set_title("The 1-in-120 fit collapse, and the safeguard that removes it",
                 color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/digest_collapse.png")
    plt.close(fig)


def read_confirmation(path):
    """{system: (rmse, nll, cov)} from logs/confirmation_verdict.log."""
    out = {}
    for line in open(path):
        m = re.match(r"(.+?)\s{2,}([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s*$", line.rstrip())
        if m and "system" not in line:
            out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)),
                                       float(m.group(4)))
    return out


def fig_confirmation(conf_log):
    """The thesis headline: one-shot confirmation RMSE, 20 untouched pieces."""
    conf = read_confirmation(conf_log)
    # log keys are verbatim row names from the FROZEN confirmation log
    # (evidence/logs/confirmation_verdict.log) — never "modernize" them
    rows = [  # (log key, display, color)
        ("GP b_featlm", "proposed model (full)", BLUE_DK),
        ("GP b_feat", "proposed model, no music model", MUTED),
        ("GP b_featlm_nograph", "proposed model, no graph", MUTED),
        ("old headline (feat+LM+harm)", "two-stage pipeline (strongest)", BLUE_LT),
        ("old LM+graph", "two-stage pipeline (plain)", MUTED),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ys = np.arange(len(rows))[::-1]
    for y, (key, label, color) in zip(ys, rows):
        r = conf[key][0]
        ax.plot([0.35, r], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([r], [y], s=85 if color == BLUE_DK else 60, color=color, zorder=3)
        ax.annotate(f"{r:.3f}", (r, y), textcoords="offset points",
                    xytext=(8, -3), color=INK2, fontsize=9)
    ax.set_xlim(0.35, 0.425)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows], color=INK)
    ax.set_xlabel("pooled RMSE on 20 untouched pieces (lower is better)")
    ax.set_title("Preregistered confirmation: the proposed model wins",
                 color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/proposed_confirmation.png")
    plt.close(fig)


def _load_gp_cells(pattern):
    import glob, pickle
    ys, prs, sds = [], [], []
    for f in sorted(glob.glob(pattern)):
        for v in pickle.load(open(f, "rb"))["cells"].values():
            ys.append(v[0]); prs.append(v[1]); sds.append(v[2])
    return np.concatenate(ys), np.concatenate(prs), np.concatenate(sds)


def fig_gp_calibration(dev_pattern, conf_pattern):
    """Reliability diagram + PIT histogram for the proposed model (appendix)."""
    import sys
    sys.path.insert(0, "src")
    from score_bundle.metrics import coverage as _coverage, pit_values as _pit

    sets = [("development", _load_gp_cells(dev_pattern), BLUE_DK),
            ("confirmation", _load_gp_cells(conf_pattern), AQUA)]
    levels = np.linspace(0.1, 0.95, 18)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color=BASE, ls="--", lw=1, label="ideal")
    for name, (y, pr, sd), color in sets:
        emp = [_coverage(y, pr, sd, level=L) for L in levels]
        ax.plot(levels, emp, "o-", ms=3.5, lw=1.6, color=color, label=name)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability — proposed model", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/proposed_reliability.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, (name, (y, pr, sd), color) in zip(axes, sets):
        u = _pit(y, pr, sd)
        ax.hist(u, bins=20, range=(0, 1), color=color, edgecolor=SURFACE)
        ax.axhline(len(u) / 20, color=INK2, ls="--", lw=1)
        ax.set_title(f"PIT — {name}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("PIT value")
        _style(ax, xgrid=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/proposed_pit.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    # all inputs read from the committed evidence archive (byte-identical copies
    # of the original logs/ and results/ artifacts), so every figure regenerates
    # from a fresh clone; see evidence/README.md
    grid_log = "evidence/logs/feature_baseline_l2_10.log"
    if "--collapse-only" not in sys.argv:
        fig_headline("evidence/logs/headline_cis.log",
                     "evidence/logs/kernels_report.log",
                     "evidence/logs/kernels_featlm_report.log",
                     "evidence/logs/eval_featlm_strict_lin.log")
        fig_channels(grid_log)
        fig_kernels("evidence/logs/kernels_report.log")
        fig_confirmation("evidence/logs/confirmation_verdict.log")
        fig_gp_calibration("evidence/results/graphgp_v2/b_featlm.shard*.pkl",
                           "evidence/results/graphgp_conf/b_featlm.shard*.pkl")
        print("headline + channels + kernels + confirmation + calibration written")
    fig_collapse("evidence/logs/diag_tau_s2x3.log",
                 "evidence/logs/diag_tau_s2x3_guarded.log")
    print("collapse written")
