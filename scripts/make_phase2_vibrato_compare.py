#!/usr/bin/env python
"""Truth vs estimator vs GP, at the note level (fig:vibrato-compare).

The three layers Ray asked to see in one place, for the f0-derived
channels: the ACTUAL pitch curve (URMP ground truth), the per-note
ESTIMATE of eq:cents-curve/eq:vibrato fitted on the tracked curve, and
the GP's PREDICTION. Two probative cases from the Fig-3.4 track and seed:

  A. a HELD-OUT note: the GP never saw this note's estimate; its
     prediction is validated against the truth-derived parameters;
  B. a VISIBLE note whose vibrato the estimator could NOT identify
     (missing cells): the GP fills them; truth says how well.

Curve panels show truth (line), tracked frames (dots), the eq-3.33 fit
where it exists, and the GP-implied curve from its predicted parameters
(phase is not a target; the gate uses the GP's predicted delay).
Parameter panels put truth / estimate / GP side by side with their
uncertainties. Reads only caches + one as-given GP fit (the published
recipe); no evaluation artifacts are touched.

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/make_phase2_vibrato_compare.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_phase2_real import _fit_systems, dev_unique_tracks  # noqa: E402

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
TRACK, SEED = (1, 1), 0
_Z90 = 1.6448536269514722
OUT = "docs/thesis/figures/phase2_vibrato_compare_dev.png"


def main() -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.phase2.intonation import (cents_from_f0,
                                                fit_vibrato_note_gated)
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)
    from score_bundle.score import Score

    d = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))[TRACK]
    py = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))[TRACK]
    tr = next(t for p, t in dev_unique_tracks() if (p.index, t.number) == TRACK)
    notes = read_notes_annotation(tr.notes)
    t_gt, f0_gt = read_f0_annotation(tr.f0s)

    # --- the published as-given fit, exactly as in the evaluation ---------
    est = np.concatenate([d["est"], d["ell"][:, None],
                          d["tau"][:, None], d["dvib"][:, None]], axis=1)
    var = np.concatenate([d["var"], d["var_ell"][:, None],
                          d["var_tau"][:, None], d["var_dvib"][:, None]],
                         axis=1)
    ident = d["ident"]
    n = est.shape[0]
    usable = np.isfinite(est[:, 0]) & (np.abs(est[:, 0]) <= 150.0)
    score = Score.from_arrays(d["midi"], d["onset"], d["duration"],
                              np.zeros(n, dtype=int))
    eig = np.linalg.eigh(laplacian(build_adjacency(score)))
    X = rich_score_features(score, rff_dim=0)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    feats = [np.concatenate([X, np.ones((n, 1))], axis=1)]
    k = 6
    scale = np.ones((n, k))
    for c in range(k):
        v = var[:, c]
        med = np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 1.0
        scale[:, c] = np.where(np.isfinite(v),
                               np.clip(v / max(med, 1e-12), 1e-2, 1e3), 1.0)
    rng = np.random.default_rng(1000 + 7 * TRACK[0] + TRACK[1] + SEED)
    held = (rng.random(n) < 0.30) & usable
    mask = np.zeros((n, k), dtype=bool)
    mask[:, 0] = usable & ~held
    mask[:, 1] = mask[:, 2] = usable & ~held & ident
    mask[:, 3] = usable & ~held & np.isfinite(est[:, 3])
    mask[:, 4] = ~held & np.isfinite(est[:, 4])
    mask[:, 5] = ~held & np.isfinite(est[:, 5])
    Yobs = np.where(mask, np.nan_to_num(est), 0.0)
    m, sd, sd_pred = _fit_systems(eig, feats, Yobs, mask, scale, var,
                                  None)["gp_asgiven"]

    # --- pick the two cases -----------------------------------------------
    gam_gt = np.where(np.isfinite(d["est_gt"][:, 1]),
                      np.exp(d["est_gt"][:, 1]), np.nan)
    cand_a = [i for i in range(n) if held[i] and ident[i] and d["ident_gt"][i]
              and np.isfinite(gam_gt[i]) and gam_gt[i] > 12
              and notes["duration"][i] > 0.8]
    cand_b = [i for i in range(n) if (not held[i]) and usable[i]
              and (not ident[i]) and d["ident_gt"][i]
              and np.isfinite(gam_gt[i]) and notes["duration"][i] > 0.45]
    A, B = cand_a[0], cand_b[0]
    print("case A (held-out):", A, "| case B (estimator-missing):", B)

    ok_py = py["voiced"] & np.isfinite(py["f0"]) & (py["f0"] > 0)
    floor = np.quantile(py["prob"][ok_py], 0.2)
    ok_py = ok_py & (py["prob"] >= floor)
    ok_gt = np.isfinite(f0_gt) & (f0_gt > 0)

    def curves(i):
        on, du = notes["onset"][i], notes["duration"][i]
        sp = ok_py & (py["t"] >= on) & (py["t"] < on + du)
        sg = ok_gt & (t_gt >= on) & (t_gt < on + du)
        semi = float(d["midi"][i] - 69)
        return ((py["t"][sp] - on, cents_from_f0(py["f0"][sp], 440.0, semi)),
                (t_gt[sg] - on, cents_from_f0(f0_gt[sg], 440.0, semi)))

    fig = plt.figure(figsize=(11.4, 6.6), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], hspace=0.55, wspace=0.22)

    for row, (i, tag) in enumerate([
            (A, "case A — held-out note: the GP never saw its estimate"),
            (B, "case B — visible note, vibrato cells estimator-missing: the GP fills them")]):
        (tp, cp), (tg, cg) = curves(i)
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(tg * 1e3, cg, color=INK, lw=1.0, alpha=0.8,
                label="actual pitch curve (ground truth)")
        ax.plot(tp * 1e3, cp, ".", ms=3.5, color=MUTED, alpha=0.8,
                label="tracked frames")
        if ident[i]:
            fitp = fit_vibrato_note_gated(tp, cp)
            g = np.linspace(0, tp.max(), 400)
            gate = g >= fitp["delta"]
            ax.plot(g * 1e3, fitp["c"] + np.where(
                gate, fitp["gamma"] * np.sin(
                    2 * np.pi * fitp["f"] * (g - fitp["delta"])), 0.0),
                color=BLUE, lw=1.5, label="estimate: eq. 3.32/3.33 fit")
        # GP-implied curve from predicted parameters
        cN, gN = m[i, 0], np.exp(m[i, 1])
        fN, dN = np.exp(m[i, 2]), m[i, 5]
        g = np.linspace(0, notes["duration"][i], 400)
        gate = g >= dN
        ax.plot(g * 1e3, cN + np.where(
            gate, gN * np.sin(2 * np.pi * fN * (g - dN)), 0.0),
            color=VERM, lw=1.5, ls="--",
            label="GP prediction (its parameters, drawn)")
        ax.fill_between(g * 1e3, cN - _Z90 * sd_pred[i, 0],
                        cN + _Z90 * sd_pred[i, 0], color=VERM, alpha=0.10,
                        linewidth=0)
        ax.set_title(f"note {i} ({notes['duration'][i]:.2f} s) — {tag}",
                     fontsize=9.5, color=INK, loc="left", pad=28)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left",
                  bbox_to_anchor=(0.0, 1.0))
        ax.set_ylabel("cents", fontsize=9, color=INK)
        if row == 1:
            ax.set_xlabel("time from note onset (ms)", fontsize=9, color=INK)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

        # parameter comparison: truth / estimate / GP per channel (card style)
        axp = fig.add_subplot(gs[row, 1])
        axp.axis("off")
        gtv = [d["est_gt"][i, 0], gam_gt[i],
               np.exp(d["est_gt"][i, 2]) if np.isfinite(d["est_gt"][i, 2]) else np.nan,
               d["dvib_gt"][i] * 1e3 if np.isfinite(d["dvib_gt"][i]) else np.nan]
        ev = [est[i, 0],
              np.exp(est[i, 1]) if np.isfinite(est[i, 1]) else np.nan,
              np.exp(est[i, 2]) if np.isfinite(est[i, 2]) else np.nan,
              est[i, 5] * 1e3 if np.isfinite(est[i, 5]) else np.nan]
        es = [np.sqrt(var[i, 0]),
              ev[1] * np.sqrt(var[i, 1]) if np.isfinite(est[i, 1]) else np.nan,
              ev[2] * np.sqrt(var[i, 2]) if np.isfinite(est[i, 2]) else np.nan,
              np.sqrt(var[i, 5]) * 1e3 if np.isfinite(var[i, 5]) else np.nan]
        gp = [m[i, 0], np.exp(m[i, 1]), np.exp(m[i, 2]), m[i, 5] * 1e3]
        gs_ = [sd_pred[i, 0], gp[1] * sd_pred[i, 1], gp[2] * sd_pred[i, 2],
               sd_pred[i, 5] * 1e3]
        names = [r"$c$ (cents)", r"$\gamma$ (cents)",
                 r"$f^{\mathrm{vib}}$ (Hz)", r"$\delta^{\mathrm{vib}}$ (ms)"]
        axp.text(0.00, 1.00, "", transform=axp.transAxes)
        cols = [(0.30, "truth", INK), (0.58, "estimate", BLUE),
                (0.90, "GP", VERM)]
        for xc, hd, colc in cols:
            axp.text(xc, 0.97, hd, fontsize=8.5, color=colc, ha="center",
                     weight="bold", transform=axp.transAxes)
        y = 0.80
        for nm, tv, e_, s_e, g_, s_g in zip(names, gtv, ev, es, gp, gs_):
            axp.text(0.0, y, nm, fontsize=9, color=INK,
                     transform=axp.transAxes)
            axp.text(0.30, y, f"{tv:.1f}", fontsize=9, color=INK,
                     ha="center", family="monospace",
                     transform=axp.transAxes)
            if np.isfinite(e_):
                axp.text(0.58, y, f"{e_:.1f} ±{s_e:.1f}", fontsize=9,
                         color=BLUE, ha="center", family="monospace",
                         transform=axp.transAxes)
            else:
                axp.text(0.58, y, "MISSING", fontsize=8.5, color=VERM,
                         style="italic", ha="center",
                         transform=axp.transAxes)
            axp.text(0.90, y, f"{g_:.1f} ±{s_g:.1f}", fontsize=9,
                     color=VERM, ha="center", family="monospace",
                     transform=axp.transAxes)
            y -= 0.20
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
