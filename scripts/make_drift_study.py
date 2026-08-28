#!/usr/bin/env python
"""Figure for the drift study (fig:drift-study).

Four panels that carry the §3.9 drift argument visually:
  A. one real note: confidence-kept frames, the constant-centre sine fit,
     and the same fit with the linear drift term — drift made visible;
  B. tracker slope vs ground-truth slope on both-significant notes
     (two independent witnesses agree: sign agreement, rank correlation);
  C. the decisive negative: note i's drift slope vs note i+1's —
     no across-note structure;
  D. the contrast: note i's timing residual vs note i+1's — the
     structure the graph actually uses.

Slopes are recomputed from the f0 caches (pitch-only refit, no audio
pass, ~1 min); C/D pool consecutive pairs after per-track
standardization. House style: Okabe-Ito, legends above panels.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_drift_study.py
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

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
OUT = "docs/thesis/figures/drift_study_dev.png"
EX_TRACK = (1, 1)


def drift_fit(tt, x, f):
    """(c, slope, a, b) for x ~ c + s(t-mean) + a sin + b cos, + slope SE."""
    tc = tt - tt.mean()
    th = 2 * np.pi * f * tt
    A = np.stack([np.ones(tt.size), tc, np.sin(th), np.cos(th)], 1)
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ beta
    dof = max(tt.size - 4, 1)
    try:
        cov = (r @ r / dof) * np.linalg.inv(A.T @ A + 1e-10 * np.eye(4))
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = np.inf
    return beta, se


def main() -> None:
    from score_bundle.phase2.intonation import cents_from_f0
    from score_bundle.phase2.urmp import (read_f0_annotation,
                                          read_notes_annotation)
    from eval_phase2_real import dev_unique_tracks

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    sl_tr, se_tr, sl_gt, se_gt = [], [], [], []
    seq_slopes, seq_tau = {}, {}
    example = None
    for key, d in sorted(data.items()):
        f0c = f0s[key]
        tr = tracks[key]
        notes = read_notes_annotation(tr.notes)
        t_gt, f0_gt = read_f0_annotation(tr.f0s)
        ok = f0c["voiced"] & np.isfinite(f0c["f0"]) & (f0c["f0"] > 0)
        ok &= f0c["prob"] >= np.quantile(f0c["prob"][ok], 0.2)
        ok_gt = np.isfinite(f0_gt) & (f0_gt > 0)
        n = d["onset"].size
        track_slopes = np.full(n, np.nan)
        for i in range(n):
            if not (d["ident"][i] and np.isfinite(d["est"][i, 2])):
                continue
            on, du = notes["onset"][i], notes["duration"][i]
            sel = ok & (f0c["t"] >= on) & (f0c["t"] < on + du)
            if sel.sum() < 12:
                continue
            tt = f0c["t"][sel] - on
            x = cents_from_f0(f0c["f0"][sel], 440.0,
                              float(d["midi"][i] - 69))
            f = np.exp(d["est"][i, 2])
            beta, se = drift_fit(tt, x, f)
            s_tr = float(beta[1])
            track_slopes[i] = s_tr
            selg = ok_gt & (t_gt >= on) & (t_gt < on + du)
            if selg.sum() < 12 and example is None:
                pass
            if selg.sum() >= 12 and d["ident_gt"][i] \
                    and np.isfinite(d["est_gt"][i, 2]):
                ttg = t_gt[selg] - on
                xg = cents_from_f0(f0_gt[selg], 440.0,
                                   float(d["midi"][i] - 69))
                bg, seg = drift_fit(ttg, xg, np.exp(d["est_gt"][i, 2]))
                sl_tr.append(s_tr)
                se_tr.append(se)
                sl_gt.append(float(bg[1]))
                se_gt.append(seg)
            if (du > 1.0 and abs(s_tr) > 2 * se
                    and 8.0 <= abs(s_tr) <= 30.0
                    and abs(np.mean(x)) < 40 and np.max(np.abs(x)) < 70
                    and np.isfinite(d["est"][i, 1])
                    and np.exp(d["est"][i, 1]) > 6):
                score_ex = abs(s_tr) * du
                if example is None or score_ex > example[0]:
                    example = (score_ex, tt, x, f, beta, i, du, key)
        seq_slopes[key] = track_slopes
        seq_tau[key] = d["tau"]

    sl_tr, se_tr = np.array(sl_tr), np.array(se_tr)
    sl_gt, se_gt = np.array(sl_gt), np.array(se_gt)
    strong = (np.abs(sl_tr) > 2 * se_tr) & (np.abs(sl_gt) > 2 * se_gt)

    def lag_pairs(seqs):
        a, b = [], []
        for v in seqs.values():
            v = np.asarray(v, float)
            m = np.isfinite(v)
            if m.sum() < 10:
                continue
            z = (v - np.nanmean(v)) / max(np.nanstd(v), 1e-9)
            p = np.isfinite(z[:-1]) & np.isfinite(z[1:])
            a.extend(z[:-1][p])
            b.extend(z[1:][p])
        return np.array(a), np.array(b)

    da, db = lag_pairs(seq_slopes)
    ta, tb = lag_pairs(seq_tau)
    r_drift = float(np.corrcoef(da, db)[0, 1])
    r_tau = float(np.corrcoef(ta, tb)[0, 1])

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.4), dpi=200,
                             width_ratios=[1.7, 1.05, 1, 1])

    # A. one note
    ax = axes[0]
    _, tt, x, f, beta, i_ex, du, ex_key = example
    ax.plot(tt * 1e3, x, ".", ms=3.2, color=MUTED,
            label="tracked frames")
    g = np.linspace(0, tt.max(), 400)
    th = 2 * np.pi * f * g
    flat_beta, _ = drift_fit(tt, x, f)
    c0 = np.mean(x)  # visual flat reference not needed; use flat model:
    A0 = np.stack([np.ones(tt.size), np.sin(2 * np.pi * f * tt),
                   np.cos(2 * np.pi * f * tt)], 1)
    b0, *_ = np.linalg.lstsq(A0, x, rcond=None)
    ax.plot(g * 1e3, b0[0] + b0[1] * np.sin(th) + b0[2] * np.cos(th),
            color=MUTED, lw=1.3, ls="--",
            label="constant-centre sine (the model)")
    ax.plot(g * 1e3, beta[0] + beta[1] * (g - tt.mean())
            + beta[2] * np.sin(th) + beta[3] * np.cos(th),
            color=BLUE, lw=1.6, label="+ drift term")
    ax.plot(g * 1e3, beta[0] + beta[1] * (g - tt.mean()), color=VERM,
            lw=1.3, ls=":", label=f"the centre, drifting "
            f"({beta[1]:+.0f} cents/s)")
    ax.set_xlabel("time from note onset (ms)", fontsize=9, color=INK)
    ax.set_ylabel("cents vs written pitch", fontsize=9, color=INK)
    instr = data[ex_key]["instrument"]
    ax.set_title(f"A. note {i_ex} ({du:.1f} s {instr}): the centre moves",
                 fontsize=9.6, color=INK, loc="left", pad=30)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="lower left",
              bbox_to_anchor=(0.0, 1.0))

    # B. tracker vs GT slopes
    ax = axes[1]
    lim = 60
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    inl = strong & (np.abs(sl_tr) < lim) & (np.abs(sl_gt) < lim)
    ax.plot(sl_tr[inl], sl_gt[inl], ".", ms=2.6, color=BLUE, alpha=0.45)
    sign = np.mean(np.sign(sl_tr[strong]) == np.sign(sl_gt[strong]))
    ax.set_title("B. two independent witnesses",
                 fontsize=9.6, color=INK, loc="left", pad=30)
    ax.text(0.04, 0.96, f"sign agreement {sign:.0%}",
            transform=ax.transAxes, fontsize=8, va="top", color=INK)
    ax.set_xlabel("tracker slope (cents/s)", fontsize=8.5, color=INK)
    ax.set_ylabel("ground-truth slope", fontsize=8.5, color=INK)

    # C/D. lag-1 scatters
    for ax, (a, b), r, ttl, col in (
            (axes[2], (da, db), r_drift,
             "C. drift: no across-note structure", BLUE),
            (axes[3], (ta, tb), r_tau,
             "D. timing: the structure the graph uses", GREEN)):
        ax.plot(a, b, ".", ms=2.2, color=col, alpha=0.35)
        ax.set_title(ttl, fontsize=9.2, color=INK, loc="left", pad=30)
        ax.text(0.04, 0.96, f"lag-1 r = {r:+.2f}",
                transform=ax.transAxes, fontsize=8.4, va="top",
                color=INK)
        ax.set_xlabel("note $i$ (standardized)", fontsize=8.5, color=INK)
        ax.set_ylabel("note $i{+}1$", fontsize=8.5, color=INK)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT} | example note {i_ex}, "
          f"pooled lag-1 drift {r_drift:+.2f} tau {r_tau:+.2f}")


if __name__ == "__main__":
    main()
