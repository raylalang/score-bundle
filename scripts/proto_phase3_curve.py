#!/usr/bin/env python
"""Phase-3 prototype: a within-note curve posterior from the waveform.

The two verified halves glued together for the first time: the
within-note GP prior (the two-component kernel of
phase2/sm_estimator.sm_kernel, here on a knot grid) and the
amplitude-collapsed waveform likelihood (phase3/waveform_model via the
chunked design of eval_phase3_waveform_dev). This is the module
docstring's documented stub -- inference over the nonlinear z beyond
scalar grids -- and the feasibility probe for the design question's
curve-level-channels depth: the two GP levels meeting at the waveform.

Model, per note (x = audio segment, u = cents curve at J knots):

    cents(t) = c + interp(t; knots, u),  u ~ N(0, K_knots)
    x | c, u ~ N(0, Phi(f0(cents)) Sigma_a Phi^T + sigma^2 I)

MAP over (c, u) by coarse c grid (flat curve) then Nelder-Mead over the
J+1 parameters; per-knot uncertainty by diagonal Laplace (prototype
honesty: diagonal only). NO tracker, NO estimator anywhere.

Output: docs/thesis/figures/phase3_curve_proto.png (three instrument-
diverse dev notes: inferred curve +/- band vs the corpus ground-truth
frames vs the tracked pYIN frames, neither of which the fit ever saw)
and printed frame RMSE of the inferred curve against the GT frames,
with the pYIN frames' own RMSE as the anchor.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/proto_phase3_curve.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_phase3_waveform_dev import (SR, chunked_design,  # noqa: E402
                                      fit_noise, loglik, selected)
from score_bundle.optimize import nelder_mead  # noqa: E402
from score_bundle.phase2.sm_estimator import sm_kernel  # noqa: E402
from score_bundle.phase3.waveform_model import collapsed_loglik_lowrank  # noqa: E402

KNOTS_PER_S = 16           # knot rate (Nyquist for ~8 Hz vibrato)
J_MIN, J_MAX = 12, 32
AMP_VAR = 10.0
# Prototype prior hyperparameters (fixed, documented): vibrato component
# ~8 cents at 5.5 Hz with a moderate band, drift component ~8 cents
# decorrelating over ~0.3 s. These are prior SCALES, not fits.
W1, MU1, V1 = 30.0, 5.5, 0.4
W2, V2 = 60.0, 0.3
INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM = "#0072B2", "#D55E00"
OUT = "docs/thesis/figures/phase3_curve_proto.png"


def cents_to_f0(midi: float, cents: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((midi - 69.0 + cents / 100.0) / 12.0)


def curve_loglik(x, t, midi, cents, nv):
    Phi = chunked_design(cents_to_f0(midi, cents), t)
    Sigma_a = np.eye(Phi.shape[1]) * AMP_VAR
    return collapsed_loglik_lowrank(x, Phi, Sigma_a, noise_var=nv)


def fit_note(x, t, midi, nm_iter=500):
    """MAP (c, u) under the knot-GP prior + waveform likelihood."""
    J = int(np.clip(round((t[-1] - t[0]) * KNOTS_PER_S), J_MIN, J_MAX))
    knots = np.linspace(t[0], t[-1], J)
    K = sm_kernel(knots[:, None] - knots[None, :], W1, MU1, V1, W2, V2)
    K[np.diag_indices_from(K)] += 1e-6 * (W1 + W2)
    Kinv = np.linalg.inv(K)

    # coarse c, flat curve (the dev study's estimator-free initializer)
    grid = np.arange(-50.0, 50.0, 1.0)
    nv0 = fit_noise(x, t, midi, 0.0)
    lls = np.array([loglik(x, t, midi, c, nv0) for c in grid])
    c0 = float(grid[int(np.argmax(lls))])
    nv = fit_noise(x, t, midi, c0)

    def neg(p):
        c, u = p[0], p[1:]
        cents = c + np.interp(t, knots, u)
        prior = -0.5 * float(u @ (Kinv @ u))
        return -(curve_loglik(x, t, midi, cents, nv) + prior)

    p = np.concatenate([[c0], np.zeros(J)])
    p = nelder_mead(neg, p, step=3.0, max_iter=nm_iter)
    p = nelder_mead(neg, p, step=1.0, max_iter=nm_iter)
    p = nelder_mead(neg, p, step=0.4, max_iter=nm_iter // 2)

    # diagonal Laplace on the knots (prototype)
    sd = np.full(J, np.nan)
    
    f0v = neg(p)
    h = 0.5
    for j in range(J):
        e = np.zeros(J + 1)
        e[1 + j] = h
        curv = (neg(p + e) - 2 * f0v + neg(p - e)) / h ** 2
        if curv > 0:
            sd[j] = 1.0 / np.sqrt(curv)
    return p[0], p[1:], knots, sd


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    from scipy.signal import resample_poly

    from make_phase2_intro import note_curves
    from score_bundle.phase2.urmp import read_notes_annotation
    from eval_phase2_real import dev_unique_tracks

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    picks = []
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        for i in range(d["onset"].size):
            du = float(notes["duration"][i])
            if (d["ident"][i] and 0.9 <= du <= 1.9
                    and d["n_frames"][i] >= 60):
                picks.append((key, i, d, tr, notes))
                break
        if len(picks) == 3:
            break

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    for ax, (key, i, d, tr, notes) in zip(axes, picks):
        t0 = time.time()
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
        on = float(notes["onset"][i])
        du = min(float(notes["duration"][i]), 2.0)
        a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
        x = audio[max(a, 0):min(b, audio.size)]
        t = np.arange(x.size) / SR
        midi = float(d["midi"][i])

        c_map, u_map, knots, sd = fit_note(x, t, midi)
        tg = np.linspace(t[0], t[-1], 300)
        curve = c_map + np.interp(tg, knots, u_map)
        band = np.interp(tg, knots, np.where(np.isfinite(sd), sd, 5.0))

        # evaluation frames (never seen by the fit); align to segment time
        tt, xx, ttg, xg = note_curves(key, i, data, f0s, tracks)
        off = 0.02
        sel_g = (ttg >= off) & (ttg <= off + t[-1])
        sel_t = (tt >= off) & (tt <= off + t[-1])
        gt_t, gt_x = ttg[sel_g] - off, xg[sel_g]
        tr_t, tr_x = tt[sel_t] - off, xx[sel_t]
        rmse_curve = float(np.sqrt(np.mean(
            (np.interp(gt_t, tg, curve) - gt_x) ** 2)))
        rmse_pyin = float(np.sqrt(np.mean(
            (np.interp(gt_t, tr_t, tr_x) - gt_x) ** 2))) if tr_t.size > 1 else np.nan

        ax.plot(gt_t, gt_x, ".", ms=3.5, color=VERM, label="ground truth frames")
        ax.plot(tr_t, tr_x, ".", ms=2.5, color="#009E73", label="pYIN frames")
        ax.plot(tg, curve, color=BLUE, lw=1.6,
                label=f"from the waveform ({rmse_curve:.1f} c vs GT)")
        ax.fill_between(tg, curve - 1.645 * band, curve + 1.645 * band,
                        color=BLUE, alpha=0.15, linewidth=0)
        ax.set_title(f"{d['instrument']}  (pYIN frames: "
                     f"{rmse_pyin:.1f} c vs GT)", loc="left")
        ax.set_xlabel("time in the note (s)")
        ax.margins(y=0.25)
        ax.legend(fontsize=7, loc="best")
        print(f"{key}#{i} {d['instrument']}: curve {rmse_curve:.2f} c, "
              f"pYIN {rmse_pyin:.2f} c vs GT  [{time.time()-t0:.0f}s]",
              flush=True)
    axes[0].set_ylabel("pitch deviation (cents)")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
