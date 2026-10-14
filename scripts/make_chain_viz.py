#!/usr/bin/env python
"""Deck figure: one chain, score + audio in, boundaries + curves out.

Runs the deployment arm (score-nominal tempo-line init, no annotation
access) of curve.fit_chain on the first eligible dev chain and draws the
first genuine Phase-3 output: A the waveform with annotated onsets,
tempo-line starting points, and inferred boundaries; B the inferred
per-note cents curves against the corpus annotation frames (never seen).

Writes docs/thesis/figures/chain_boundary_deck.png (house style).

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_chain_viz.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_phase3_chain_dev import iter_chains
from eval_phase3_curve_dev import load_hypers

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
OUT = "docs/thesis/figures/chain_boundary_deck.png"


def chain_curve_laplace(x, t, midis, fit, hypers, softness=0.010,
                        n_harm=8, n_chunk=2, amp_var=10.0):
    """Joint Laplace over all curve parameters (c's and u's) at the chain
    MAP, boundaries fixed at their fitted values (batch fits skip this —
    it is computed here once, for the figure). Returns the per-note
    (c_k, u_k) covariance blocks of the JOINT inverse Hessian, so
    cross-note coupling through the shared likelihood is kept."""
    import torch

    from score_bundle.phase3.curve import (CurvePrior,
                                           collapsed_loglik_torch,
                                           design_support_torch,
                                           interp_knots, knot_gram)

    dev = torch.device("cpu")
    x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64)
    t_np = np.asarray(t, float)
    t_t = torch.as_tensor(t_np, dtype=torch.float64)
    m = t_np.size
    k_n = len(midis)
    prior = CurvePrior()
    b_lo = torch.as_tensor(fit.b_lo, dtype=torch.float64)
    b_hi = torch.as_tensor(fit.b_hi, dtype=torch.float64)
    cuts = np.concatenate([[t_np[0]],
                           0.5 * (fit.b_lo + fit.b_hi), [t_np[-1]]])
    knots_t, chols, slices, sizes = [], [], [], []
    for k in range(k_n):
        kt = torch.as_tensor(fit.knots[k], dtype=torch.float64)
        _, chol = knot_gram(kt, prior, **(hypers or {}))
        knots_t.append(kt)
        chols.append(chol)
        lo, hi = fit.knots[k][0], fit.knots[k][-1]
        ia = int(np.searchsorted(t_np, max(t_np[0], lo - 0.2)))
        ib = int(np.searchsorted(t_np, min(t_np[-1], hi + 0.2)))
        slices.append((ia, ib))
        sizes.append(1 + fit.knots[k].size)

    def nlj(theta):
        pos, cols = 0, []
        prior_term = 0.0
        for k in range(k_n):
            c_k = theta[pos]
            u_k = theta[pos + 1:pos + sizes[k]]
            pos += sizes[k]
            ia, ib = slices[k]
            ts = t_t[ia:ib]
            cents = c_k + interp_knots(u_k, knots_t[k], ts)
            Phi_k = design_support_torch(cents, ts, float(midis[k]), ia,
                                         m, n_harm, n_chunk)
            w = torch.ones(m, dtype=torch.float64)
            if k > 0:
                w = w * torch.sigmoid((t_t - b_lo[k - 1]) / softness)
            if k < k_n - 1:
                w = w * torch.sigmoid((b_hi[k] - t_t) / softness)
            cols.append(Phi_k * w.unsqueeze(1))
            prior_term = prior_term + 0.5 * (u_k @ torch.cholesky_solve(
                u_k.unsqueeze(1), chols[k]).squeeze(1))
        Phi = torch.cat(cols, dim=1)
        return -collapsed_loglik_torch(x_t, Phi, fit.noise_var,
                                       amp_var) + prior_term

    theta0 = torch.as_tensor(
        np.concatenate([np.concatenate([[fit.c[k]], fit.us[k]])
                        for k in range(k_n)]), dtype=torch.float64)
    H = torch.autograd.functional.hessian(nlj, theta0)
    H = 0.5 * (H + H.T)
    try:
        cov = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        ev, evec = torch.linalg.eigh(H)
        ev = ev.clamp(min=1e-8 * float(ev.max().clamp(min=1.0)))
        cov = (evec / ev) @ evec.T
    cov = cov.numpy()
    blocks, pos = [], 0
    for k in range(k_n):
        blocks.append(cov[pos:pos + sizes[k], pos:pos + sizes[k]])
        pos += sizes[k]
    return blocks


def curve_band(fit, k, tq, cov_k):
    """90 percent band of note k's curve under its joint-Laplace block."""
    j = fit.knots[k].size
    idx = np.clip(np.searchsorted(fit.knots[k], tq) - 1, 0, j - 2)
    w = ((tq - fit.knots[k][idx])
         / (fit.knots[k][idx + 1] - fit.knots[k][idx]))
    A = np.zeros((tq.size, j + 1))
    A[:, 0] = 1.0
    rows = np.arange(tq.size)
    A[rows, 1 + idx] += 1.0 - w
    A[rows, 1 + idx + 1] += w
    var = np.einsum("ij,jk,ik->i", A, cov_k, A)
    return 1.645 * np.sqrt(np.maximum(var, 0.0))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from score_bundle.phase3.curve import chain_reference, fit_chain

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })

    hypers = load_hypers()
    ch = next(c for _, c in iter_chains())
    x, t, midis = ch["x"], ch["t"], ch["midis"]
    c_nom, nv_nom = chain_reference(x, t, midis, ch["b_nom"],
                                    guard_frac=0.2)
    fit = fit_chain(x, t, midis, ch["b_nom"], hypers=hypers, c_init=c_nom,
                    noise_var=nv_nom, free_width=True)
    covs = chain_curve_laplace(x, t, midis, fit, hypers)

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(10.6, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.35], "hspace": 0.12})

    # A: waveform + the three boundary sets
    dec = max(1, x.size // 6000)
    axa.plot(t[::dec], x[::dec], color=VERM, lw=0.5, alpha=0.8)
    for j in range(len(midis) - 1):
        axa.axvline(ch["b_annot"][j], color=INK, lw=1.2)
        axa.axvline(ch["b_nom"][j], color=MUTED, lw=1.0, ls="--")
        axa.axvline(fit.b_lo[j], color=BLUE, lw=1.6)
    axa.plot([], [], color=INK, lw=1.2, label="annotated onset")
    axa.plot([], [], color=MUTED, lw=1.0, ls="--",
             label="tempo-line start (score only)")
    axa.plot([], [], color=BLUE, lw=1.6, label="inferred boundary")
    axa.set_ylabel("audio")
    axa.set_yticks([])
    axa.legend(fontsize=7.5, loc="upper right", ncol=3)
    axa.set_title(
        f"A    violin, {len(midis)} consecutive notes: boundaries from the "
        "score and the waveform alone", loc="left")

    # B: inferred curves vs the annotation frames
    cuts = ch["cuts"]
    for k in range(len(midis)):
        tq = np.linspace(cuts[k], cuts[k + 1], 150)
        band = curve_band(fit, k, tq, covs[k])
        mean_k = fit.curve(k, tq)
        axb.fill_between(tq, mean_k - band, mean_k + band, color=BLUE,
                         alpha=0.18, linewidth=0,
                         label="90 percent band" if k == 0 else None)
        axb.plot(tq, mean_k, color=BLUE, lw=1.5,
                 label="inferred pitch curve" if k == 0 else None)
        if ch["gts"][k] is not None:
            gt_t, gt_x = ch["gts"][k]
            sel = (gt_t >= cuts[k]) & (gt_t <= cuts[k + 1])
            axb.plot(gt_t[sel], gt_x[sel], ".", ms=2.8, color=VERM,
                     label="annotation frames (never seen)"
                     if k == 0 else None)
        axb.axvline(cuts[k + 1] if k < len(midis) - 1 else np.nan,
                    color="#E5E7EB", lw=0.8)
    for j in range(len(midis) - 1):
        axb.axvline(fit.b_lo[j], color=BLUE, lw=1.0, alpha=0.5)
    axb.set_ylim(-60, 60)
    axb.set_xlabel("time in the segment (s)")
    axb.set_ylabel("pitch deviation (cents)")
    axb.legend(fontsize=7.5, loc="upper right", ncol=2)
    axb.set_title("B    per-note pitch curves from the same joint fit",
                  loc="left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    err = np.abs(fit.b_lo - ch["b_annot"]) * 1000
    ini = np.abs(ch["b_nom"] - ch["b_annot"]) * 1000
    print(f"chain {ch['key']} notes {ch['idx']}: init err "
          + ",".join(f"{e:.0f}" for e in ini) + " ms -> inferred "
          + ",".join(f"{e:.0f}" for e in err) + " ms")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
