#!/usr/bin/env python
"""Deck figure: the waveform chapter's end goal, prototyped on one passage.

A real URMP violin passage decomposed from score and audio alone — no
tracker, no annotation access in the fit (deployment protocol: nominal
tempo-line starts). Three panels on one timeline:

  A  the waveform with inferred note boundaries,
  B  the inferred per-note pitch curves with joint-Laplace 90% bands
     (annotation frames overlaid, never seen by the fit),
  C  the per-note timbre strand: harmonic reconstruction envelopes read
     back from the closed-form amplitude posterior at the MAP.

Honesty: chains are fitted independently (3-5 notes each, the current
machinery); notes outside the chain rule are not fitted and show as
gaps; the envelope strand is read out post hoc, not yet under the prior.

Writes docs/thesis/figures/endgoal_viz.png.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/make_endgoal_viz.py
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

from eval_phase3_chain_dev import SR, chain_segment, select_chains
from eval_phase3_curve_dev import load_hypers
from make_chain_viz import chain_curve_laplace, curve_band

TRACK_KEY = (1, 1)
MAX_CHAINS = 3
MAX_CHAIN_GAP_S = 4.0
INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"
OUT = "docs/thesis/figures/endgoal_viz.png"


def note_envelopes(x, t, midis, fit, win_s=0.012, n_harm=8, n_chunk=2,
                   amp_var=10.0, softness=0.010):
    """Per-note reconstruction envelopes from the amplitude posterior
    mean at the MAP (ridge LS with the collapsed model's ridge)."""
    import torch

    from score_bundle.phase3.curve import (design_support_torch,
                                           interp_knots)

    t_np = np.asarray(t, float)
    m = t_np.size
    k_n = len(midis)
    with torch.no_grad():
        x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64)
        t_t = torch.as_tensor(t_np, dtype=torch.float64)
        b_lo = torch.as_tensor(fit.b_lo, dtype=torch.float64)
        b_hi = torch.as_tensor(fit.b_hi, dtype=torch.float64)
        cols, sizes = [], []
        for k in range(k_n):
            lo, hi = fit.knots[k][0], fit.knots[k][-1]
            ia = int(np.searchsorted(t_np, max(t_np[0], lo - 0.2)))
            ib = int(np.searchsorted(t_np, min(t_np[-1], hi + 0.2)))
            ts = t_t[ia:ib]
            kt = torch.as_tensor(fit.knots[k], dtype=torch.float64)
            u_k = torch.as_tensor(fit.us[k], dtype=torch.float64)
            cents = fit.c[k] + interp_knots(u_k, kt, ts)
            Phi_k = design_support_torch(cents, ts, float(midis[k]), ia,
                                         m, n_harm, n_chunk)
            w = torch.ones(m, dtype=torch.float64)
            if k > 0:
                w = w * torch.sigmoid((t_t - b_lo[k - 1]) / softness)
            if k < k_n - 1:
                w = w * torch.sigmoid((b_hi[k] - t_t) / softness)
            cols.append(Phi_k * w.unsqueeze(1))
            sizes.append(Phi_k.shape[1])
        Phi = torch.cat(cols, dim=1)
        p = Phi.shape[1]
        G = Phi.T @ Phi + (fit.noise_var / amp_var) * torch.eye(
            p, dtype=torch.float64)
        m_a = torch.cholesky_solve((Phi.T @ x_t).unsqueeze(1),
                                   torch.linalg.cholesky(G)).squeeze(1)
        envs, pos = [], 0
        k_w = max(int(win_s * SR), 8)
        kern = np.ones(k_w) / k_w
        for k in range(k_n):
            xk = (cols[k] @ m_a[pos:pos + sizes[k]]).numpy()
            pos += sizes[k]
            env = np.sqrt(np.convolve(xk ** 2, kern, mode="same"))
            envs.append(env)
    return envs


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    from scipy.signal import resample_poly

    from eval_phase2_real import dev_unique_tracks
    from make_phase2_intro import note_curves
    from score_bundle.phase2.urmp import read_notes_annotation
    from score_bundle.phase3.curve import chain_reference, fit_chain

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
    d, tr = data[TRACK_KEY], tracks[TRACK_KEY]
    notes = read_notes_annotation(tr.notes)
    audio48, sr48 = sf.read(tr.audio)
    audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
    tau = d["tau"]
    hypers = load_hypers()

    chains = select_chains(notes)
    picked = [chains[0]]
    for idx in chains[1:]:
        gap = (float(notes["onset"][idx[0]])
               - float(notes["onset"][picked[-1][-1]]
                       + notes["duration"][picked[-1][-1]]))
        if gap <= MAX_CHAIN_GAP_S and len(picked) < MAX_CHAINS:
            picked.append(idx)
    print("passage chains:", picked, flush=True)

    fits = []
    for idx in picked:
        x, t, t0, b_annot = chain_segment(audio, notes, idx)
        midis = [float(d["midi"][i]) for i in idx]
        b_nom = b_annot.copy()
        for j, i in enumerate(idx[1:]):
            if np.isfinite(tau[i]):
                b_nom[j] = b_annot[j] - float(tau[i])
        t0_ = time.time()
        c0, nv = chain_reference(x, t, midis, b_nom, guard_frac=0.2)
        fit = fit_chain(x, t, midis, b_nom, hypers=hypers, c_init=c0,
                        noise_var=nv, free_width=True)
        covs = chain_curve_laplace(x, t, midis, fit, hypers)
        envs = note_envelopes(x, t, midis, fit)
        fits.append(dict(idx=idx, x=x, t=t, t0=t0, b_annot=b_annot,
                         midis=midis, fit=fit, covs=covs, envs=envs))
        print(f"chain {idx}: fitted [{time.time() - t0_:.0f}s]", flush=True)

    t_start = fits[0]["t0"]
    t_end = fits[-1]["t0"] + fits[-1]["t"][-1]
    fig, (axa, axb, axc) = plt.subplots(
        3, 1, figsize=(11.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.3, 0.8], "hspace": 0.14})

    sa, sb = int(t_start * SR), int(t_end * SR)
    dec = max(1, (sb - sa) // 9000)
    tt = np.arange(sa, sb, dec) / SR - t_start
    axa.plot(tt, audio[sa:sb:dec], color=VERM, lw=0.45, alpha=0.85)
    for F in fits:
        off = F["t0"] - t_start
        for j in range(len(F["midis"]) - 1):
            axa.axvline(off + F["fit"].b_lo[j], color=BLUE, lw=1.5)
            axa.axvline(off + F["b_annot"][j], color=INK, lw=0.9,
                        alpha=0.65)
    axa.plot([], [], color=BLUE, lw=1.5, label="inferred boundary")
    axa.plot([], [], color=INK, lw=0.9, alpha=0.65,
             label="annotated onset (reference only)")
    axa.legend(fontsize=7.5, loc="upper right", ncol=2)
    axa.set_yticks([])
    axa.set_ylabel("audio")
    axa.set_title("A    the recording, segmented into notes", loc="left")

    first = True
    for F in fits:
        off = F["t0"] - t_start
        cuts = np.concatenate([[0.0], F["b_annot"], [F["t"][-1]]])
        for k in range(len(F["midis"])):
            tq = np.linspace(cuts[k], cuts[k + 1], 120)
            mean_k = F["fit"].curve(k, tq)
            band = curve_band(F["fit"], k, tq, F["covs"][k])
            axb.fill_between(off + tq, mean_k - band, mean_k + band,
                             color=BLUE, alpha=0.18, linewidth=0)
            axb.plot(off + tq, mean_k, color=BLUE, lw=1.3,
                     label="inferred pitch curve, 90 percent band"
                     if first else None)
            try:
                _, _, ttg, xg = note_curves(TRACK_KEY, F["idx"][k], data,
                                            f0s, tracks)
                gt_t = float(notes["onset"][F["idx"][k]]) - t_start + ttg
                sel = (gt_t >= off + cuts[k]) & (gt_t <= off + cuts[k + 1])
                axb.plot(gt_t[sel], xg[sel], ".", ms=2.2, color=VERM,
                         label="annotation frames (never seen)"
                         if first else None)
            except Exception:
                pass
            first = False
    axb.set_ylim(-55, 65)
    axb.set_ylabel("pitch deviation (cents)")
    axb.legend(fontsize=7.5, loc="upper right", ncol=2)
    axb.set_title("B    per-note pitch expression, with uncertainty",
                  loc="left")

    for F in fits:
        off = F["t0"] - t_start
        td = F["t"][::8]
        for k, env in enumerate(F["envs"]):
            axc.fill_between(off + td, 0.0, env[::8],
                             color=(BLUE if k % 2 == 0 else AMBER),
                             alpha=0.55, linewidth=0)
    axc.set_ylabel("envelope")
    axc.set_yticks([])
    axc.set_xlabel("time in the passage (s)")
    axc.set_title("C    the timbre strand: per-note harmonic envelopes "
                  "from the amplitude posterior (alternating colours = "
                  "adjacent notes)", loc="left")

    fig.savefig(OUT, bbox_inches="tight")
    n_notes = sum(len(F["midis"]) for F in fits)
    print(f"passage: {len(fits)} chains, {n_notes} notes, "
          f"{t_end - t_start:.1f}s")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
