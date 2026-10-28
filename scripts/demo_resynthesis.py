#!/usr/bin/env python
"""Resynthesis demo: the model's read of one passage, played back.

DEV track, exploratory, no claim, nothing registered touched.  Same
passage and the same deployment fits as make_endgoal_viz.py (score-nominal
tempo-line starts, no annotation in the fit, chains fitted independently).
From each fitted chain the posterior-mean pitch curves and the inferred
boundaries define the harmonic design Phi(z); the closed-form amplitude
posterior mean at the MAP (eq:amp-post) gives a; the render is Phi a.
Three clips over the passage:

  recording.wav    the URMP track over the passage span
  resynthesis.wav  Phi(z_hat) a_hat: the model's expression (pitch curves,
                   boundaries) under its per-chunk harmonic envelopes
  flat.wav         the SAME amplitudes under score-nominal pitch (0 cents,
                   no curve) and tempo-line boundaries: expression removed
  residual.wav     recording minus resynthesis (timbre and noise, by
                   construction)

Honesty: harmonics only (no noise band, no transient at the joints);
notes outside the chain rule are not fitted and are silent in the renders;
the one number reported, the fraction of the recording's energy the
harmonic render explains, is a diagnostic of the signal model, not an
evaluation of the expression estimate (Chapter 8 evaluates that against
the reference curves directly).

Writes results/resynthesis_demo/*.wav (gitignored),
docs/thesis/figures/resynthesis_demo.png, results/resynthesis_demo.md.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/demo_resynthesis.py
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
from make_endgoal_viz import MAX_CHAINS, MAX_CHAIN_GAP_S, TRACK_KEY

INK, MUTED = "#1A1A1A", "#6B7280"
BLUE, VERM, GREEN, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"
OUT_DIR = "results/resynthesis_demo"
OUT_FIG = "docs/thesis/figures/resynthesis_demo.png"
OUT_MD = "results/resynthesis_demo.md"
N_HARM, N_CHUNK, AMP_VAR, SOFTNESS = 8, 2, 10.0, 0.010


def chain_render(x, t, midis, fit, b_lo, b_hi, cents_of, m_a=None):
    """Render Phi a over the chain segment.  cents_of(k, ts) gives note k's
    pitch deviation on the sample times ts; b_lo/b_hi the joint windows.
    If m_a is None the amplitude posterior mean at the given design is
    solved (ridge = noise_var / AMP_VAR, as in the collapsed model);
    otherwise the given amplitudes are reused (same column layout)."""
    import torch

    from score_bundle.phase3.curve import design_support_torch

    t_np = np.asarray(t, float)
    m = t_np.size
    k_n = len(midis)
    with torch.no_grad():
        x_t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64)
        t_t = torch.as_tensor(t_np, dtype=torch.float64)
        b_lo_t = torch.as_tensor(np.asarray(b_lo, float), dtype=torch.float64)
        b_hi_t = torch.as_tensor(np.asarray(b_hi, float), dtype=torch.float64)
        cols, sizes = [], []
        for k in range(k_n):
            lo, hi = fit.knots[k][0], fit.knots[k][-1]
            ia = int(np.searchsorted(t_np, max(t_np[0], lo - 0.2)))
            ib = int(np.searchsorted(t_np, min(t_np[-1], hi + 0.2)))
            ts = t_t[ia:ib]
            cents = cents_of(k, ts)
            Phi_k = design_support_torch(cents, ts, float(midis[k]), ia,
                                         m, N_HARM, N_CHUNK)
            w = torch.ones(m, dtype=torch.float64)
            if k > 0:
                w = w * torch.sigmoid((t_t - b_lo_t[k - 1]) / SOFTNESS)
            if k < k_n - 1:
                w = w * torch.sigmoid((b_hi_t[k] - t_t) / SOFTNESS)
            cols.append(Phi_k * w.unsqueeze(1))
            sizes.append(Phi_k.shape[1])
        Phi = torch.cat(cols, dim=1)
        if m_a is None:
            p = Phi.shape[1]
            G = Phi.T @ Phi + (fit.noise_var / AMP_VAR) * torch.eye(
                p, dtype=torch.float64)
            m_a = torch.cholesky_solve((Phi.T @ x_t).unsqueeze(1),
                                       torch.linalg.cholesky(G)).squeeze(1)
        xhat = (Phi @ m_a).numpy()
    return xhat, m_a


def fade(sig, ms=10.0):
    n = int(SR * ms / 1000.0)
    if sig.size <= 2 * n:
        return sig
    ramp = np.linspace(0.0, 1.0, n)
    out = sig.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    from eval_phase2_real import dev_unique_tracks
    from score_bundle.phase2.urmp import read_notes_annotation
    from score_bundle.phase3.curve import (chain_reference, fit_chain,
                                           interp_knots)

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })
    os.makedirs(OUT_DIR, exist_ok=True)

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
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

    t_wall = time.time()
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
        print(f"chain {idx}: fitted [{time.time() - t0_:.0f}s]  "
              f"c = {np.round(fit.c, 1)}", flush=True)

        def cents_expr(k, ts, fit=fit):
            kt = torch.as_tensor(fit.knots[k], dtype=torch.float64)
            u_k = torch.as_tensor(fit.us[k], dtype=torch.float64)
            return fit.c[k] + interp_knots(u_k, kt, ts)

        def cents_flat(k, ts):
            return torch.zeros_like(ts)

        xhat, m_a = chain_render(x, t, midis, fit, fit.b_lo, fit.b_hi,
                                 cents_expr)
        xflat, _ = chain_render(x, t, midis, fit, b_nom, b_nom, cents_flat,
                                m_a=m_a)
        # per-note explained energy inside the inferred note windows
        cuts = np.concatenate([[0.0], 0.5 * (fit.b_lo + fit.b_hi),
                               [float(t[-1])]])
        rows = []
        for k in range(len(midis)):
            sel = (t >= cuts[k]) & (t < cuts[k + 1])
            e_x = float(np.sum(x[sel] ** 2))
            e_r = float(np.sum((x[sel] - xhat[sel]) ** 2))
            rows.append((idx[k], midis[k], float(fit.c[k]),
                         1.0 - e_r / e_x if e_x > 0 else np.nan))
        fits.append(dict(idx=idx, x=x, t=t, t0=t0, midis=midis, fit=fit,
                         xhat=xhat, xflat=xflat, rows=rows))

    t_start = fits[0]["t0"]
    t_end = fits[-1]["t0"] + fits[-1]["t"][-1]
    sa, sb = int(t_start * SR), int(t_end * SR)
    rec = audio[sa:sb].copy()
    resyn = np.zeros_like(rec)
    flat = np.zeros_like(rec)
    for F in fits:
        off = int((F["t0"] - t_start) * SR)
        n = min(F["xhat"].size, rec.size - off)
        resyn[off:off + n] += F["xhat"][:n]
        flat[off:off + n] += F["xflat"][:n]
    resid = rec - resyn
    e_tot = 1.0 - float(np.sum(resid ** 2)) / float(np.sum(rec ** 2))
    fitted = np.zeros(rec.size, bool)
    for F in fits:
        off = int((F["t0"] - t_start) * SR)
        fitted[off:off + F["xhat"].size] = True
    e_fit = 1.0 - (float(np.sum(resid[fitted] ** 2))
                   / float(np.sum(rec[fitted] ** 2)))
    peak = float(np.max(np.abs(rec)))
    for name, sig in [("recording", rec), ("resynthesis", resyn),
                      ("flat", flat), ("residual", resid)]:
        out = np.clip(fade(sig) / peak * 0.9, -1.0, 1.0)
        sf.write(os.path.join(OUT_DIR, f"{name}.wav"), out, SR)
    wall = time.time() - t_wall

    # figure: recording, resynthesis, a 40 ms zoom, explained energy
    fig, axes = plt.subplots(
        4, 1, figsize=(11.0, 8.0),
        gridspec_kw={"height_ratios": [0.8, 0.8, 1.0, 0.8], "hspace": 0.62})
    axa, axb, axc, axd = axes
    dec = max(1, rec.size // 9000)
    tt = np.arange(0, rec.size, dec) / SR
    axa.plot(tt, rec[::dec], color=VERM, lw=0.45, alpha=0.85)
    axb.plot(tt, resyn[::dec], color=BLUE, lw=0.45, alpha=0.85)
    for F in fits:
        off = F["t0"] - t_start
        for j in range(len(F["midis"]) - 1):
            for ax in (axa, axb):
                ax.axvline(off + F["fit"].b_lo[j], color=INK, lw=0.8,
                           alpha=0.5)
    for ax in (axa, axb):
        ax.set_yticks([])
        ax.set_xlim(0, rec.size / SR)
    axa.set_title("A    the recording (vertical lines: inferred note "
                  "boundaries; unfitted notes are gaps in B)", loc="left")
    axb.set_title("B    the resynthesis Phi(z) a: posterior-mean curves, "
                  "inferred boundaries, amplitude posterior mean",
                  loc="left")
    axb.set_xlabel("time in the passage (s)")
    # zoom: 40 ms in the middle of the loudest fitted note
    F0 = max(fits, key=lambda F: np.max(np.abs(F["xhat"])))
    k0 = int(np.argmax([r[3] if np.isfinite(r[3]) else -1
                        for r in F0["rows"]]))
    cuts = np.concatenate([[0.0], 0.5 * (F0["fit"].b_lo + F0["fit"].b_hi),
                           [float(F0["t"][-1])]])
    tm = 0.5 * (cuts[k0] + cuts[k0 + 1])
    i0 = int((F0["t0"] - t_start + tm) * SR)
    w = int(0.020 * SR)
    zt = (np.arange(i0 - w, i0 + w) - (i0 - w)) * 1000.0 / SR
    axc.plot(zt, rec[i0 - w:i0 + w], color=VERM, lw=1.0, label="recording")
    axc.plot(zt, resyn[i0 - w:i0 + w], color=BLUE, lw=1.0,
             label="resynthesis")
    axc.plot(zt, resid[i0 - w:i0 + w], color=MUTED, lw=0.8,
             label="residual (timbre and noise)")
    ymax = 1.55 * float(np.max(np.abs(rec[i0 - w:i0 + w])))
    axc.set_ylim(-1.1 * ymax / 1.55, ymax)
    axc.legend(fontsize=7.5, loc="upper right", ncol=3)
    axc.set_xlabel("time (ms), 40 ms inside the best-explained note")
    axc.set_yticks([])
    axc.set_title(f"C    forty milliseconds of note {F0['idx'][k0]} "
                  f"(MIDI {F0['midis'][k0]:.0f})", loc="left")
    labels, vals = [], []
    for F in fits:
        for (i, mi, c, e) in F["rows"]:
            labels.append(f"{i}\n{mi:.0f}")
            vals.append(e)
    axd.bar(np.arange(len(vals)), vals, color=BLUE, width=0.7)
    axd.set_xticks(np.arange(len(vals)))
    axd.set_xticklabels(labels, fontsize=7.5)
    axd.set_ylim(0, 1)
    axd.set_ylabel("explained energy")
    axd.set_xlabel("note index / MIDI pitch")
    axd.set_title(f"D    fraction of the recording's energy the harmonic "
                  f"render explains, per note (fitted span: {e_fit:.2f})",
                  loc="left")
    fig.savefig(OUT_FIG, bbox_inches="tight")

    n_notes = sum(len(F["midis"]) for F in fits)
    lines = [
        "# Resynthesis demo (dev track, exploratory, no claim)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
        f"`scripts/demo_resynthesis.py` in {wall:.0f} s "
        f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}).",
        "",
        f"Passage: URMP dev track {TRACK_KEY}, {len(fits)} chains, "
        f"{n_notes} notes, {t_end - t_start:.1f} s, the same passage and "
        "the same deployment fits as `docs/thesis/figures/endgoal_viz.png` "
        "(score-nominal tempo-line starts, no annotation in the fit, "
        "chains fitted independently, harmonics only).",
        "",
        "Clips in `results/resynthesis_demo/` (gitignored, 16 kHz, peak-"
        "normalized to the recording):",
        "",
        "- `recording.wav`: the track over the passage span.",
        "- `resynthesis.wav`: Phi(z_hat) a_hat, the posterior-mean pitch "
        "curves and inferred boundaries under the amplitude posterior mean "
        "at the MAP (eq:amp-post). Unfitted notes are silent.",
        "- `flat.wav`: the same amplitudes under score-nominal pitch "
        "(0 cents, no curve) and tempo-line boundaries: the expression "
        "removed, the timbre strand kept.",
        "- `residual.wav`: recording minus resynthesis: timbre and noise, "
        "by construction.",
        "",
        "Diagnostic (NOT an evaluation of the expression estimate; Chapter 8 "
        "evaluates that against the reference curves directly): fraction of "
        "the recording's energy the harmonic render explains, "
        "1 - ||x - Phi a||^2 / ||x||^2.",
        "",
        f"- Whole passage: {e_tot:.3f}  (fitted span only: {e_fit:.3f}).",
        "",
        "| chain | note | MIDI | inferred c (cents) | explained energy |",
        "|---|---|---|---|---|",
    ]
    for ci, F in enumerate(fits):
        for (i, mi, c, e) in F["rows"]:
            lines.append(f"| {ci} | {i} | {mi:.0f} | {c:+.1f} | {e:.3f} |")
    lines += [
        "",
        "Honesty: the signal model is harmonics with piecewise-constant "
        "amplitudes over two chunks per note and sigmoid windows at the "
        "joints; no noise band, no transient model. The remainder is what "
        "the decomposition (score + expression + timbre + noise) assigns to "
        "timbre and noise, and what Future Work (`sec:future-phase3`) "
        "proposes to bring under the prior.",
        "",
        f"Figure: `{OUT_FIG}`.",
    ]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"explained energy: passage {e_tot:.3f}, fitted span {e_fit:.3f}")
    print("wrote", OUT_DIR, OUT_FIG, OUT_MD, f"[{wall:.0f}s]")


if __name__ == "__main__":
    main()
