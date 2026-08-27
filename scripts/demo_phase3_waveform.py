#!/usr/bin/env python
"""Phase-3 feasibility: waveform-likelihood inference of intonation on a
real URMP note (DEV data, exploratory; no tracker in the loop).

The Phase-3 claim in miniature: the audio itself is the observation.  For
one note, parameterize the f0 curve by the Phase-2 channel variables
(eq:cents-curve/eq:vibrato: centre c, extent gamma, rate f, delay delta),
build the harmonic design matrix Phi(z) (phase coherence lives in the
deterministic basis), marginalize the harmonic amplitudes exactly
(waveform_model.collapsed_loglik_lowrank), and grid the collapsed
likelihood over c (and over f at the c optimum).  Grid inference is exact
up to discretization in 1-D --- no Laplace machinery is claimed.

Amplitudes are piecewise-constant over four equal time chunks (the
eq:loudness segmentation) x K harmonics in quadrature, which absorbs the
attack/decay envelope at chunk resolution; the noise floor is fit from the
residual of an initial amplitude regression at the estimator's curve.

Validation targets per note: the Phase-2 estimator's c (tracked frames)
and the quasi-truth c (same estimator on URMP's ground-truth curve).

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/demo_phase3_waveform.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

TRACK = (1, 1)
SR = 16000
N_HARM = 8
N_CHUNK = 4
AMP_VAR = 10.0
C_HALF, C_STEP = 30.0, 0.5          # cents grid: est_c +/- 30, 0.5-cent steps


def f0_curve(t, midi, c, gamma=0.0, f=0.0, delta=0.0):
    """eq:cents-curve/eq:vibrato as an f0(t) in Hz on the note's clock."""
    cents = c + (np.where(t >= delta,
                          gamma * np.sin(2 * np.pi * f * (t - delta)), 0.0)
                 if gamma > 0 and f > 0 else 0.0)
    return 440.0 * 2.0 ** ((midi - 69.0 + cents / 100.0) / 12.0)


def chunked_design(f0, t, n_harm, n_chunk):
    """Phi(z) with piecewise-constant amplitudes: per-chunk harmonic blocks."""
    from score_bundle.phase3.synth import harmonic_design_matrix
    base = harmonic_design_matrix(f0, t, n_harm)          # (m, 2K)
    m = t.size
    edges = np.linspace(0, m, n_chunk + 1).astype(int)
    cols = []
    for q in range(n_chunk):
        blk = np.zeros_like(base)
        blk[edges[q]:edges[q + 1]] = base[edges[q]:edges[q + 1]]
        cols.append(blk)
    return np.concatenate(cols, axis=1)                   # (m, 2K*n_chunk)


def note_segment(audio, sr, onset, duration, pad=0.02):
    a = int((onset + pad) * sr)
    b = int((onset + duration - pad) * sr)
    return audio[max(a, 0):min(b, audio.size)]


def loglik_grid(x, t, midi, c_grid, noise_var, **curve_kw):
    from score_bundle.phase3.waveform_model import collapsed_loglik_lowrank
    p = 2 * N_HARM * N_CHUNK
    Sigma_a = np.eye(p) * AMP_VAR
    lls = np.empty(c_grid.size)
    for i, c in enumerate(c_grid):
        Phi = chunked_design(f0_curve(t, midi, c, **curve_kw), t,
                             N_HARM, N_CHUNK)
        lls[i] = collapsed_loglik_lowrank(x, Phi, Sigma_a,
                                          noise_var=noise_var)
    return lls


def posterior_stats(grid, lls):
    w = np.exp(lls - lls.max())
    w /= w.sum()
    mean = float(w @ grid)
    sd = float(np.sqrt(w @ (grid - mean) ** 2))
    return grid[int(np.argmax(lls))], mean, sd


def fit_noise_var(x, t, midi, c0, **curve_kw):
    Phi = chunked_design(f0_curve(t, midi, c0, **curve_kw), t,
                         N_HARM, N_CHUNK)
    beta, *_ = np.linalg.lstsq(Phi, x, rcond=None)
    r = x - Phi @ beta
    return float(r @ r / max(x.size - Phi.shape[1], 1))


def main() -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    from score_bundle.phase2.urmp import read_notes_annotation
    from eval_phase2_real import dev_unique_tracks

    d = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))[TRACK]
    tr = next(t for p, t in dev_unique_tracks()
              if (p.index, t.number) == TRACK)
    notes = read_notes_annotation(tr.notes)
    audio48, sr48 = sf.read(tr.audio)
    audio = resample_poly(np.asarray(audio48, dtype=float),
                          SR, int(sr48))

    n = len(d["ident"])
    dur = notes["duration"]
    # vibrato exemplar (the Fig-3.2 note) + a steady long note
    vib = [i for i in range(n) if d["ident"][i]
           and np.isfinite(d["dvib"][i]) and d["dvib"][i] > 0.08
           and dur[i] > 0.8 and np.isfinite(d["est"][i, 1])
           and np.exp(d["est"][i, 1]) > 12.0][2]
    steady = max((i for i in range(n) if not d["ident"][i]
                  and np.isfinite(d["est"][i, 0]) and dur[i] > 0.3
                  and d["n_frames"][i] >= 30),
                 key=lambda i: dur[i])

    print(f"track {TRACK} ({d['instrument']}): vibrato note {vib} "
          f"({dur[vib]:.2f}s), steady note {steady} ({dur[steady]:.2f}s)\n")

    for name, i, use_vib in (("steady", steady, False),
                             ("vibrato", vib, True)):
        x = note_segment(audio, SR, notes["onset"][i], dur[i])
        t = np.arange(x.size) / SR
        midi = float(d["midi"][i])
        est_c, est_sd = d["est"][i, 0], float(np.sqrt(d["var"][i, 0]))
        gt_c = d["est_gt"][i, 0]
        kw = {}
        if use_vib:
            kw = dict(gamma=float(np.exp(d["est"][i, 1])),
                      f=float(np.exp(d["est"][i, 2])),
                      delta=float(d["dvib"][i]))
        nv = fit_noise_var(x, t, midi, est_c, **kw)
        snr = 10 * np.log10(float(x @ x) / x.size / nv)
        grid = est_c + np.arange(-C_HALF, C_HALF + 1e-9, C_STEP)
        lls = loglik_grid(x, t, midi, grid, nv, **kw)
        coarse_map = grid[int(np.argmax(lls))]
        # refine: the collapsed likelihood can be sharper than the coarse
        # step; a 0.02-cent pass around the peak resolves the true width
        fine = coarse_map + np.arange(-0.6, 0.6 + 1e-9, 0.02)
        lls_fine = loglik_grid(x, t, midi, fine, nv, **kw)
        grid = np.concatenate([grid, fine])
        lls = np.concatenate([lls, lls_fine])
        order = np.argsort(grid)
        grid, lls = grid[order], lls[order]
        map_c, mean_c, sd_c = posterior_stats(grid, lls)
        print(f"[{name}] note {i}  m={x.size} samples  SNR {snr:.1f} dB")
        print(f"  waveform posterior c: MAP {map_c:+.2f}, "
              f"mean {mean_c:+.2f} +/- {sd_c:.2f} cents")
        print(f"  Phase-2 estimator c:  {est_c:+.2f} +/- {est_sd:.2f} "
              f"(tracked frames)")
        print(f"  quasi-truth c (GT):   {gt_c:+.2f}")
        print(f"  |waveform - GT| = {abs(mean_c - gt_c):.2f} cents; "
              f"|estimator - GT| = {abs(est_c - gt_c):.2f} cents\n")

        if use_vib:                       # second variable: vibrato rate
            f_est = kw["f"]
            fgrid = np.linspace(max(f_est - 1.5, 2.5), f_est + 1.5, 61)

            def f_lls(fg):
                out = np.empty(fg.size)
                for j, fv in enumerate(fg):
                    out[j] = loglik_grid(x, t, midi, np.array([mean_c]),
                                         nv, **dict(kw, f=float(fv)))[0]
                return out

            lls_f = f_lls(fgrid)
            f_map0 = fgrid[int(np.argmax(lls_f))]
            ffine = f_map0 + np.arange(-0.05, 0.05 + 1e-9, 0.002)
            fgrid = np.concatenate([fgrid, ffine])
            lls_f = np.concatenate([lls_f, f_lls(ffine)])
            o = np.argsort(fgrid)
            map_f, mean_f, sd_f = posterior_stats(fgrid[o], lls_f[o])
            gt_f = (np.exp(d["est_gt"][i, 2])
                    if np.isfinite(d["est_gt"][i, 2]) else np.nan)
            print(f"  waveform posterior f_vib: MAP {map_f:.2f}, mean "
                  f"{mean_f:.2f} +/- {sd_f:.2f} Hz | estimator "
                  f"{f_est:.2f} | quasi-truth {gt_f:.2f}\n")


if __name__ == "__main__":
    main()
