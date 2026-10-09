#!/usr/bin/env python
"""Differentiable Phase-3 curve inference (torch): the DDSP-direction step.

Ports the chunked harmonic design and the amplitude-collapsed Gaussian
log likelihood to torch (exact, Woodbury form), equality-checks them
against the numpy path, then runs gradient-based MAP over (c, knots)
under the same two-component GP prior as proto_phase3_curve, on the
same three notes, reporting GT-frame RMSE and wall time vs Nelder-Mead.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/proto_phase3_torch.py [cuda:N]
"""
from __future__ import annotations

import pickle
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from eval_phase3_waveform_dev import (SR, N_HARM, N_CHUNK, AMP_VAR,
                                      chunked_design, f0_curve, fit_noise,
                                      loglik, selected)
from proto_phase3_curve import (KNOTS_PER_S, J_MIN, J_MAX, W1, MU1, V1,
                                W2, V2)
from score_bundle.phase2.sm_estimator import sm_kernel

DEV = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
DTYPE = torch.float64


def torch_design(cents: torch.Tensor, t: torch.Tensor,
                 midi: float) -> torch.Tensor:
    """Chunked harmonic design Phi(z), differentiable in cents."""
    f0 = 440.0 * torch.pow(torch.tensor(2.0, dtype=DTYPE, device=DEV),
                           (midi - 69.0 + cents / 100.0) / 12.0)
    dt = torch.diff(t, prepend=t[:1])
    integral = torch.cumsum(f0 * dt, dim=0)
    cols = []
    for k in range(1, N_HARM + 1):
        phi = 2.0 * np.pi * k * integral
        cols.append(torch.cos(phi))
        cols.append(torch.sin(phi))
    base = torch.stack(cols, dim=1)
    m = t.shape[0]
    edges = np.linspace(0, m, N_CHUNK + 1).astype(int)
    blocks = []
    for q in range(N_CHUNK):
        msk = torch.zeros(m, 1, dtype=DTYPE, device=DEV)
        msk[edges[q]:edges[q + 1]] = 1.0
        blocks.append(base * msk)
    return torch.cat(blocks, dim=1)


def torch_loglik(x: torch.Tensor, Phi: torch.Tensor, nv: float) -> torch.Tensor:
    """log N(x; 0, AMP_VAR * Phi Phi^T + nv I), Woodbury, differentiable."""
    m, p = Phi.shape
    G = Phi.T @ Phi + (nv / AMP_VAR) * torch.eye(p, dtype=DTYPE, device=DEV)
    L = torch.linalg.cholesky(G)
    Px = Phi.T @ x
    w = torch.cholesky_solve(Px.unsqueeze(1), L).squeeze(1)
    quad = (x @ x - Px @ w) / nv
    logdet = (m - p) * np.log(nv) + 2.0 * torch.log(
        torch.diagonal(L)).sum() + p * np.log(AMP_VAR)
    return -0.5 * (m * np.log(2 * np.pi) + logdet + quad)


def fit_note_torch(x_np, t_np, midi, n_steps=400, lr=0.4):
    x = torch.tensor(x_np, dtype=DTYPE, device=DEV)
    t = torch.tensor(t_np, dtype=DTYPE, device=DEV)
    J = int(np.clip(round((t_np[-1] - t_np[0]) * KNOTS_PER_S), J_MIN, J_MAX))
    knots_np = np.linspace(t_np[0], t_np[-1], J)
    K = sm_kernel(knots_np[:, None] - knots_np[None, :], W1, MU1, V1, W2, V2)
    K[np.diag_indices_from(K)] += 1e-6 * (W1 + W2)
    Kinv = torch.tensor(np.linalg.inv(K), dtype=DTYPE, device=DEV)
    knots = torch.tensor(knots_np, dtype=DTYPE, device=DEV)

    # coarse c grid, flat curve (same initializer as the NM prototype)
    grid = np.arange(-50.0, 50.0, 1.0)
    nv0 = fit_noise(x_np, t_np, midi, 0.0)
    lls = [loglik(x_np, t_np, midi, c, nv0) for c in grid]
    c0 = float(grid[int(np.argmax(lls))])
    nv = fit_noise(x_np, t_np, midi, c0)

    def interp(u):
        # linear interpolation of knot values onto t (differentiable)
        idx = torch.clamp(torch.searchsorted(knots, t) - 1, 0, J - 2)
        t0, t1 = knots[idx], knots[idx + 1]
        w = (t - t0) / (t1 - t0)
        return (1 - w) * u[idx] + w * u[idx + 1]

    c = torch.tensor(c0, dtype=DTYPE, device=DEV, requires_grad=True)
    u = torch.zeros(J, dtype=DTYPE, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([c, u], lr=lr)
    for step in range(n_steps):
        opt.zero_grad()
        cents = c + interp(u)
        Phi = torch_design(cents, t, midi)
        nll = -torch_loglik(x, Phi, nv) + 0.5 * (u @ (Kinv @ u))
        nll.backward()
        opt.step()
    with torch.no_grad():
        cents = (c + interp(u)).cpu().numpy()
    return float(c.detach().cpu()), cents, knots_np, J


def main():
    import soundfile as sf
    from scipy.signal import resample_poly
    from make_phase2_intro import note_curves
    from eval_phase2_real import dev_unique_tracks
    from score_bundle.phase2.urmp import read_notes_annotation

    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    picks = []
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        for i in range(d["onset"].size):
            du = float(notes["duration"][i])
            if d["ident"][i] and 0.9 <= du <= 1.9 and d["n_frames"][i] >= 60:
                picks.append((key, i, d, tr, notes))
                break
        if len(picks) == 3:
            break

    for key, i, d, tr, notes in picks:
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
        on = float(notes["onset"][i])
        du = min(float(notes["duration"][i]), 2.0)
        a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
        x = audio[max(a, 0):min(b, audio.size)]
        t = np.arange(x.size) / SR
        midi = float(d["midi"][i])

        # equality pin: torch loglik vs numpy at the flat score-nominal curve
        nv = fit_noise(x, t, midi, 0.0)
        ll_np = loglik(x, t, midi, 0.0, nv)
        with torch.no_grad():
            cents0 = torch.zeros(t.size, dtype=DTYPE, device=DEV)
            Phi0 = torch_design(cents0, torch.tensor(t, dtype=DTYPE,
                                                     device=DEV), midi)
            ll_t = float(torch_loglik(torch.tensor(x, dtype=DTYPE,
                                                   device=DEV), Phi0, nv))
        assert abs(ll_np - ll_t) < 1e-4 * max(1.0, abs(ll_np)), (ll_np, ll_t)

        t0 = time.time()
        c_map, cents_map, knots, J = fit_note_torch(x, t, midi)
        wall = time.time() - t0

        tt, xx, ttg, xg = note_curves(key, i, data, f0s, tracks)
        off = 0.02
        sel_g = (ttg >= off) & (ttg <= off + t[-1])
        gt_t, gt_x = ttg[sel_g] - off, xg[sel_g]
        tg = np.linspace(t[0], t[-1], 300)
        curve = np.interp(tg, t, cents_map)
        rmse = float(np.sqrt(np.mean((np.interp(gt_t, tg, curve) - gt_x) ** 2)))
        print(f"{key}#{i} {d['instrument']}: torch MAP {rmse:.2f} c vs GT "
              f"(J={J}, c={c_map:+.1f}) [{wall:.0f}s, loglik pinned "
              f"{ll_t:.1f}≈{ll_np:.1f}]", flush=True)


if __name__ == "__main__":
    main()
