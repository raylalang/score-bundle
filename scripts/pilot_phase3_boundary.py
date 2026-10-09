#!/usr/bin/env python
"""Phase-3 feasibility pilot: a differentiable note boundary (dev).

The open problem (waveform_model.infer_positions; thesis: joint
inference over z): can gradient MAP recover a note boundary from the
waveform?  Smallest honest probe, per docs/ddsp_review.md:

Two-note segments from one dev track.  Each note gets its own cents
curve (c_k + knot interp, the fixed prototype prior) and its own chunked
harmonic design built over the WHOLE segment; the two designs are
multiplied by complementary sigmoid soft windows around a boundary
parameter b (softness ~10 ms) so the likelihood is differentiable in b —
the hard chunk edges of the single-note model cannot carry gradients,
the soft window is the one new ingredient.  Amplitudes collapsed exactly
as everywhere else; FLAT prior on b (the question is whether the
LIKELIHOOD pulls it home, so nothing else may).

Protocol: init b at the annotated boundary + delta for
delta in {-80, -40, +40, +80} ms, Adam over (b, c1, c2, u1, u2),
report |b - b_annot| before/after.  Recovery = final error <= 10 ms
(half a tracker hop) or <= 25% of the initial offset.  A hostile loss
surface (b stuck or diverging) is a finding, not a failure — it is the
measured trigger for considering phase-blind objectives later.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/pilot_phase3_boundary.py
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

import soundfile as sf
import torch
from scipy.signal import resample_poly

from eval_phase3_waveform_dev import SR
from eval_phase2_real import dev_unique_tracks
from score_bundle.phase2.urmp import read_notes_annotation
from score_bundle.phase3.curve import (CurvePrior, chunked_design_torch,
                                       collapsed_loglik_torch, interp_knots,
                                       knot_gram)

TRACK_KEY = (1, 1)                 # violin, the prototype's track
N_PAIRS = 8
DELTAS_MS = (-80.0, -40.0, 40.0, 80.0)
SOFT_S = 0.010                     # sigmoid softness (s)
HALF_SPAN = 0.8                    # segment: boundary +/- this (s)
N_HARM, N_CHUNK, AMP_VAR = 8, 2, 10.0
N_STEPS, LR_CURVE, LR_B_MS = 400, 0.4, 2.0
PRIOR = CurvePrior(knots_per_s=8.0, j_min=4, j_max=16)
OUT_MD = "results/phase3_boundary_pilot_dev.md"


def segment_pairs():
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}
    d, tr = data[TRACK_KEY], tracks[TRACK_KEY]
    notes = read_notes_annotation(tr.notes)
    audio48, sr48 = sf.read(tr.audio)
    audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
    pairs = []
    n = d["onset"].size
    for i in range(n - 1):
        on1, du1 = float(notes["onset"][i]), float(notes["duration"][i])
        on2, du2 = float(notes["onset"][i + 1]), float(notes["duration"][i + 1])
        gap = on2 - (on1 + du1)
        if du1 < 0.4 or du2 < 0.4 or not (-0.01 <= gap <= 0.04):
            continue
        b_annot = on2
        a = max(on1 + 0.02, b_annot - HALF_SPAN)
        z = min(on2 + du2 - 0.02, b_annot + HALF_SPAN)
        sa, sz = int(a * SR), int(z * SR)
        x = audio[sa:sz]
        if x.size < SR // 2:
            continue
        pairs.append({
            "i": i, "x": x, "t0": sa / SR, "b_annot": b_annot,
            "midi1": float(d["midi"][i]), "midi2": float(d["midi"][i + 1]),
            "interval": float(d["midi"][i + 1] - d["midi"][i]),
        })
        if len(pairs) == N_PAIRS:
            break
    return pairs


def fit_boundary(pair, delta_ms):
    """Adam MAP over (b, c1, c2, u1, u2); returns (b_err0_ms, b_err_ms, wall)."""
    x = torch.as_tensor(pair["x"], dtype=torch.float64)
    m = x.shape[0]
    t = torch.as_tensor(pair["t0"] + np.arange(m) / SR, dtype=torch.float64)
    t_np = t.numpy()
    j = PRIOR.n_knots(float(t_np[-1] - t_np[0]))
    knots = torch.as_tensor(np.linspace(t_np[0], t_np[-1], j),
                            dtype=torch.float64)
    _, chol_k = knot_gram(knots, PRIOR)

    def design(b_s, c1, c2, u1, u2):
        w1 = torch.sigmoid((b_s - t) / SOFT_S)
        w2 = torch.sigmoid((t - b_s) / SOFT_S)
        cents1 = c1 + interp_knots(u1, knots, t)
        cents2 = c2 + interp_knots(u2, knots, t)
        Phi1 = chunked_design_torch(cents1, t, pair["midi1"], N_HARM, N_CHUNK)
        Phi2 = chunked_design_torch(cents2, t, pair["midi2"], N_HARM, N_CHUNK)
        return torch.cat([Phi1 * w1.unsqueeze(1), Phi2 * w2.unsqueeze(1)],
                         dim=1)

    b_init = pair["b_annot"] + delta_ms / 1000.0
    zero = torch.zeros(j, dtype=torch.float64)
    with torch.no_grad():
        Phi0 = design(torch.tensor(b_init, dtype=torch.float64),
                      torch.tensor(0.0, dtype=torch.float64),
                      torch.tensor(0.0, dtype=torch.float64), zero, zero)
        beta = torch.linalg.lstsq(Phi0, x.unsqueeze(1)).solution.squeeze(1)
        r = x - Phi0 @ beta
        noise_var = float(r @ r) / max(m - Phi0.shape[1], 1)

    b_ms = torch.tensor(b_init * 1000.0, dtype=torch.float64,
                        requires_grad=True)
    c1 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    c2 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    u1 = zero.clone().requires_grad_(True)
    u2 = zero.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [c1, c2, u1, u2], "lr": LR_CURVE},
                            {"params": [b_ms], "lr": LR_B_MS}])
    t0 = time.time()
    for _ in range(N_STEPS):
        opt.zero_grad()
        Phi = design(b_ms / 1000.0, c1, c2, u1, u2)
        prior = sum(0.5 * (u @ torch.cholesky_solve(
            u.unsqueeze(1), chol_k).squeeze(1)) for u in (u1, u2))
        loss = -collapsed_loglik_torch(x, Phi, noise_var, AMP_VAR) + prior
        loss.backward()
        opt.step()
    b_fin = float(b_ms.detach()) / 1000.0
    return (abs(b_init - pair["b_annot"]) * 1000.0,
            abs(b_fin - pair["b_annot"]) * 1000.0, time.time() - t0)


def main():
    pairs = segment_pairs()
    print(f"{len(pairs)} two-note pairs from track {TRACK_KEY} "
          f"(vn); deltas {DELTAS_MS} ms, softness {SOFT_S * 1000:.0f} ms",
          flush=True)
    rows = []
    for p in pairs:
        for dm in DELTAS_MS:
            e0, e1, wall = fit_boundary(p, dm)
            rec = (p["i"], p["interval"], dm, e0, e1, wall)
            rows.append(rec)
            print(f"pair #{p['i']} (interval {p['interval']:+.0f} st) "
                  f"delta {dm:+.0f} ms: |err| {e0:.0f} -> {e1:.1f} ms "
                  f"[{wall:.0f}s]", flush=True)

    rec_ok = [r for r in rows if r[4] <= 10.0 or r[4] <= 0.25 * r[3]]
    lines = [
        "# Phase 3: differentiable note boundary — feasibility pilot "
        "(DEV, exploratory, no claims)\n",
        "\nTwo-note violin segments, complementary sigmoid windows "
        f"(softness {SOFT_S * 1000:.0f} ms) over per-note\nchunked designs, "
        "amplitudes collapsed, FLAT prior on the boundary b; Adam over\n"
        "(b, c1, c2, u1, u2) from the annotated boundary + delta. "
        "Recovery = final error <= 10 ms\nor <= 25% of the initial offset. "
        "Script: scripts/pilot_phase3_boundary.py.\n",
        f"\n{len(pairs)} pairs x {len(DELTAS_MS)} perturbations = "
        f"{len(rows)} fits; recovered {len(rec_ok)}/{len(rows)}.\n",
        "\n| pair | interval (st) | delta (ms) | err init (ms) | "
        "err final (ms) |\n|---|---|---|---|---|\n"]
    for i, iv, dm, e0, e1, _ in rows:
        lines.append(f"| #{i} | {iv:+.0f} | {dm:+.0f} | {e0:.0f} | "
                     f"{e1:.1f} |\n")
    final_errs = np.array([r[4] for r in rows])
    lines.append(f"\nfinal |err|: median {np.median(final_errs):.1f} ms, "
                 f"q90 {np.quantile(final_errs, .9):.1f} ms; wall per fit "
                 f"median {np.median([r[5] for r in rows]):.0f} s\n")
    open(OUT_MD, "w").writelines(lines)
    print("".join(lines[2:]))
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
