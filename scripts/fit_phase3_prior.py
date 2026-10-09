#!/usr/bin/env python
"""Evidence-based fit of the Phase-3 curve-prior hyperparameters (dev).

The prototype prior (curve.CurvePrior defaults: W1=30, MU1=5.5, V1=0.4,
W2=60, V2=0.3) is hand-fixed scales.  This study fits those five
hyperparameters by coordinate ascent on the summed per-note Laplace
evidence over a small dev subset, SHARED across notes (not per-note —
the spectral-mixture study's lesson: per-note freedom has nothing to
discover here):

  round r: refit every note's MAP at the current hyperparameters
           (curve.fit_note), freeze each MAP and its likelihood-only
           curvature (curve.loglik_hessian), then Adam-ascend the
           hyperparameters through curve.evidence_fixed_map (the only
           phi-dependent terms are the prior quadratic/normalizer and
           the Laplace log det — exact at the frozen MAP, envelope
           first-order in the MAP's own movement).

Reparameterization keeps everything positive (log for w1, v1, w2, v2)
and the vibrato mean in a physical band (mu1 = 0.5 + 14.5 sigmoid).

Gate (recorded either way): the fitted prior must not degrade GT-frame
RMSE/coverage on the three prototype notes of
results/phase3_curve_proto_dev.md vs the fixed prior.

Outputs: .cache/phase3_prior_fit.json + printed record material.
DEV tracks only; exploratory; no claims.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/fit_phase3_prior.py
"""
from __future__ import annotations

import json
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

from eval_phase3_waveform_dev import SR, selected
from eval_phase2_real import dev_unique_tracks
from make_phase2_intro import note_curves
from score_bundle.phase2.urmp import read_notes_annotation
from score_bundle.phase3.curve import (CurvePrior, chunked_design_torch,
                                       collapsed_loglik_torch,
                                       evidence_fixed_map, fit_note,
                                       interp_knots, loglik_hessian)

N_PER_TRACK = 4
N_ROUNDS = 3
MAP_STEPS = 300
PHI_STEPS = 80
PHI_LR = 0.05
OUT_JSON = ".cache/phase3_prior_fit.json"
BASE = CurvePrior()


def hypers_from_psi(psi):
    """psi (5,) -> dict of torch scalars (w1, mu1, v1, w2, v2)."""
    return {
        "w1": torch.exp(psi[0]),
        "mu1": 0.5 + 14.5 * torch.sigmoid(psi[1]),
        "v1": torch.exp(psi[2]),
        "w2": torch.exp(psi[3]),
        "v2": torch.exp(psi[4]),
    }


def psi_from_floats(w1, mu1, v1, w2, v2):
    s = (mu1 - 0.5) / 14.5
    return torch.tensor([np.log(w1), np.log(s / (1.0 - s)),
                         np.log(v1), np.log(w2), np.log(v2)],
                        dtype=torch.float64)


def floats_from_psi(psi):
    return {k: float(v) for k, v in hypers_from_psi(psi.detach()).items()}


def load_segments():
    """(label, x, t, midi) for up to N_PER_TRACK eligible notes per track."""
    segs = []
    proto_picks = []
    for key, d, tr in selected():
        notes = read_notes_annotation(tr.notes)
        audio48, sr48 = sf.read(tr.audio)
        audio = resample_poly(np.asarray(audio48, float), SR, int(sr48))
        elig = [i for i in range(d["onset"].size)
                if d["ident"][i] and 0.5 <= float(notes["duration"][i]) <= 2.0
                and d["n_frames"][i] >= 40]
        take = elig[:: max(1, len(elig) // N_PER_TRACK)][:N_PER_TRACK]
        for i in take:
            on = float(notes["onset"][i])
            du = min(float(notes["duration"][i]), 2.0)
            a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
            x = audio[max(a, 0):min(b, audio.size)]
            t = np.arange(x.size) / SR
            segs.append((f"{key}#{i} {d['instrument']}", x, t,
                         float(d["midi"][i])))
        # the three prototype notes (first per track matching its rule)
        if len(proto_picks) < 3:
            for i in range(d["onset"].size):
                du = float(notes["duration"][i])
                if d["ident"][i] and 0.9 <= du <= 1.9 and d["n_frames"][i] >= 60:
                    on = float(notes["onset"][i])
                    du = min(du, 2.0)
                    a, b = int((on + 0.02) * SR), int((on + du - 0.02) * SR)
                    proto_picks.append(
                        (key, i, d["instrument"],
                         audio[max(a, 0):min(b, audio.size)],
                         float(d["midi"][i])))
                    break
    return segs, proto_picks


def map_pass(segs, hypers_f, tag):
    """Fit every note's MAP at the given hyperparameters (floats).

    Returns per-note frozen pieces for the phi step and the summed evidence.
    """
    frozen, sum_ev = [], 0.0
    for n, (label, x, t, midi) in enumerate(segs):
        t0 = time.time()
        fit = fit_note(x, t, midi, BASE, n_steps=MAP_STEPS, hypers=hypers_f)
        x_t = torch.as_tensor(x, dtype=torch.float64)
        t_t = torch.as_tensor(t, dtype=torch.float64)
        knots = torch.as_tensor(fit.knots, dtype=torch.float64)
        u_t = torch.as_tensor(fit.u, dtype=torch.float64)
        cents = fit.c + interp_knots(u_t, knots, t_t)
        Phi = chunked_design_torch(cents, t_t, midi, 8, 4)
        ll_map = float(collapsed_loglik_torch(x_t, Phi, fit.noise_var, 10.0))
        H_lik = torch.as_tensor(loglik_hessian(x, t, midi, fit),
                                dtype=torch.float64)
        frozen.append((ll_map, u_t, knots, H_lik))
        sum_ev += fit.log_evidence
        print(f"  [{tag}] {n + 1}/{len(segs)} {label}: "
              f"logev {fit.log_evidence:.0f} [{time.time() - t0:.0f}s]",
              flush=True)
    return frozen, sum_ev


def phi_step(frozen, psi):
    psi = psi.clone().requires_grad_(True)
    opt = torch.optim.Adam([psi], lr=PHI_LR)
    for _ in range(PHI_STEPS):
        opt.zero_grad()
        hyp = hypers_from_psi(psi)
        loss = -sum(evidence_fixed_map(ll, u, kn, H, BASE, hyp)
                    for ll, u, kn, H in frozen)
        loss.backward()
        opt.step()
    return psi.detach(), float(-loss)


def gate(proto_picks, data, f0s, tracks, hypers_f, tag):
    rows = []
    for key, i, inst, x, midi in proto_picks:
        t = np.arange(x.size) / SR
        fit = fit_note(x, t, midi, BASE, hypers=hypers_f)
        tt, xx, ttg, xg = note_curves(key, i, data, f0s, tracks)
        off = 0.02
        sel = (ttg >= off) & (ttg <= off + t[-1])
        gt_t, gt_x = ttg[sel] - off, xg[sel]
        mean, sd = fit.curve(gt_t)
        rmse = float(np.sqrt(np.mean((mean - gt_x) ** 2)))
        cov = float(np.mean(np.abs(mean - gt_x) <= 1.645 * sd))
        rows.append((f"{key}#{i} {inst}", rmse, cov))
        print(f"  [gate {tag}] {key}#{i} {inst}: {rmse:.2f} c, "
              f"cover90 {cov:.2f}", flush=True)
    return rows


def main():
    data = pickle.load(open(".cache/urmp_targets_dev.pkl", "rb"))
    f0s = pickle.load(open(".cache/urmp_f0_dev.pkl", "rb"))
    tracks = {(p.index, t.number): t for p, t in dev_unique_tracks()}

    segs, proto_picks = load_segments()
    print(f"subset: {len(segs)} notes from {len(selected())} dev tracks; "
          f"{N_ROUNDS} rounds x (MAP pass + {PHI_STEPS} phi steps)",
          flush=True)

    psi = psi_from_floats(BASE.w1, BASE.mu1, BASE.v1, BASE.w2, BASE.v2)
    sum_ev_fixed = None
    for r in range(N_ROUNDS):
        hyp_f = floats_from_psi(psi)
        print(f"round {r}: hypers {({k: round(v, 3) for k, v in hyp_f.items()})}",
              flush=True)
        frozen, sum_ev = map_pass(segs, hyp_f, f"r{r}")
        if r == 0:
            sum_ev_fixed = sum_ev
        psi, ev_after = phi_step(frozen, psi)
        print(f"round {r}: sum logev {sum_ev:.1f} -> {ev_after:.1f} "
              f"(frozen-MAP) -> {floats_from_psi(psi)}", flush=True)

    hyp_fitted = floats_from_psi(psi)
    _, sum_ev_fitted = map_pass(segs, hyp_fitted, "final")
    print(f"SUM EVIDENCE: fixed {sum_ev_fixed:.1f} vs fitted "
          f"{sum_ev_fitted:.1f} (delta {sum_ev_fitted - sum_ev_fixed:+.1f}, "
          f"n={len(segs)})", flush=True)

    rows_fixed = gate(proto_picks, data, f0s, tracks, None, "fixed")
    rows_fit = gate(proto_picks, data, f0s, tracks, hyp_fitted, "fitted")

    os.makedirs(".cache", exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump({"hypers": hyp_fitted, "sum_ev_fixed": sum_ev_fixed,
                   "sum_ev_fitted": sum_ev_fitted, "n_notes": len(segs),
                   "gate_fixed": rows_fixed, "gate_fitted": rows_fit,
                   "base": {"w1": BASE.w1, "mu1": BASE.mu1, "v1": BASE.v1,
                            "w2": BASE.w2, "v2": BASE.v2}}, fh, indent=1)
    print("wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
