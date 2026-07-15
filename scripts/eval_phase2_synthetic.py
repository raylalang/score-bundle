#!/usr/bin/env python
"""Phase-2 synthetic pilot: the full intonation/vibrato pipeline, end to end.

The first Phase-2 result. On synthetic monophonic pieces with KNOWN per-note
ground truth sampled from a graph GP, run the entire specified pipeline:

    f0/cents curves (eq:vibrato + noise)
      -> per-note joint NLLS estimator (fit_vibrato_note; variances + the
         identifiability rule => per-(note, channel) missingness, for real)
      -> heteroscedastic multi-output graph GP with the new 2-D cell masks
         (estimator variances as per-cell noise scales, per-channel level
          learned by evidence)
      -> exact posterior, scored against the KNOWN truth.

Channels: [c (cents), log gamma, log f].  Three questions answered:
  Q1 imputation   — fully hidden notes: does the graph recover them with
                    calibrated latent intervals? (vs the no-graph ablation)
  Q2 missingness  — vibrato cells the estimator could not identify on VISIBLE
                    notes: the new capability; no baseline can even produce
                    these values from the note itself.
  Q3 denoising    — observed cells: does the GP posterior beat the raw
                    estimator against the truth?

Everything synthetic and deterministic; no real data, no confirmation contact.

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/eval_phase2_synthetic.py \
        | tee logs/phase2_synthetic.log
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from score_bundle.baselines import rich_score_features
from score_bundle.gp import MultiOutputGraphGP, shape_cov
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.phase2.intonation import fit_vibrato_note
from score_bundle.score import Score

_Z90 = 1.6448536269514722
N_PIECES, N_NOTES = 20, 120
S_TRUE = 3.0
MEAN = np.array([5.0, np.log(25.0), np.log(5.5)])      # c, log gamma, log f
B_TRUE = np.array([[15.0 ** 2, 0.5 * 15 * 0.35, 0.0],
                   [0.5 * 15 * 0.35, 0.35 ** 2, 0.3 * 0.35 * 0.10],
                   [0.0, 0.3 * 0.35 * 0.10, 0.10 ** 2]])
CENTS_NOISE, SR = 6.0, 100.0
HOLD_FRAC = 0.30
CH = ["c (cents)", "log gamma", "log f"]


def make_piece(rng):
    onset = np.cumsum(rng.choice([0.3, 0.5, 0.8], size=N_NOTES))
    dur = rng.choice([0.25, 0.4, 0.7, 1.2, 2.0], size=N_NOTES,
                     p=[0.25, 0.2, 0.25, 0.2, 0.1])
    pitch = np.clip(64 + np.cumsum(rng.integers(-3, 4, size=N_NOTES)), 48, 84)
    score = Score.from_arrays(pitch, onset, dur, np.zeros(N_NOTES, dtype=int))
    nu, U = np.linalg.eigh(laplacian(build_adjacency(score)))
    Kg = shape_cov(nu, U, "additive", S_TRUE)
    C = np.kron(B_TRUE, Kg) + 1e-8 * np.eye(3 * N_NOTES)
    y = (MEAN.repeat(N_NOTES)
         + np.linalg.cholesky(C) @ rng.normal(size=3 * N_NOTES))
    y_true = y.reshape(3, N_NOTES).T                    # (N, 3)
    return score, nu, U, dur, y_true


def synth_and_estimate(rng, dur, y_true):
    est = np.full_like(y_true, np.nan)
    var = np.full_like(y_true, np.nan)
    ident = np.zeros(len(dur), dtype=bool)
    for i in range(len(dur)):
        c, g, f = y_true[i, 0], np.exp(y_true[i, 1]), np.exp(y_true[i, 2])
        t = np.arange(0.0, dur[i], 1.0 / SR)
        delta = rng.uniform(0.0, 0.05)
        x = c + g * np.sin(2 * np.pi * f * (t - delta)) \
            + rng.normal(0, CENTS_NOISE, t.size)
        out = fit_vibrato_note(t, x)
        est[i, 0], var[i, 0] = out["c"], out["var_c"]
        ident[i] = out["vibrato_identifiable"]
        if ident[i]:
            g_h, f_h = max(out["gamma"], 1e-6), max(out["f"], 1e-6)
            est[i, 1], var[i, 1] = np.log(g_h), out["var_gamma"] / g_h ** 2
            est[i, 2], var[i, 2] = np.log(f_h), out["var_f"] / f_h ** 2
    return est, var, ident


def score_cells(y_true, m, sd, cells):
    """Per-channel (RMSE, cov@90) lists; None where a channel has no cells."""
    out = []
    for c in range(3):
        sel = cells[:, c]
        if not sel.any():
            out.append(None); continue
        err = y_true[sel, c] - m[sel, c]
        s = sd[sel, c]
        out.append((float(np.sqrt(np.mean(err ** 2))),
                    float(np.mean(np.abs(err) <= _Z90 * s))))
    return out


def main() -> None:
    rng = np.random.default_rng(42)
    acc = {q: {k: ([], []) for k in ("gp", "gp_asgiven", "nograph", "raw")}
           for q in ("impute", "missing", "denoise")}
    n_missing_cells = 0

    for pi in range(N_PIECES):
        score, nu, U, dur, y_true = make_piece(rng)
        est, var, ident = synth_and_estimate(rng, dur, y_true)
        held = rng.random(N_NOTES) < HOLD_FRAC

        mask = np.zeros((N_NOTES, 3), dtype=bool)
        mask[:, 0] = ~held                              # c observed if visible
        mask[:, 1] = mask[:, 2] = (~held) & ident       # vibrato if identifiable
        n_missing_cells += int(((~held) & ~ident).sum()) * 2

        # per-cell noise scales: estimator variances, relative per channel
        scale = np.ones((N_NOTES, 3))
        for c in range(3):
            v = var[:, c][mask[:, c]]
            med = np.median(v) if v.size else 1.0
            scale[:, c] = np.where(np.isfinite(var[:, c]),
                                   np.clip(var[:, c] / max(med, 1e-12),
                                           1e-2, 1e3), 1.0)
        Yobs = np.where(mask, est, 0.0)                 # masked cells never read

        X = rich_score_features(score, rff_dim=0)
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        feats = [np.concatenate([X, np.ones((N_NOTES, 1))], axis=1)]
        floor = 0.05 * np.array([float(np.var(Yobs[mask[:, c], c]))
                                 for c in range(3)])

        med_var = np.array([np.median(var[:, c][mask[:, c]])
                            if mask[:, c].any() else 1.0 for c in range(3)])
        fits = {}
        for name, kern, fixed in (("gp", "additive", None),
                                  ("gp_asgiven", "additive", med_var),
                                  ("nograph", "none", None)):
            g = MultiOutputGraphGP(nu, U, kernel=kern, features=feats,
                                   n_channels=3)
            g.noise_scale = scale
            x_hat, _ = g.fit(Yobs, mask, noise_floor=floor, maxiter=200,
                             noise_fixed=fixed)
            m, sd = g.posterior(Yobs, mask, x_hat)
            fits[name] = (m, sd)

        cells_imp = np.repeat(held[:, None], 3, axis=1)
        cells_mis = np.zeros_like(mask); cells_mis[:, 1:] = \
            ((~held) & ~ident)[:, None]
        cells_obs = mask
        for name, (m, sd) in fits.items():
            for q, cells in (("impute", cells_imp), ("missing", cells_mis),
                             ("denoise", cells_obs)):
                if cells.sum() == 0:
                    continue
                acc[q][name][0].append(score_cells(y_true, m, sd, cells))
        # raw estimator where it exists (visible identifiable cells + hidden?
        # raw needs the note's own curve, so it exists wherever est is finite)
        raw_sd = np.sqrt(var)
        for q, cells in (("impute", cells_imp & np.isfinite(est)),
                         ("denoise", cells_obs)):
            acc[q]["raw"][0].append(score_cells(y_true, est, raw_sd, cells))

    print("== Phase-2 synthetic pilot (20 pieces x 120 notes, truth known) ==")
    print(f"estimator-unidentifiable vibrato cells on visible notes: "
          f"{n_missing_cells} (the per-(note,channel) missingness case)\n")
    label = {"gp": "graph GP (learned noise scale)",
             "gp_asgiven": "graph GP (estimator variances as given)",
             "nograph": "no-graph ablation",
             "raw": "estimator (ORACLE on hidden notes: sees their curves)"}
    for q, title in (("impute", "Q1 imputation (fully hidden notes)"),
                     ("missing", "Q2 estimator-missing vibrato cells (visible notes)"),
                     ("denoise", "Q3 denoising (observed cells)")):
        print(title)
        for name in ("gp", "gp_asgiven", "nograph", "raw"):
            rows = acc[q][name][0]
            if not rows:
                if name == "raw" and q == "missing":
                    print(f"  {label[name]:<52} undefined -- the estimator has "
                          f"no value at these cells; only the prior can answer")
                continue
            parts = []
            for c in range(3):
                vals = [r[c] for r in rows if r[c] is not None]
                if not vals:
                    parts.append(f"{CH[c]}: --")
                    continue
                rm = np.mean([v[0] for v in vals])
                cv = np.mean([v[1] for v in vals])
                parts.append(f"{CH[c]}: {rm:.3f}/{cv:.2f}")
            print(f"  {label[name]:<52} " + "   ".join(parts))
        print()

    os.makedirs("results", exist_ok=True)
    with open("results/phase2_synthetic.pkl", "wb") as fh:
        pickle.dump({"acc": {q: {k: list(v[0]) for k, v in d.items()}
                             for q, d in acc.items()},
                     "meta": {"pieces": N_PIECES, "notes": N_NOTES,
                              "channels": CH, "s_true": S_TRUE,
                              "B_true": B_TRUE.tolist(),
                              "cents_noise": CENTS_NOISE,
                              "hold_frac": HOLD_FRAC, "seed": 42}}, fh)
    print("wrote results/phase2_synthetic.pkl")


if __name__ == "__main__":
    main()
