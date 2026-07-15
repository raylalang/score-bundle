#!/usr/bin/env python
"""Calibrated DEEP baselines under the strict protocol (DEV) — the missing rival.

The one baseline the thesis lacked: a deep model that produces its own
per-note uncertainty. Two systems, both consuming exactly the proposed model's
information set (25 hand-built features + music-model embeddings, per-piece
standardized), trained cross-piece on the head pieces and evaluated on the
identical 30 dev pieces x 4 published anchor masks:

  hetero-mlp     an MLP head with a heteroscedastic Gaussian output
                 (mu_c(x), log sigma_c^2(x) per channel), Gaussian-NLL trained;
  deep-ensemble  five such heads (seeds 0-4); predictive = mixture moments.

Fairness: evaluation embeddings are the same strict mask-aware dump the GP's
feature kernel uses; predictive variances receive the same 5%-of-observed
per-piece floor the GP's noise floor provides (both floored and unfloored
numbers are stored). Cells are written in the standard (yt, pr, sd, ch)
schema so every paired tool applies. DEV only; confirmation untouched.

    PYTHONPATH=src:scripts python scripts/eval_deep_baseline.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import INPUTS, bootstrap_ci, zscore_cols  # noqa: E402

OUT_DIR = "results/deep_baseline"
N_ENSEMBLE = 5
FLOOR_FRAC = 0.05


def build_inputs(p, emb):
    from score_bundle.baselines import rich_score_features
    from score_bundle.downstream import piece_score
    X = zscore_cols(rich_score_features(piece_score(p), rff_dim=0))
    E = zscore_cols(np.asarray(emb, dtype=np.float64))
    return np.concatenate([X, E], axis=1).astype(np.float32)


def train_members(head, seeds, device):
    import torch
    from torch import nn

    Xtr = np.concatenate([build_inputs(p, p["emb_leakfree"]) for p in head])
    Ytr = np.concatenate([np.asarray(p["y"], dtype=np.float32) for p in head])
    # piece-level train/val split (8 val pieces), fixed
    sizes = [len(p["y"]) for p in head]
    owner = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    val_pieces = set(range(32, 40))
    tr_idx = np.flatnonzero(~np.isin(owner, list(val_pieces)))
    va_idx = np.flatnonzero(np.isin(owner, list(val_pieces)))

    def make_model(seed):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 6)).to(device)

    def gauss_nll(out, y):
        mu, logvar = out[:, :3], out[:, 3:].clamp(-10, 6)
        return (0.5 * (logvar + (y - mu) ** 2 / logvar.exp())).mean()

    Xt = torch.tensor(Xtr, device=device)
    Yt = torch.tensor(Ytr, device=device)
    members = []
    for seed in seeds:
        model = make_model(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(1000 + seed)
        best, best_state, patience = np.inf, None, 0
        for epoch in range(400):
            model.train()
            perm = torch.randperm(len(tr_idx), generator=g).numpy()
            for k in range(0, len(perm), 512):
                idx = tr_idx[perm[k:k + 512]]
                opt.zero_grad()
                loss = gauss_nll(model(Xt[idx]), Yt[idx])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(gauss_nll(model(Xt[va_idx]), Yt[va_idx]))
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        members.append(model)
        print(f"member seed {seed}: val NLL {best:.4f} ({epoch + 1} epochs)",
              flush=True)
    return members


def main() -> None:
    import torch
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.metrics import evaluate

    device = "cuda:1" if torch.cuda.device_count() > 1 else (
        "cuda" if torch.cuda.is_available() else "cpu")
    head, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    ev = ev[:30]
    RATES = {  # tag -> (inputs pkl, embedding dump)
        "obs0.60": (INPUTS, ".cache/kernel_sweep_emb_ma.pkl"),
        "obs0.50": (".cache/masksweep_inputs_obs0.50.pkl", ".cache/masksweep_emb_obs0.50.pkl"),
        "obs0.70": (".cache/masksweep_inputs_obs0.70.pkl", ".cache/masksweep_emb_obs0.70.pkl"),
        "obs0.80": (".cache/masksweep_inputs_obs0.80.pkl", ".cache/masksweep_emb_obs0.80.pkl"),
        "obs0.90": (".cache/masksweep_inputs_obs0.90.pkl", ".cache/masksweep_emb_obs0.90.pkl"),
    }
    members = train_members(head, list(range(N_ENSEMBLE)), device)

    for tag, (inp_path, emb_path) in RATES.items():
      with open(inp_path, "rb") as fh:
          masks = pickle.load(fh)["masks"]
      with open(emb_path, "rb") as fh:
          emb_dump = pickle.load(fh)["emb_ma"]
      cells = {"hetero": {}, "ensemble": {}}
      for s in range(4):
          for pi, p in enumerate(ev):
            Y = np.asarray(p["y"], dtype=float)
            mask = masks[(pi, s)]
            held = ~mask
            X = torch.tensor(build_inputs(p, emb_dump[(pi, s)]), device=device)
            with torch.no_grad():
                outs = [m(X).cpu().numpy() for m in members]
            mus = np.stack([o[:, :3] for o in outs])
            v_s = np.stack([np.exp(np.clip(o[:, 3:], -10, 6)) for o in outs])
            floor = FLOOR_FRAC * np.array([float(np.var(Y[mask, c]))
                                           for c in range(3)])
            for name, mu, var in (
                    ("hetero", mus[0], v_s[0]),
                    ("ensemble", mus.mean(0),
                     (v_s + mus ** 2).mean(0) - mus.mean(0) ** 2)):
                var_f = np.maximum(var, floor[None, :])
                yt = np.concatenate([Y[held, c] for c in range(3)])
                pr = np.concatenate([mu[held, c] for c in range(3)])
                sd = np.concatenate([np.sqrt(var_f[held, c]) for c in range(3)])
                sd_raw = np.concatenate([np.sqrt(var[held, c]) for c in range(3)])
                ch = np.concatenate([np.full(int(held.sum()), c, dtype=int)
                                     for c in range(3)])
                cells[name][("GP", pi, s)] = (yt, pr, sd, ch)
                cells[name].setdefault("_raw_sd", {})[(pi, s)] = sd_raw

      out_dir = OUT_DIR if tag == "obs0.60" else f"{OUT_DIR}_{tag}"
      os.makedirs(out_dir, exist_ok=True)
      print(f"\n=== {tag} ===")
      for name in ("hetero", "ensemble"):
        raw_sd = cells[name].pop("_raw_sd")
        c = cells[name]
        yt = np.concatenate([v[0] for v in c.values()])
        pr = np.concatenate([v[1] for v in c.values()])
        sd = np.concatenate([v[2] for v in c.values()])
        m = evaluate(yt, pr, sd, level=0.9)
        print(f"{name:<22} {m['rmse']:8.4f} {m['nll']:8.3f} "
              f"{m['coverage@0.90']:7.3f}")
        with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as fh:
            pickle.dump({"row": name, "cells": c, "raw_sd": raw_sd,
                         "meta": {"protocol": f"strict masks {tag}, validation",
                                  "floor_frac": FLOOR_FRAC,
                                  "members": N_ENSEMBLE,
                                  "inputs": "feat25+emb (mask-aware at eval)"}},
                        fh)


if __name__ == "__main__":
    main()
