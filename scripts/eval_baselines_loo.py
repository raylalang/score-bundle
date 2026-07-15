#!/usr/bin/env python
"""Leave-one-out for the two external baselines (validation pieces).

Fills the two missing LOO grid cells:

* mean + smoothing (best): the cross-piece feat+LM mean (head fit on the
  training pieces, applied via emb_leakfree, exactly the published head
  protocol), then the per-channel graph smoother fit on the full residual
  and scored with the scalar LOO identity (downstream.loo_predictive) —
  the same estimator the joint-GP LOO uses.
* MLP ensemble: each note is predicted from its own features only, so
  leave-one-out equals full evaluation with the leak-free embeddings
  (the note's own targets never enter its features).

    PYTHONPATH=src:scripts python scripts/eval_baselines_loo.py
"""
from __future__ import annotations

import numpy as np

_Z90 = 1.6448536269514722


def perch(y, m, sd):
    out = []
    for c in range(3):
        e = y[:, c] - m[:, c]
        v = sd[:, c] ** 2
        nll = float(np.mean(0.5 * (np.log(2.0 * np.pi * v) + e ** 2 / v)))
        out.append((float(np.sqrt(np.mean(e ** 2))),
                    float(np.mean(np.abs(e) <= _Z90 * sd[:, c])), nll))
    return out


def main() -> None:
    from score_bundle.baselines import rich_score_features
    from score_bundle.downstream import load_piece_arrays, piece_score
    from score_bundle.graph import build_adjacency_harmonic, laplacian
    from score_bundle.lm import features as lmfeat
    from score_bundle.model import fit_spectral_field_guarded

    head, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    ev = ev[:30]

    def rep(p):
        X = rich_score_features(piece_score(p), rff_dim=0)
        return np.concatenate([X, np.asarray(p["emb_leakfree"], np.float64)],
                              axis=1)

    # cross-piece head, exactly the published protocol (feat_lm, l2=10)
    H = np.concatenate([rep(p) for p in head])
    Yh = np.concatenate([p["y"] for p in head])
    W_lf = lmfeat.fit_prior_mean_head(H, Yh, l2=10.0)

    names = ["tau", "logr", "v"]
    dump = {}

    # ---- mean + smoothing (best): scalar LOO identity per channel ----------
    ys, ms, sds = [], [], []
    for pi, p in enumerate(ev):
        y = np.asarray(p["y"], dtype=float)
        mu = lmfeat.apply_prior_mean(rep(p), W_lf)
        L = laplacian(build_adjacency_harmonic(piece_score(p),
                                               chord_weight=1.0, vl_weight=1.0))
        eig = np.linalg.eigh(L)
        mask = np.ones(len(y), dtype=bool)
        m_all = np.empty_like(y)
        s_all = np.empty_like(y)
        for c in range(3):
            floor = 0.05 * float(np.var(y[:, c] - mu[:, c]))
            field, hp = fit_spectral_field_guarded(
                None, y[:, c], kernel="additive", mask=mask, mean=mu[:, c],
                noise_floor=floor, rng=np.random.default_rng(0), eig=eig)
            # scalar LOO identity on C = K + nv I, r = y - mean
            C = field.K + hp["noise_var"] * np.eye(field.N)
            P = np.linalg.inv(C)
            r_ = y[:, c] - np.asarray(field.mean, dtype=float)
            dii = np.clip(np.diag(P), 1e-12, None)
            m_all[:, c] = y[:, c] - (P @ r_) / dii
            s_all[:, c] = np.sqrt(1.0 / dii)
        ys.append(y); ms.append(m_all); sds.append(s_all)
        if (pi + 1) % 10 == 0:
            print(f"mean+smoothing: {pi + 1}/30 pieces", flush=True)
    r = perch(np.concatenate(ys), np.concatenate(ms), np.concatenate(sds))
    print("mean+smoothing LOO  " + "  ".join(
        f"{names[c]}: {r[c][0]:.3f}/{r[c][1]:.2f}/nll {r[c][2]:+.3f}"
        for c in range(3)))
    dump["mean_smoothing"] = (np.concatenate(ys), np.concatenate(ms),
                              np.concatenate(sds))

    # ---- MLP ensemble: pointwise predictor, LOO = full evaluation ----------
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_deep_baseline import build_inputs, train_members
    import torch
    device = "cuda:1" if torch.cuda.device_count() > 1 else (
        "cuda" if torch.cuda.is_available() else "cpu")
    members = train_members(head, list(range(5)), device)
    ys, ms, sds = [], [], []
    for p in ev:
        y = np.asarray(p["y"], dtype=float)
        X = torch.tensor(build_inputs(p, p["emb_leakfree"]), device=device)
        with torch.no_grad():
            outs = [mem(X).cpu().numpy() for mem in members]
        mus = np.stack([o[:, :3] for o in outs])
        v_s = np.stack([np.exp(np.clip(o[:, 3:], -10, 6)) for o in outs])
        mu = mus.mean(0)
        var = (v_s + mus ** 2).mean(0) - mu ** 2
        floor = 0.05 * y.var(axis=0)
        ys.append(y); ms.append(mu)
        sds.append(np.sqrt(np.maximum(var, floor[None, :])))
    r = perch(np.concatenate(ys), np.concatenate(ms), np.concatenate(sds))
    print("MLP ensemble LOO    " + "  ".join(
        f"{names[c]}: {r[c][0]:.3f}/{r[c][1]:.2f}/nll {r[c][2]:+.3f}"
        for c in range(3)))
    dump["mlp_ensemble"] = (np.concatenate(ys), np.concatenate(ms),
                            np.concatenate(sds))
    import pickle
    with open("results/baselines_loo.pkl", "wb") as fh:
        pickle.dump(dump, fh)
    print("wrote results/baselines_loo.pkl")


if __name__ == "__main__":
    main()
