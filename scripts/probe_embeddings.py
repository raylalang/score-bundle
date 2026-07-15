#!/usr/bin/env python
"""Linear probes: does the music model's embedding encode music theory?

Motivated by two redundancy negatives (harmonic edges and the 14 theory feature
columns both add nothing next to the embeddings): those results only *infer*
that the embeddings carry the tonal/metrical/repetition signal. This study
demonstrates (or refutes) it directly.

Protocol (DEV-only; confirmation pieces never read): for each of the 14
music-theory columns of ``baselines._theory_block``, fit a cross-piece ridge
probe on the HEAD pieces and score it on the 30 DEV pieces — once from the
leak-free music-model embeddings (``emb_leakfree``, 512-d), and once from the
25 hand-built base score features (the control: if the hand-built features
predict a theory column equally well, the embedding carries nothing special
there). Continuous columns report out-of-sample R^2; binary columns
additionally report AUC. Inputs are z-scored per piece, targets per piece for
R^2 comparability. Deterministic; no confirmation contact.

    OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/probe_embeddings.py
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

THEORY_COLS = ["key_clarity", "mode_major", "deg_sin", "deg_cos", "in_scale",
               "fifths_motion", "metric_weight", "dissonance", "is_bass",
               "lbdm_ioi", "lbdm_pitch", "repeat_count", "v_pitch_z", "v_pos"]
BINARY = {"mode_major", "in_scale", "is_bass"}


def _rff(X, dim, seed):
    rng = np.random.default_rng(seed)
    G = rng.normal(0.0, 1.0 / np.sqrt(X.shape[1]), size=(X.shape[1], dim))
    phase = rng.uniform(0.0, 2 * np.pi, size=dim)
    return np.sqrt(2.0 / dim) * np.cos(X @ G + phase)


def zscore_cols(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / np.maximum(sd, 1e-9)


def ridge_fit(X, y, l2):
    XtX = X.T @ X + l2 * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--rff-dim", type=int, default=0,
                    help="apply deterministic random Fourier features of this "
                         "dimension to the inputs before the ridge probe "
                         "(a nonlinear probe; 0 = linear)")
    ap.add_argument("--raw", action="store_true",
                    help="skip per-piece z-scoring of inputs (probes piece-level "
                         "information the GP's per-piece-standardized kernel cannot use)")
    ap.add_argument("--out", default="results/probe_embeddings.pkl")
    args = ap.parse_args()

    from score_bundle.baselines import _theory_block, rich_score_features
    from score_bundle.downstream import load_piece_arrays, piece_score

    head, ev, _ = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]  # DEV head only

    def rep(p):
        score = piece_score(p)
        order = np.lexsort((score.pitch, score.onset))
        T = np.empty((len(score), 14))
        T[order] = _theory_block(score.pitch.astype(float)[order],
                                 score.onset.astype(float)[order],
                                 score.duration.astype(float)[order],
                                 score.voice.astype(float)[order])
        emb = np.asarray(p["emb_leakfree"], dtype=np.float64)
        feat = rich_score_features(score, rff_dim=0)
        if not args.raw:
            emb, feat = zscore_cols(emb), zscore_cols(feat)
        if args.rff_dim > 0:
            emb, feat = _rff(emb, args.rff_dim, 0), _rff(feat, args.rff_dim, 1)
        one = np.ones((len(T), 1))
        return (np.concatenate([emb, one], axis=1),
                np.concatenate([feat, one], axis=1), T)

    print("building representations...", flush=True)
    head_rep = [rep(p) for p in head]
    dev_rep = [rep(p) for p in ev]

    He = np.concatenate([r[0] for r in head_rep])
    Hf = np.concatenate([r[1] for r in head_rep])
    Ht = np.concatenate([r[2] for r in head_rep])
    De = np.concatenate([r[0] for r in dev_rep])
    Df = np.concatenate([r[1] for r in dev_rep])
    Dt = np.concatenate([r[2] for r in dev_rep])

    rows = []
    print(f"{'theory column':<15} {'R2(emb)':>8} {'R2(feat)':>9} "
          f"{'AUC(emb)':>9} {'AUC(feat)':>10}")
    for j, name in enumerate(THEORY_COLS):
        y_tr, y_te = Ht[:, j], Dt[:, j]
        mu, sd = y_tr.mean(), max(y_tr.std(), 1e-9)
        out = {"col": name}
        for tag, Xtr, Xte in (("emb", He, De), ("feat", Hf, Df)):
            w = ridge_fit(Xtr, (y_tr - mu) / sd, args.l2)
            pred = Xte @ w * sd + mu
            ss_res = float(np.sum((y_te - pred) ** 2))
            ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
            out[f"r2_{tag}"] = 1.0 - ss_res / max(ss_tot, 1e-12)
            out[f"auc_{tag}"] = (auc(pred, y_te) if name in BINARY
                                 else float("nan"))
        rows.append(out)
        print(f"{name:<15} {out['r2_emb']:8.3f} {out['r2_feat']:9.3f} "
              f"{out['auc_emb']:9.3f} {out['auc_feat']:10.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"rows": rows, "meta": {
            "probe": "ridge, head pieces -> dev pieces (cross-piece, out-of-sample)",
            "l2": args.l2, "n_head": len(head), "n_dev": len(ev),
            "emb": "emb_leakfree (512-d) + bias", "control": "25 base features + bias",
            "targets": "baselines._theory_block (14 cols)",
            "note": "DEV only; confirmation never read"}}, fh)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
