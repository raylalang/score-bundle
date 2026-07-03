#!/usr/bin/env python
"""Strict-protocol check for the feat+LM candidate headline.

feat+LM+graph confirmed at the published l2=10 head (0.3879 / -0.347, significantly
better than LM+graph on both RMSE and NLL — logs/feature_baseline_l2_10.log). Last
gate before adoption: the strict mask-aware protocol. Features are score-only (mask-
independent); only the LM embedding side changes — held-out notes' velocities are
replaced by the placeholder in the LM input, embeddings recomputed per (piece, seed).

Variants (identical masks; graph off/on):
    feat+LM      cached emb_leakfree + features        — this morning's confirmation
    feat+LM-ma   mask-aware embeddings + features      — the strict candidate

    python scripts/eval_featlm_strict.py --checkpoint checkpoints/maestro_scaled/best.pt
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def bootstrap_ci(vals, B=2000, rng=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--checkpoint", default="checkpoints/maestro_scaled/best.pt")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--rff-dim", type=int, default=256)
    ap.add_argument("--noise-floor-frac", type=float, default=0.05)
    ap.add_argument("--placeholder-vel", type=int, default=64)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("Needs PyTorch:  pip install -e '.[train]'"); sys.exit(1)

    from score_bundle import imputation_eval as ie
    from score_bundle.baselines import rich_score_features
    from score_bundle.downstream import load_piece_arrays, piece_score
    from score_bundle.lm import features as lmfeat
    from score_bundle.lm.model_torch import build_model
    from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent
    from score_bundle.metrics import evaluate

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    tok = MidiTokenizer()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    print(f"{len(head)} head + {len(ev)} eval | l2={args.l2} | guard={args.guard} "
          f"| device {device}", flush=True)

    def feats(p):
        return rich_score_features(piece_score(p), rff_dim=args.rff_dim)

    def notes_of(p, vel):
        return [NoteEvent(int(pi), float(oi), float(di), int(np.clip(v, 1, 127)))
                for pi, oi, di, v in zip(p["pitch"], p["onset"], p["duration"], vel)]

    def maskaware_emb(p, mask):
        vel = np.where(mask, p["velocity"], float(args.placeholder_vel))
        return lmfeat.note_embeddings_long(model, tok, notes_of(p, vel),
                                           readout="pre_velocity")

    Yh = np.concatenate([p["y"] for p in head])
    H_cat = np.concatenate([np.concatenate([feats(p), p["emb_leakfree"]], axis=1)
                            for p in head])
    W = lmfeat.fit_prior_mean_head(H_cat, Yh, l2=args.l2)
    print("head fit (feat + emb_leakfree concat)", flush=True)

    pool, per_piece = {}, {}
    for s in range(args.seeds):
        seed_rng = np.random.default_rng(1000 + s)
        for pi, p in enumerate(ev):
            score, y = piece_score(p), p["y"]
            F = feats(p)
            mask = ie.random_mask(len(y), seed_rng, observed_frac=args.observed_frac)
            X_lf = np.concatenate([F, p["emb_leakfree"]], axis=1)
            X_ma = np.concatenate([F, maskaware_emb(p, mask)], axis=1)
            means = {
                "feat+LM": lmfeat.apply_prior_mean(X_lf, W),
                "feat+LM-ma": lmfeat.apply_prior_mean(X_ma, W),
            }
            cells = ie.impute_methods(score, y, means, mask, fit_hyper=True,
                                      rng=seed_rng, guard=args.guard,
                                      noise_floor_frac=args.noise_floor_frac)
            for key, cell in cells.items():
                buf = pool.setdefault(key, [[], [], []])
                for i, arr in enumerate((cell.y, cell.pred, cell.std)):
                    buf[i].append(arr)
                pp = per_piece.setdefault(key, {}).setdefault(pi, [[], [], []])
                pp[0].append(cell.y); pp[1].append(cell.pred); pp[2].append(cell.std)
        print(f"seed {s + 1}/{args.seeds} done", flush=True)

    print("\nPooled (identical masks):")
    print(f"{'variant':12s} {'graph':6s} {'RMSE':>8s} {'NLL':>9s} {'cov@.9':>8s}")
    for key in sorted(pool, key=str):
        m = evaluate(*[np.concatenate(v) for v in pool[key]], level=0.9)
        print(f"{key[0]:12s} {('on' if key[1] else 'off'):6s} {m['rmse']:8.4f} "
              f"{m['nll']:9.4f} {m['coverage@0.90']:8.3f}")

    def ppm(key, field):
        return {pi: evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                             np.concatenate(v[2]), level=0.9)[field]
                for pi, v in per_piece[key].items()}

    boot_rng = np.random.default_rng(11)
    print("\nPaired per-piece diff (strict - non-strict; positive = strictness costs):")
    for field in ("rmse", "nll"):
        da, db = ppm(("feat+LM-ma", True), field), ppm(("feat+LM", True), field)
        common = sorted(set(da) & set(db))
        d = np.array([da[pi] - db[pi] for pi in common])
        m, lo, hi = bootstrap_ci(d, rng=boot_rng)
        sig = "*" if (lo > 0) or (hi < 0) else " "
        print(f"  {field.upper():4s} {m:+8.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")


if __name__ == "__main__":
    main()
