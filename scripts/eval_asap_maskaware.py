#!/usr/bin/env python
"""Mask-aware embeddings A/B — is `emb_leakfree` still cheating through neighbours?

`emb_leakfree` reads each note's embedding at the pre-velocity token (causally blind to
the note's *own* velocity), but the LM input still contains the performed velocities of
**all** notes — including notes that the imputation task holds out. A held-out note's
velocity therefore reaches *other* notes' prior means, and information can flow back to
its own prediction through the graph coupling (a joint-target leak, second-order).

The deployment-honest variant is **mask-aware**: the LM input carries real velocities
only for *observed* notes; held-out notes get the placeholder (64); read-out stays at
the pre-velocity token. Embeddings then depend on the mask, so they are extracted per
(piece, seed) with the live model (GPU, cheap).

This script runs, with identical masks per (piece, seed):

    LM-lf     cached emb_leakfree (full velocity context)      — the current default
    LM-ma     mask-aware embeddings, head fit on emb_leakfree  — deployment-realistic
    LM-ma-mh  mask-aware embeddings, head fit on mask-aware head embeddings
              (distribution-matched head; one random mask per head piece)

each with graph off/on, plus paired per-piece bootstrap contrasts. If LM-ma ≈ LM-lf the
joint-target concern is immaterial; if LM-ma is worse, IT is the honest benchmark number.

    python scripts/eval_asap_maskaware.py --checkpoint checkpoints/maestro_scaled/best.pt
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
    ap.add_argument("--noise-floor-frac", type=float, default=0.05)
    ap.add_argument("--placeholder-vel", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("Needs PyTorch:  pip install -e '.[train]'"); sys.exit(1)

    from score_bundle import imputation_eval as ie
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
    assert tok.vocab_size == ckpt["cfg"].vocab_size

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    if meta.get("schema_version", 1) < 3:
        print("cache lacks raw velocities; regenerate with scripts/extract_asap_arrays.py")
        sys.exit(1)
    ev = ev[: args.n_eval_pieces]
    print(f"{len(head)} head + {len(ev)} eval pieces | observed_frac={args.observed_frac} "
          f"| noise_floor_frac={args.noise_floor_frac} | device {device}", flush=True)

    def notes_of(p, vel):
        return [NoteEvent(int(pi), float(oi), float(di), int(np.clip(v, 1, 127)))
                for pi, oi, di, v in zip(p["pitch"], p["onset"], p["duration"], vel)]

    def maskaware_emb(p, mask):
        vel = np.where(mask, p["velocity"], float(args.placeholder_vel))
        return lmfeat.note_embeddings_long(model, tok, notes_of(p, vel),
                                           readout="pre_velocity")

    # --- heads ----------------------------------------------------------------
    H_lf = np.concatenate([p["emb_leakfree"] for p in head])
    Yh = np.concatenate([p["y"] for p in head])
    W_lf = lmfeat.fit_prior_mean_head(H_lf, Yh, l2=args.l2)
    # distribution-matched head: one random mask per head piece, mask-aware inputs
    head_rng = np.random.default_rng(7)
    H_ma = np.concatenate([
        maskaware_emb(p, ie.random_mask(len(p["y"]), head_rng, args.observed_frac))
        for p in head
    ])
    W_ma = lmfeat.fit_prior_mean_head(H_ma, Yh, l2=args.l2)
    print("heads fit (leak-free and mask-aware)", flush=True)

    variants = ["LM-lf", "LM-ma", "LM-ma-mh"]
    channels = ["tau", "log r", "v"]
    acc = ie.MetricAccumulator()
    per_piece = {}  # (mean, graph) -> {pi: [y[],pred[],std[]]}

    for s in range(args.seeds):
        seed_rng = np.random.default_rng(1000 + s)
        for pi, p in enumerate(ev):
            score = piece_score(p)
            y = p["y"]
            mask = ie.random_mask(len(y), seed_rng, observed_frac=args.observed_frac)
            emb_ma = maskaware_emb(p, mask)
            means = {
                "LM-lf": lmfeat.apply_prior_mean(p["emb_leakfree"], W_lf),
                "LM-ma": lmfeat.apply_prior_mean(emb_ma, W_lf),
                "LM-ma-mh": lmfeat.apply_prior_mean(emb_ma, W_ma),
            }
            cells = ie.impute_methods(score, y, means, mask, fit_hyper=True, rng=seed_rng,
                                      noise_floor_frac=args.noise_floor_frac)
            acc.add(cells)
            for key, cell in cells.items():
                pp = per_piece.setdefault(key, {}).setdefault(pi, [[], [], []])
                pp[0].append(cell.y); pp[1].append(cell.pred); pp[2].append(cell.std)
        print(f"seed {s + 1}/{args.seeds} done", flush=True)

    print("\nPooled (identical masks across variants):")
    print(ie.format_report(acc.report(level=0.9)))
    print("\nPer-channel breakdown:")
    print(ie.format_report_by_channel(acc.report_by_channel(channels, level=0.9), channels))

    # --- paired contrasts ------------------------------------------------------
    def per_piece_metric(key, field):
        return {pi: evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                             np.concatenate(v[2]), level=0.9)[field]
                for pi, v in per_piece[key].items()}

    boot_rng = np.random.default_rng(11)
    print("\nPaired bootstrap, per-piece diff (negative = first better):")
    contrasts = [
        (("LM-ma", True), ("LM-lf", True), "mask-aware vs leak-free   (graph on)"),
        (("LM-ma-mh", True), ("LM-lf", True), "mask-aware+mh vs leak-free (graph on)"),
        (("LM-ma", False), ("LM-lf", False), "mask-aware vs leak-free   (mean only)"),
        (("LM-ma", True), ("LM-ma", False), "graph on vs off           (mask-aware)"),
    ]
    for ka, kb, desc in contrasts:
        if ka not in per_piece or kb not in per_piece:
            continue
        for field in ("rmse", "nll"):
            da = per_piece_metric(ka, field); db = per_piece_metric(kb, field)
            common = sorted(set(da) & set(db))
            diff = np.array([da[pi] - db[pi] for pi in common])
            m, lo, hi = bootstrap_ci(diff, rng=boot_rng)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            print(f"  {field.upper():4s} {desc:40s} {m:+8.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")


if __name__ == "__main__":
    main()
