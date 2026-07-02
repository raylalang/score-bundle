#!/usr/bin/env python
"""Stage-2 A/B — does the masked, score-conditioned LM beat the Stage-1 read-out?

Stage 1 reads a causal next-token model at the pre-velocity token (leak-free, but the
state was never explicitly trained to expose expression, and sees only left context).
Stage 2 (`scripts/train_lm_masked.py`) trains a bidirectional model on exactly the
inference task: predict each hidden note's velocity bin from all notes' score tokens
plus observed velocities. Its read-out is **mask-aware by construction** — a hidden
note's velocity token is simply absent from the input — so this A/B compares against
BOTH published Stage-1 numbers: the leak-free cell (its non-strict headline) and, in
spirit, the mask-aware strict protocol (which Stage 2 satisfies automatically).

Variants (identical masks per piece x seed; graph off/on each):

    LM-s1   cached Stage-1 emb_leakfree + its ridge head       — published reference
    LM-s2   Stage-2 embeddings at the [MASK]ed velocity token + ridge head fit on
            Stage-2 head-piece embeddings (one random mask per head piece)
    LM-s2d  LM-s2 for tau / log r; the model's OWN velocity-bin expectation for v
            (no ridge head on the v channel — the direct amortized prediction)

    python scripts/eval_asap_stage2.py --checkpoint checkpoints/maestro_masked/best.pt
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
    ap.add_argument("--checkpoint", default="checkpoints/maestro_masked/best.pt")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--noise-floor-frac", type=float, default=0.05)
    ap.add_argument("--tag", default="LM-s2", help="row label for the Stage-2 variant")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("Needs PyTorch:  pip install -e '.[train]'"); sys.exit(1)

    from score_bundle import imputation_eval as ie
    from score_bundle.downstream import load_piece_arrays, piece_score
    from score_bundle.lm import features as lmfeat
    from score_bundle.lm import masked as mk
    from score_bundle.lm.model_torch import build_model
    from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent
    from score_bundle.metrics import evaluate

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("objective") != "masked_velocity":
        print(f"{args.checkpoint} is not a Stage-2 masked checkpoint"); sys.exit(1)
    model = build_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    tok = MidiTokenizer()
    assert mk.masked_vocab_size(tok) == ckpt["cfg"].vocab_size

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    if meta.get("schema_version", 1) < 3:
        print("cache lacks raw velocities; regenerate with scripts/extract_asap_arrays.py")
        sys.exit(1)
    ev = ev[: args.n_eval_pieces]
    print(f"{len(head)} head + {len(ev)} eval pieces | observed_frac={args.observed_frac} "
          f"| noise_floor_frac={args.noise_floor_frac} | device {device} | "
          f"ckpt val_mce {ckpt.get('val_loss', float('nan')):.4f}", flush=True)

    def notes_of(p):
        return [NoteEvent(int(pi), float(oi), float(di), int(np.clip(v, 1, 127)))
                for pi, oi, di, v in zip(p["pitch"], p["onset"], p["duration"],
                                         p["velocity"])]

    def s2_emb(p, mask, want_logits=False):
        return mk.masked_note_embeddings_long(model, tok, notes_of(p), mask,
                                              return_vel_logits=want_logits)

    # --- heads ----------------------------------------------------------------
    Yh = np.concatenate([p["y"] for p in head])
    W_s1 = lmfeat.fit_prior_mean_head(
        np.concatenate([p["emb_leakfree"] for p in head]), Yh, l2=args.l2)
    head_rng = np.random.default_rng(7)
    H_s2 = np.concatenate([
        s2_emb(p, ie.random_mask(len(p["y"]), head_rng, args.observed_frac))
        for p in head
    ])
    W_s2 = lmfeat.fit_prior_mean_head(H_s2, Yh, l2=args.l2)
    print("heads fit (Stage-1 leak-free and Stage-2 masked)", flush=True)

    tag, tag_d = args.tag, args.tag + "d"
    channels = ["tau", "log r", "v"]
    acc = ie.MetricAccumulator()
    per_piece = {}

    for s in range(args.seeds):
        seed_rng = np.random.default_rng(1000 + s)
        for pi, p in enumerate(ev):
            score = piece_score(p)
            y = p["y"]
            mask = ie.random_mask(len(y), seed_rng, observed_frac=args.observed_frac)
            emb2, vlog = s2_emb(p, mask, want_logits=True)
            mu_s2 = lmfeat.apply_prior_mean(emb2, W_s2)
            mu_s2d = mu_s2.copy()
            mu_s2d[:, 2] = mk.direct_velocity_mean(tok, vlog, p["velocity"], mask)
            means = {
                "LM-s1": lmfeat.apply_prior_mean(p["emb_leakfree"], W_s1),
                tag: mu_s2,
                tag_d: mu_s2d,
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

    def per_piece_metric(key, field):
        return {pi: evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                             np.concatenate(v[2]), level=0.9)[field]
                for pi, v in per_piece[key].items()}

    boot_rng = np.random.default_rng(11)
    print("\nPaired bootstrap, per-piece diff (negative = first better):")
    contrasts = [
        ((tag, True), ("LM-s1", True), f"{tag} vs Stage-1        (graph on)"),
        ((tag, False), ("LM-s1", False), f"{tag} vs Stage-1        (mean only)"),
        ((tag_d, True), ((tag, True)), f"{tag_d} vs {tag}       (graph on)"),
        ((tag, True), ((tag, False)), f"graph on vs off        ({tag})"),
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
            print(f"  {field.upper():4s} {desc:38s} {m:+8.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
    print("\nNote: LM-s2 rows are mask-aware by construction (the strict protocol); the "
          "LM-s1 row is the published leak-free (non-strict) reference — its strict "
          "counterpart is ~+0.013 NLL worse (see docs/phase1_calibration_results.md).")


if __name__ == "__main__":
    main()
