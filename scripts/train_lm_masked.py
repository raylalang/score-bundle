#!/usr/bin/env python
"""Stage 2 — pretrain the masked, score-conditioned music LM on MAESTRO.

Same tokenizer and data pipeline as ``scripts/train_lm.py`` (cached streams are
shared), but the objective matches the Phase-1 read-out: a **bidirectional**
transformer trained to predict each note's velocity bin from the score-like tokens of
all notes plus the *observed* notes' velocities, with a per-window observed fraction
``rho ~ U(frac-lo, frac-hi)``.  Checkpoints on masked CE at the fixed 60%-observed
validation rate (the published Phase-1 protocol).  Defaults mirror the Stage-1
``maestro_scaled`` checkpoint's geometry (d=512, L=6, h=8, block 512, 15 x 500 steps)
for a budget-matched A/B.

    python scripts/train_lm_masked.py --maestro-root ../data/maestro-v3.0.0 \
        --cache-dir .cache/lm --out-dir checkpoints/maestro_masked --device cuda:0

Needs the train extra (torch + pretty_midi + tqdm): pip install -e '.[train]'.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maestro-root", required=True, help="dir containing maestro-v3.0.0.json")
    # tokenizer (must match the Stage-1 runs for a fair A/B)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--max-shift-steps", type=int, default=96)
    ap.add_argument("--max-dur-steps", type=int, default=96)
    ap.add_argument("--n-vel-bins", type=int, default=32)
    # model (defaults = maestro_scaled geometry)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    # objective
    ap.add_argument("--frac-lo", type=float, default=0.1,
                    help="lower bound of the per-window observed fraction")
    ap.add_argument("--frac-hi", type=float, default=0.9)
    # data / optimization (defaults = Stage-1 budget)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--max-notes", type=int, default=None)
    # io
    ap.add_argument("--out-dir", default="checkpoints/maestro_masked")
    ap.add_argument("--cache-dir", default=None, help="cache tokenized streams here")
    ap.add_argument("--device", default=None, help="cuda[:i] | cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("This script needs PyTorch. Install with:  pip install -e '.[train]'")
        sys.exit(1)

    from score_bundle.lm.tokenizer import MidiTokenizer
    from score_bundle.lm.model_torch import GPTConfig, build_model
    from score_bundle.lm import data as lmdata
    from score_bundle.lm import masked as mk
    from score_bundle.lm import train as lmtrain

    if not os.path.exists(os.path.join(args.maestro_root, lmdata.MAESTRO_JSON)):
        print(f"No {lmdata.MAESTRO_JSON} under {args.maestro_root!r}.")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tok = MidiTokenizer(
        grid=args.grid,
        max_shift_steps=args.max_shift_steps,
        max_dur_steps=args.max_dur_steps,
        n_vel_bins=args.n_vel_bins,
    )
    print(f"device={device}  tokenizer vocab={tok.vocab_size} "
          f"(+1 [MASK] -> model vocab {mk.masked_vocab_size(tok)})", flush=True)

    cfg = lmtrain.TrainConfig(
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        max_notes=args.max_notes,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    train_stream = lmtrain.build_stream(
        args.maestro_root, tok, "train", cfg.train_limit, cfg.max_notes, cfg.cache_dir
    )
    val_stream = lmtrain.build_stream(
        args.maestro_root, tok, "validation", cfg.val_limit, cfg.max_notes, cfg.cache_dir
    )

    gpt_cfg = GPTConfig(
        vocab_size=mk.masked_vocab_size(tok),
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        dropout=args.dropout,
        causal=False,
    )
    model = build_model(gpt_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"masked MusicGPT (bidirectional): {n_params/1e6:.2f}M params "
          f"({args.n_layer}L/{args.d_model}d/{args.n_head}h) | "
          f"rho ~ U({args.frac_lo}, {args.frac_hi})", flush=True)

    hist = mk.train_masked(model, train_stream, val_stream, cfg, tok,
                           frac_range=(args.frac_lo, args.frac_hi))

    if hist.val_loss:
        print(f"\nval masked CE@0.6: {hist.val_loss[0]:.4f} -> {hist.val_loss[-1]:.4f} "
              f"(best {hist.best_val:.4f} at {hist.best_path}; "
              f"chance = log n_vel_bins = {float(__import__('math').log(args.n_vel_bins)):.3f})")


if __name__ == "__main__":
    main()
