#!/usr/bin/env python
"""Pretrain the Phase-0 music LM on MAESTRO.

Tokenizes a MAESTRO release, trains the from-scratch :class:`MusicGPT` with next-token
cross-entropy, logs train/val loss + perplexity (tqdm), checkpoints the best-val model,
and samples a continuation each epoch.  Single-GPU (uses CUDA when available).

    python scripts/train_lm.py --maestro-root /path/to/maestro-v3.0.0 \
        --d-model 256 --n-layer 4 --epochs 10 --cache-dir .cache/lm

Needs the train extra (torch + pretty_midi + tqdm): pip install -e '.[train]'.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maestro-root", required=True, help="dir containing maestro-v3.0.0.json")
    # tokenizer
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--max-shift-steps", type=int, default=96)
    ap.add_argument("--max-dur-steps", type=int, default=96)
    ap.add_argument("--n-vel-bins", type=int, default=32)
    # model
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    # data / optimization
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-limit", type=int, default=None, help="cap #train pieces")
    ap.add_argument("--val-limit", type=int, default=None, help="cap #val pieces")
    ap.add_argument("--max-notes", type=int, default=None, help="cap notes per piece")
    # io
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--cache-dir", default=None, help="cache tokenized streams here")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
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
    print(f"device={device}  vocab_size={tok.vocab_size}")

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
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        dropout=args.dropout,
    )
    model = build_model(gpt_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MusicGPT: {n_params/1e6:.2f}M params  ({args.n_layer}L/{args.d_model}d/{args.n_head}h)")

    # a real prompt for per-epoch sampling: first notes of the first val piece
    val_recs = lmdata.load_maestro_meta(args.maestro_root, split="validation")
    prompt_notes = lmdata.maestro_note_events(val_recs[0].midi_path, max_notes=16)

    hist = lmtrain.train(model, train_stream, val_stream, cfg, tok=tok, sample_prompt=prompt_notes)

    if hist.val_ppl:
        print(f"\nval perplexity: {hist.val_ppl[0]:.2f} -> {hist.val_ppl[-1]:.2f} "
              f"(best val_loss {hist.best_val:.4f} at {hist.best_path})")


if __name__ == "__main__":
    main()
