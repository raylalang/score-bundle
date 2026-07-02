#!/usr/bin/env python
"""Step 3 — LM embeddings -> prior mean -> graph posterior, on held-out ASAP.

Extracts per-note LM embeddings h_i on ASAP, fits the head h_i -> mu_LM for
y = [tau, log r, v] on a *train* split of pieces, then runs the held-out imputation
comparison on a disjoint *eval* split:

    mean source  in {zero, ridge-feature, LM}   x   graph residual in {off, on}

reporting recovery (RMSE) and calibration (coverage / PIT-cal-error / NLL).  Eval pieces
are contamination-filtered: any ASAP performance whose MAESTRO twin was in Phase-0
pretraining is dropped (so the LM never saw the eval performance).

    python scripts/eval_asap_calibration.py \
        --asap-root ../data/asap-dataset --maestro-root ../data/maestro-v3.0.0 \
        --checkpoint checkpoints/maestro_scaled/best.pt

Needs the train extra (torch + pretty_midi). The comparison math itself is numpy-only.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asap-root", required=True)
    ap.add_argument("--maestro-root", default=None, help="for the contamination filter")
    ap.add_argument("--checkpoint", required=True, help="MusicGPT best.pt from train_lm.py")
    ap.add_argument("--n-head-pieces", type=int, default=40, help="pieces to fit the mu_LM head")
    ap.add_argument("--n-eval-pieces", type=int, default=20, help="disjoint held-out pieces")
    ap.add_argument("--max-notes", type=int, default=400, help="cap notes/piece (speed)")
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--l2", type=float, default=10.0, help="ridge for the LM head")
    ap.add_argument("--fixed-hyper", action="store_true", help="skip per-piece EB graph fit")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=2.0)
    ap.add_argument("--embeddings", default="scoreonly", choices=["scoreonly", "perf"],
                    help="'scoreonly' (default) tokenizes with a constant placeholder "
                         "velocity so mu_LM cannot read the note's own v target; 'perf' "
                         "reproduces the original (leaky) 2026-06 tables")
    ap.add_argument("--noise-floor-frac", type=float, default=0.05,
                    help="EB noise_var floor (fraction of observed residual variance)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("Needs PyTorch + pretty_midi:  pip install -e '.[train]'")
        sys.exit(1)

    from score_bundle import features, imputation_eval as ie
    from score_bundle.baselines import score_features
    from score_bundle.lm import features as lmfeat
    from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent
    from score_bundle.lm.model_torch import GPTConfig, build_model
    from score_bundle.lm import data as lmdata

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # --- load the trained LM ------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: GPTConfig = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = MidiTokenizer()  # default scheme used in training
    assert tok.vocab_size == cfg.vocab_size, (
        f"tokenizer vocab {tok.vocab_size} != checkpoint vocab {cfg.vocab_size}"
    )
    print(f"loaded {args.checkpoint}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f} | device {device}")

    # --- pick clean, aligned ASAP pieces (piece-disjoint head/eval) ---------
    ann = features.load_asap_annotations(args.asap_root)
    meta = features.load_asap_meta(args.asap_root)
    meta = [r for r in meta if ann.get(r.performance, {}).get("score_and_performance_aligned")]
    if args.maestro_root:
        train_rel = [
            r.midi_path.split("maestro-v3.0.0/")[-1]
            for r in lmdata.load_maestro_meta(args.maestro_root, split="train")
        ]
        before = len(meta)
        meta = features.asap_clean_performances(meta, train_rel)
        print(f"contamination filter: {before} -> {len(meta)} performances (MAESTRO-train twins dropped)")

    # one performance per piece folder, shuffled
    by_folder = {}
    for r in meta:
        by_folder.setdefault(r.folder, r)
    folders = list(by_folder.values())
    rng.shuffle(folders)
    need = args.n_head_pieces + args.n_eval_pieces
    folders = folders[:need]
    head_recs = folders[: args.n_head_pieces]
    eval_recs = folders[args.n_head_pieces : need]
    print(f"pieces: {len(head_recs)} head-fit + {len(eval_recs)} eval (disjoint)")

    def piece_arrays(rec):
        """(score_matched, y[N,3], embeddings[N,d]) for one ASAP performance, capped."""
        score, obs = features.load_asap(rec.performance, args.asap_root, annotations=ann)
        score_m, y = features.asap_performance_variables(score, obs)
        vel = np.asarray(obs["velocity"], dtype=float)[obs["mask"]]
        if args.max_notes and len(score_m) > args.max_notes:
            keep = slice(0, args.max_notes)
            from score_bundle.score import Score
            score_m = Score(score_m.notes[keep]); y = y[keep]; vel = vel[keep]
        notes = [
            NoteEvent(int(n.pitch), float(n.onset), float(n.duration),
                      64 if args.embeddings == "scoreonly" else int(np.clip(vel[i], 1, 127)))
            for i, n in enumerate(score_m.notes)
        ]
        emb = lmfeat.note_embeddings_long(model, tok, notes)
        return score_m, y, emb

    # --- fit the mu_LM head on the head split -------------------------------
    H_list, Y_list = [], []
    for rec in head_recs:
        try:
            _, y, emb = piece_arrays(rec)
        except Exception as exc:
            print(f"  skip head {rec.performance}: {exc}")
            continue
        if len(emb) == len(y) and len(y) > 0:
            H_list.append(emb); Y_list.append(y)
    if not H_list:
        print("no usable head pieces; aborting"); sys.exit(1)
    W = lmfeat.fit_prior_mean_head(np.concatenate(H_list), np.concatenate(Y_list), l2=args.l2)
    print(f"fit mu_LM head on {len(H_list)} pieces, {sum(len(h) for h in H_list)} notes")

    # --- evaluate on held-out pieces ----------------------------------------
    acc = ie.MetricAccumulator()
    n_used = 0
    for rec in eval_recs:
        try:
            score_m, y, emb = piece_arrays(rec)
        except Exception as exc:
            print(f"  skip eval {rec.performance}: {exc}")
            continue
        if len(emb) != len(y) or len(y) < 8:
            continue
        mask = ie.random_mask(len(score_m), rng, observed_frac=args.observed_frac)
        mu_lm = lmfeat.apply_prior_mean(emb, W)
        means = {
            "zero": np.zeros_like(y),
            "ridge": ie.ridge_mean(score_m, y, mask),
            "LM": mu_lm,
        }
        cells = ie.impute_methods(
            score_m, y, means, mask,
            fit_hyper=not args.fixed_hyper, lam=args.lam, eta=args.eta,
            noise_floor_frac=args.noise_floor_frac,
        )
        acc.add(cells)
        n_used += 1
    print(f"evaluated {n_used} held-out pieces\n")

    channels = ["tau", "log r", "v"]
    rep = acc.report(level=0.9)
    print("Held-out ASAP imputation — y = [tau, log r, v], pooled over notes/channels")
    print(ie.format_report(rep))
    print("\nPer-channel breakdown:")
    print(ie.format_report_by_channel(acc.report_by_channel(channels, level=0.9), channels))
    print("\nReading: lower RMSE/NLL/cal-err better; coverage closer to 0.90 better.")
    print("The thesis claim is that 'graph on' improves calibration (and RMSE) over 'graph off'")
    print("at each mean, and that the LM mean beats zero/ridge.")


if __name__ == "__main__":
    main()
