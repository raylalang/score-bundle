#!/usr/bin/env python
"""Extract named, contamination-filtered per-piece arrays for the downstream evals.

Same extraction as ``eval_asap_robust.py`` (score<->performance matching + LM per-note
embeddings), but the cache **records provenance**: each entry keeps the ASAP performance
path, piece folder and composer, and the blob stores the split seed and whether the
MAESTRO contamination filter was applied.  The old ``robust_arrays.pkl`` stores arrays
only, so its filtering cannot be audited — downstream evals should use this cache.

    python scripts/extract_asap_arrays.py --asap-root ../data/asap-dataset \
        --maestro-root ../data/maestro-v3.0.0 --checkpoint checkpoints/maestro_scaled/best.pt \
        --out .cache/asap_arrays_named.pkl

Needs the train extra (torch + pretty_midi).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asap-root", required=True)
    ap.add_argument("--maestro-root", required=True, help="for the contamination filter")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-head-pieces", type=int, default=40)
    ap.add_argument("--n-eval-pieces", type=int, default=50)
    ap.add_argument("--max-notes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0, help="piece-selection seed (matches prior evals)")
    ap.add_argument("--out", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("Needs PyTorch + pretty_midi:  pip install -e '.[train]'"); sys.exit(1)

    from score_bundle import features
    from score_bundle.lm import data as lmdata
    from score_bundle.lm import features as lmfeat
    from score_bundle.lm.model_torch import build_model
    from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent
    from score_bundle.score import Score

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    tok = MidiTokenizer()
    assert tok.vocab_size == ckpt["cfg"].vocab_size
    print(f"loaded {args.checkpoint} | device {device}", flush=True)

    ann = features.load_asap_annotations(args.asap_root)
    meta = features.load_asap_meta(args.asap_root)
    meta = [r for r in meta if ann.get(r.performance, {}).get("score_and_performance_aligned")]
    train_rel = [r.midi_path.split("maestro-v3.0.0/")[-1]
                 for r in lmdata.load_maestro_meta(args.maestro_root, split="train")]
    before = len(meta)
    meta = features.asap_clean_performances(meta, train_rel)
    print(f"contamination filter: {before} -> {len(meta)} performances", flush=True)

    by_folder = {}
    for r in meta:
        by_folder.setdefault(r.folder, r)
    folders = list(by_folder.values())
    np.random.default_rng(args.seed).shuffle(folders)
    head_recs = folders[: args.n_head_pieces]
    eval_recs = folders[args.n_head_pieces : args.n_head_pieces + args.n_eval_pieces]
    print(f"extracting {len(head_recs)} head + {len(eval_recs)} eval pieces", flush=True)

    def extract(recs, tag):
        out = []
        for rec in recs:
            try:
                score, obs = features.load_asap(rec.performance, args.asap_root, annotations=ann)
                score_m, y = features.asap_performance_variables(score, obs)
                vel = np.asarray(obs["velocity"], dtype=float)[obs["mask"]]
                if args.max_notes and len(score_m) > args.max_notes:
                    keep = slice(0, args.max_notes)
                    score_m = Score(score_m.notes[keep]); y = y[keep]; vel = vel[keep]
                notes = [NoteEvent(int(n.pitch), float(n.onset), float(n.duration),
                                   int(np.clip(vel[i], 1, 127)))
                         for i, n in enumerate(score_m.notes)]
                # (leaky) historical variant: real performed velocity, readout at the
                # VELOCITY token -> leaks v_i into mu_LM. Kept for the A/B baseline.
                emb = lmfeat.note_embeddings_long(model, tok, notes)
                # score-only band-aid: constant placeholder velocity kills the leak by
                # corrupting the input (loses some real velocity signal).
                notes_so = [NoteEvent(n.pitch, n.onset, n.duration, 64) for n in notes]
                emb_scoreonly = lmfeat.note_embeddings_long(model, tok, notes_so)
                # leak-free readout: REAL performed velocity in the input, but the embedding
                # is read at the pre-velocity (DURATION) token, which is causally blind to
                # this note's own velocity. No band-aid, no retraining.
                emb_leakfree = lmfeat.note_embeddings_long(
                    model, tok, notes, readout="pre_velocity"
                )
                if len(emb) != len(y) or len(y) < 8:
                    continue
                out.append({
                    "performance": rec.performance, "folder": rec.folder,
                    "composer": rec.composer, "title": rec.title,
                    "pitch": np.array([n.pitch for n in score_m.notes]),
                    "onset": np.array([n.onset for n in score_m.notes]),
                    "duration": np.array([n.duration for n in score_m.notes]),
                    "voice": np.array([getattr(n, "voice", 0) for n in score_m.notes]),
                    "y": np.asarray(y), "emb": np.asarray(emb),
                    "emb_scoreonly": np.asarray(emb_scoreonly),
                    "emb_leakfree": np.asarray(emb_leakfree),
                    # raw MIDI velocities: needed to rebuild NoteEvents at eval time
                    # (mask-aware embeddings replace held-out notes' velocities per mask)
                    "velocity": np.asarray(vel, dtype=float),
                })
                print(f"  [{tag}] {rec.performance}: {len(y)} notes", flush=True)
            except Exception as exc:
                print(f"  skip {tag} {rec.performance}: {exc}", flush=True)
        return out

    head_arr = extract(head_recs, "head")
    eval_arr = extract(eval_recs, "eval")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blob = {
        "head": head_arr, "eval": eval_arr,
        "meta": {
            "schema_version": 3,  # v2: emb_leakfree; v3: raw per-note MIDI velocities
            "embeddings": ["emb", "emb_scoreonly", "emb_leakfree"],
            "seed": args.seed, "max_notes": args.max_notes,
            "checkpoint": args.checkpoint,
            "contamination_filtered": True,
            "n_meta_before_filter": before, "n_meta_after_filter": len(meta),
        },
    }
    with open(args.out, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"wrote {args.out}: {len(head_arr)} head + {len(eval_arr)} eval pieces", flush=True)


if __name__ == "__main__":
    main()
