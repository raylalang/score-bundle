#!/usr/bin/env python
"""Kernel comparison on the held-out ASAP imputation task (docs/kernel_comparison_experiment.md).

Compare graph-GP kernels from simplest to experimental, holding EVERYTHING else at
the published strict protocol: leak-free network mean (strict mask-aware embeddings,
head fit on emb_leakfree at l2=10), contamination-filtered cache, 30 pieces x 4 seeds,
40% hidden, identical masks across every kernel row, noise_floor_frac=0.05, EB guard
ON, predictive-variance floor.  Only the precision/kernel construction changes row to
row.  A mu=0 block isolates the kernel effect from the learned mean.

Three stages (so the GPU part runs once and the sweep is numpy-only and parallel):

  precompute  (torch)  replicate the strict mask sequence, compute mask-aware mu_LM
                       per (piece, seed), cache masks + means:
      python scripts/eval_kernels.py --stage precompute

  run         (numpy)  one or more kernel rows; per (piece, seed, channel, mean)
                       guarded spectral EB fit + posterior; results pickle per row:
      python scripts/eval_kernels.py --stage run --kernels additive,matern2

  report      (numpy)  master table + per-channel appendix + paired per-piece
                       bootstrap vs the additive baseline row:
      python scripts/eval_kernels.py --stage report

Rows (see ROWS): independent, chain, additive (reference), matern1/2/3, diffusion,
norm_additive, tonal, harmonic, harmonic_vl.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

INPUTS_DEFAULT = ".cache/kernel_sweep_inputs.pkl"
RESULTS_DIR_DEFAULT = "results/kernels"
CHANNELS = ["tau", "log r", "v"]

# row name -> (adjacency builder name, normalized Laplacian?, spectral kernel)
# Only the graph/kernel construction differs between rows; the additive row is the
# published default and the reference for all paired tests.
ROWS = {
    "independent":  ("combinatorial", False, "independent"),
    "chain":        ("chain",         False, "additive"),
    "additive":     ("combinatorial", False, "additive"),
    "matern1":      ("combinatorial", False, "matern1"),
    "matern2":      ("combinatorial", False, "matern2"),
    "matern3":      ("combinatorial", False, "matern3"),
    "diffusion":    ("combinatorial", False, "diffusion"),
    "norm_additive": ("combinatorial", True,  "additive"),
    "tonal":        ("tonal",         False, "additive"),
    "harmonic":     ("harmonic",      False, "additive"),
    "harmonic_vl":  ("harmonic_vl",   False, "additive"),
}


def build_graph_laplacian(name: str, normalized: bool, score, p: dict) -> np.ndarray:
    from score_bundle.graph import (build_adjacency, build_adjacency_harmonic,
                                    build_adjacency_tonal, chain_adjacency, laplacian)

    if name == "combinatorial":
        W = build_adjacency(score)
    elif name == "chain":
        order = np.argsort(np.asarray(p["onset"]), kind="stable")
        W = chain_adjacency(order=order)
    elif name == "tonal":
        W = build_adjacency_tonal(score)
    elif name == "harmonic":
        W = build_adjacency_harmonic(score, chord_weight=1.0, vl_weight=0.0)
    elif name == "harmonic_vl":
        W = build_adjacency_harmonic(score, chord_weight=1.0, vl_weight=1.0)
    else:
        raise ValueError(f"unknown graph {name!r}")
    return laplacian(W, normalized=normalized)


def bootstrap_ci(vals, B=2000, rng=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- precompute
def stage_precompute(args) -> None:
    try:
        import torch
    except ImportError:
        print("precompute needs PyTorch:  pip install -e '.[train]'"); sys.exit(1)

    from score_bundle import imputation_eval as ie
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.lm import features as lmfeat
    from score_bundle.lm.model_torch import build_model
    from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent

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
    print(f"{len(head)} head + {len(ev)} eval pieces | device {device}", flush=True)

    def notes_of(p, vel):
        return [NoteEvent(int(pi), float(oi), float(di), int(np.clip(v, 1, 127)))
                for pi, oi, di, v in zip(p["pitch"], p["onset"], p["duration"], vel)]

    def maskaware_emb(p, mask):
        vel = np.where(mask, p["velocity"], float(args.placeholder_vel))
        return lmfeat.note_embeddings_long(model, tok, notes_of(p, vel),
                                           readout="pre_velocity")

    # representation feeding the head: LM embeddings alone, or the candidate-headline
    # concat of score-only features (feat-lin, rff_dim=0) + LM embeddings.  Features
    # are score-only hence mask-independent; only the embedding side is mask-aware.
    if args.mean == "feat_lm":
        from score_bundle.baselines import rich_score_features
        from score_bundle.downstream import piece_score

        def feats(p):
            return rich_score_features(piece_score(p), rff_dim=0)

        def head_rep(p):
            return np.concatenate([feats(p), p["emb_leakfree"]], axis=1)

        def eval_rep(p, emb_ma):
            return np.concatenate([feats(p), emb_ma], axis=1)
    else:
        def head_rep(p):
            return p["emb_leakfree"]

        def eval_rep(p, emb_ma):
            return emb_ma

    H = np.concatenate([head_rep(p) for p in head])
    Yh = np.concatenate([p["y"] for p in head])
    W_lf = lmfeat.fit_prior_mean_head(H, Yh, l2=args.l2)
    print(f"head fit (mean={args.mean}, l2={args.l2})", flush=True)

    # EXACTLY the published strict loop (scripts/eval_asap_maskaware.py): one rng per
    # seed, masks drawn per piece in eval order — identical masks to the headline run.
    masks, mus = {}, {}
    for s in range(args.seeds):
        seed_rng = np.random.default_rng(1000 + s)
        for pi, p in enumerate(ev):
            mask = ie.random_mask(len(p["y"]), seed_rng, observed_frac=args.observed_frac)
            emb_ma = maskaware_emb(p, mask)
            masks[(pi, s)] = mask
            mus[(pi, s)] = lmfeat.apply_prior_mean(eval_rep(p, emb_ma), W_lf)
        print(f"seed {s + 1}/{args.seeds} done", flush=True)

    blob = {
        "masks": masks, "mu_lm": mus,
        "meta": {
            "arrays_cache": args.arrays_cache, "checkpoint": args.checkpoint,
            "n_eval_pieces": len(ev), "seeds": args.seeds,
            "observed_frac": args.observed_frac, "l2": args.l2,
            "placeholder_vel": args.placeholder_vel, "embeddings": "mask-aware strict",
            "mean": args.mean,
        },
    }
    os.makedirs(os.path.dirname(args.inputs) or ".", exist_ok=True)
    with open(args.inputs, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"cached strict masks + mu_LM -> {args.inputs}", flush=True)


# --------------------------------------------------------------------------- run
def stage_run(args) -> None:
    from score_bundle import imputation_eval as ie
    from score_bundle.downstream import load_piece_arrays, piece_score
    from score_bundle.model import fit_spectral_field_guarded

    with open(args.inputs, "rb") as fh:
        inputs = pickle.load(fh)
    masks, mus, imeta = inputs["masks"], inputs["mu_lm"], inputs["meta"]
    _, ev, _ = load_piece_arrays(args.arrays_cache)
    ev = ev[: imeta["n_eval_pieces"]]
    seeds = imeta["seeds"]
    os.makedirs(args.out_dir, exist_ok=True)

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    unknown = [k for k in kernels if k not in ROWS]
    if unknown:
        print(f"unknown rows {unknown}; known: {sorted(ROWS)}"); sys.exit(1)

    for row in kernels:
        graph_name, normalized, kernel = ROWS[row]
        print(f"\n=== row {row}: graph={graph_name} normalized={normalized} "
              f"kernel={kernel} | guard ON, noise_floor_frac={args.noise_floor_frac}",
              flush=True)
        # one eigendecomposition per piece, shared across seeds/channels/means
        eigs = []
        for p in ev:
            L = build_graph_laplacian(graph_name, normalized, piece_score(p), p)
            eigs.append(np.linalg.eigh(L))
        print(f"eigendecomposed {len(eigs)} pieces", flush=True)

        cells = {}   # (mean_name, pi, s) -> (y, pred, std, channel)
        guard_counts = {"marglik": 0, "calib": 0, "conservative": 0}
        for s in range(seeds):
            for pi, p in enumerate(ev):
                y = np.asarray(p["y"], dtype=float)
                mask = masks[(pi, s)]
                held = ~mask
                mu_lm = np.asarray(mus[(pi, s)], dtype=float)
                for mean_name, M in (("LM", mu_lm), ("zero", np.zeros_like(y))):
                    yt, pr, sd, ch = [], [], [], []
                    for c in range(y.shape[1]):
                        floor = args.noise_floor_frac * float(
                            np.var((y[:, c] - M[:, c])[mask]))
                        field, hp = fit_spectral_field_guarded(
                            None, y[:, c], kernel=kernel, mask=mask, mean=M[:, c],
                            noise_floor=floor, rng=np.random.default_rng(0),
                            eig=eigs[pi])
                        guard_counts[hp["guard"]] += 1
                        nv = hp["noise_var"]
                        m, std = field.posterior(y[:, c], nv, mask=mask)
                        # predictive-variance floor: held-out y = f + eps
                        pred_std = np.sqrt(std[held] ** 2 + nv)
                        yt.append(y[held, c]); pr.append(m[held]); sd.append(pred_std)
                        ch.append(np.full(int(held.sum()), c, dtype=int))
                    cells[(mean_name, pi, s)] = tuple(
                        np.concatenate(a) for a in (yt, pr, sd, ch))
            print(f"[{row}] seed {s + 1}/{seeds} done "
                  f"(guard: {guard_counts})", flush=True)

        out_path = os.path.join(args.out_dir, f"{row}.pkl")
        with open(out_path, "wb") as fh:
            pickle.dump({"row": row, "spec": ROWS[row], "cells": cells,
                         "guard_counts": guard_counts,
                         "meta": {**imeta, "noise_floor_frac": args.noise_floor_frac}},
                        fh)
        print(f"[{row}] wrote {out_path}", flush=True)

        # quick pooled readout per mean block (full stats in --stage report)
        for mean_name in ("LM", "zero"):
            acc = ie.MetricAccumulator()
            for (mn, pi, s), (yt, pr, sd, ch) in cells.items():
                if mn == mean_name:
                    acc.add({(row, True): ie.CellResult(yt, pr, sd, ch)})
            m = acc.report(level=0.9)[(row, True)]
            print(f"[{row}] mu={mean_name:4s}  RMSE {m['rmse']:.4f}  NLL {m['nll']:.4f}  "
                  f"cov@.9 {m['coverage@0.90']:.3f}  cal-err {m['calibration_error']:.3f}  "
                  f"med-cell {m['rmse_median_cell']:.4f}  worst {m['rmse_worst_cell']:.4f}",
                  flush=True)


# --------------------------------------------------------------------------- report
def stage_report(args) -> None:
    from score_bundle import imputation_eval as ie
    from score_bundle.metrics import evaluate

    rows = {}
    for row in ROWS:
        path = os.path.join(args.out_dir, f"{row}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                rows[row] = pickle.load(fh)
    if "additive" not in rows:
        print("need the additive baseline row for paired tests"); sys.exit(1)
    print(f"loaded rows: {list(rows)}\n")

    def pooled(blob, mean_name):
        acc = ie.MetricAccumulator()
        for (mn, pi, s), (yt, pr, sd, ch) in blob["cells"].items():
            if mn == mean_name:
                acc.add({("k", True): ie.CellResult(yt, pr, sd, ch)})
        return acc

    def per_piece_metric(blob, mean_name, field):
        by_piece = {}
        for (mn, pi, s), (yt, pr, sd, ch) in blob["cells"].items():
            if mn == mean_name:
                b = by_piece.setdefault(pi, [[], [], []])
                b[0].append(yt); b[1].append(pr); b[2].append(sd)
        return {pi: evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                             np.concatenate(v[2]), level=0.9)[field]
                for pi, v in by_piece.items()}

    boot_rng = np.random.default_rng(11)
    for mean_name in ("LM", "zero"):
        print(f"===== mu = {'mu_LM (strict mask-aware)' if mean_name == 'LM' else '0'} =====")
        print(f"{'kernel':14s} {'RMSE':>8s} {'NLL':>9s} {'cov@.9':>8s} {'cal-err':>8s} "
              f"{'med-cell':>9s} {'worst':>8s}  {'dRMSE vs additive [95% CI]':>30s} "
              f"{'dNLL vs additive [95% CI]':>28s}  guard")
        base = {f: per_piece_metric(rows["additive"], mean_name, f)
                for f in ("rmse", "nll")}
        for row in ROWS:
            if row not in rows:
                continue
            blob = rows[row]
            m = pooled(blob, mean_name).report(level=0.9)[("k", True)]
            diff_txt = {}
            for f in ("rmse", "nll"):
                if row == "additive":
                    diff_txt[f] = f"{'—':>28s}"
                    continue
                mine = per_piece_metric(blob, mean_name, f)
                common = sorted(set(mine) & set(base[f]))
                d = np.array([mine[pi] - base[f][pi] for pi in common])
                mu, lo, hi = bootstrap_ci(d, B=args.boot, rng=boot_rng)
                sig = "*" if (lo > 0) or (hi < 0) else " "
                diff_txt[f] = f"{mu:+8.4f} [{lo:+.4f},{hi:+.4f}]{sig}"
            g = blob["guard_counts"]
            gtxt = f"{g['calib']}c/{g['conservative']}x" if (g["calib"] or g["conservative"]) else "clean"
            print(f"{row:14s} {m['rmse']:8.4f} {m['nll']:9.4f} {m['coverage@0.90']:8.3f} "
                  f"{m['calibration_error']:8.3f} {m['rmse_median_cell']:9.4f} "
                  f"{m['rmse_worst_cell']:8.4f}  {diff_txt['rmse']:>30s} "
                  f"{diff_txt['nll']:>28s}  {gtxt}")
        print()

    print("===== per-channel appendix (RMSE / cov@.9) =====")
    for mean_name in ("LM", "zero"):
        print(f"-- mu = {mean_name}")
        hdr = f"{'kernel':14s}" + "".join(f" {c + ' RMSE':>10s} {c + ' cov':>9s}" for c in CHANNELS)
        print(hdr)
        for row in ROWS:
            if row not in rows:
                continue
            rep = pooled(rows[row], mean_name).report_by_channel(CHANNELS, level=0.9)
            line = f"{row:14s}"
            for c in CHANNELS:
                m = rep.get(("k", True, c))
                line += (f" {m['rmse']:10.4f} {m['coverage@0.90']:9.3f}" if m
                         else f" {'—':>10s} {'—':>9s}")
            print(line)
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["precompute", "run", "report"])
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--inputs", default=INPUTS_DEFAULT)
    ap.add_argument("--out-dir", default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--checkpoint", default="checkpoints/maestro_scaled/best.pt")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--placeholder-vel", type=int, default=64)
    ap.add_argument("--noise-floor-frac", type=float, default=0.05)
    ap.add_argument("--mean", default="lm", choices=["lm", "feat_lm"],
                    help="precompute: representation feeding the strict prior mean — "
                         "'lm' (published default) or 'feat_lm' (candidate-headline "
                         "concat of feat-lin score features + LM embeddings)")
    ap.add_argument("--kernels", default=",".join(ROWS))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    {"precompute": stage_precompute, "run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
