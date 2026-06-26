#!/usr/bin/env python
"""Step 3, robust: bootstrap CIs + calibration figures for the held-out ASAP comparison.

Strengthens ``scripts/eval_asap_calibration.py`` against the obvious critique (n=20, one mask
seed, no error bars).  The expensive part -- the score<->performance matching and the LM
embedding extraction -- is **seed-independent**, so we extract per-piece arrays ONCE (cached,
restartable) and then resample held-out masks over many seeds cheaply.

Outputs:
  * a pooled table (RMSE / NLL / coverage / cal-err) over all pieces x seeds x channels, for
    means {zero, ridge, LM} x graph {off, on(marglik), on(calib)};
  * **bootstrap 95% CIs over pieces** for each cell's RMSE/NLL, plus a **paired** bootstrap CI
    on the gap (LM+graph) - (LM mean only) and (LM+graph) - (ridge mean only) -- so "the graph
    helps" and "LM beats ridge" come with significance, not just point estimates;
  * a **reliability diagram** (empirical vs nominal coverage) and a **PIT histogram** PNG.

    python scripts/eval_asap_robust.py --asap-root ../data/asap-dataset \
        --maestro-root ../data/maestro-v3.0.0 --checkpoint checkpoints/maestro_scaled/best.pt \
        --n-eval-pieces 50 --n-head-pieces 40 --seeds 15 --out-dir figures

Needs the train extra (torch + pretty_midi); the comparison math is numpy-only.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


# --------------------------------------------------------------------------- arrays
def extract_arrays(args, model, tok, lmfeat, features, NoteEvent, ann, recs, tag):
    """[(pitch,onset,dur,voice, y[N,3], emb[N,d]) ...] for a list of ASAP records."""
    from score_bundle.score import Score

    out = []
    for rec in recs:
        try:
            score, obs = features.load_asap(rec.performance, args.asap_root, annotations=ann)
            score_m, y = features.asap_performance_variables(score, obs)
            vel = np.asarray(obs["velocity"], dtype=float)[obs["mask"]]
            if args.max_notes and len(score_m) > args.max_notes:
                keep = slice(0, args.max_notes)
                score_m = Score(score_m.notes[keep]); y = y[keep]; vel = vel[keep]
            notes = [
                NoteEvent(int(n.pitch), float(n.onset), float(n.duration),
                          int(np.clip(vel[i], 1, 127)))
                for i, n in enumerate(score_m.notes)
            ]
            emb = lmfeat.note_embeddings_long(model, tok, notes)
            if len(emb) != len(y) or len(y) < 8:
                continue
            out.append({
                "pitch": np.array([n.pitch for n in score_m.notes]),
                "onset": np.array([n.onset for n in score_m.notes]),
                "duration": np.array([n.duration for n in score_m.notes]),
                "voice": np.array([getattr(n, "voice", 0) for n in score_m.notes]),
                "y": np.asarray(y), "emb": np.asarray(emb),
            })
            print(f"  [{tag}] {rec.performance.split('/')[-1]}: {len(y)} notes", flush=True)
        except Exception as exc:
            print(f"  skip {tag} {rec.performance}: {exc}", flush=True)
    return out


def piece_score(p):
    from score_bundle.score import Score
    return Score.from_arrays(p["pitch"], p["onset"], p["duration"], p["voice"])


# --------------------------------------------------------------------------- metrics
def cell_metrics(y, pred, std, level=0.9):
    from score_bundle.metrics import evaluate as _eval
    return _eval(np.asarray(y), np.asarray(pred), np.asarray(std), level=level)


def bootstrap_ci(per_piece_vals, B=2000, rng=None, lo=2.5, hi=97.5):
    """Percentile bootstrap CI of the mean of a per-piece statistic."""
    v = np.asarray([x for x in per_piece_vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asap-root", required=True)
    ap.add_argument("--maestro-root", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-head-pieces", type=int, default=40)
    ap.add_argument("--n-eval-pieces", type=int, default=50)
    ap.add_argument("--max-notes", type=int, default=400)
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, default=15, help="number of held-out mask seeds")
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-calib", action="store_true",
                    help="skip the (expensive) calibration-split graph variant in the bootstrap")
    ap.add_argument("--seed", type=int, default=0, help="base seed for piece selection")
    ap.add_argument("--arrays-cache", default=None, help="pickle cache of per-piece arrays")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("Needs PyTorch + pretty_midi:  pip install -e '.[train]'"); sys.exit(1)

    from score_bundle import features, imputation_eval as ie
    from score_bundle.lm import features as lmfeat
    from score_bundle.lm.tokenizer import MidiTokenizer, NoteEvent
    from score_bundle.lm.model_torch import GPTConfig, build_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    tok = MidiTokenizer()
    assert tok.vocab_size == cfg.vocab_size
    print(f"loaded {args.checkpoint}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"params, val_loss {ckpt.get('val_loss', float('nan')):.4f} | device {device}", flush=True)

    cache = args.arrays_cache
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            blob = pickle.load(fh)
        head_arr, eval_arr = blob["head"], blob["eval"]
        # the cache may hold more eval pieces than requested; honour --n-eval-pieces
        eval_arr = eval_arr[: args.n_eval_pieces]
        print(f"loaded cached arrays: {len(head_arr)} head + {len(eval_arr)} eval "
              f"(of {len(blob['eval'])} cached)", flush=True)
    else:
        from score_bundle.lm import data as lmdata
        ann = features.load_asap_annotations(args.asap_root)
        meta = features.load_asap_meta(args.asap_root)
        meta = [r for r in meta if ann.get(r.performance, {}).get("score_and_performance_aligned")]
        if args.maestro_root:
            train_rel = [r.midi_path.split("maestro-v3.0.0/")[-1]
                         for r in lmdata.load_maestro_meta(args.maestro_root, split="train")]
            before = len(meta); meta = features.asap_clean_performances(meta, train_rel)
            print(f"contamination filter: {before} -> {len(meta)} performances", flush=True)
        by_folder = {}
        for r in meta:
            by_folder.setdefault(r.folder, r)
        folders = list(by_folder.values())
        np.random.default_rng(args.seed).shuffle(folders)
        need = args.n_head_pieces + args.n_eval_pieces
        head_recs = folders[: args.n_head_pieces]
        eval_recs = folders[args.n_head_pieces: need]
        print(f"extracting arrays: {len(head_recs)} head + {len(eval_recs)} eval", flush=True)
        head_arr = extract_arrays(args, model, tok, lmfeat, features, NoteEvent, ann, head_recs, "head")
        eval_arr = extract_arrays(args, model, tok, lmfeat, features, NoteEvent, ann, eval_recs, "eval")
        if cache:
            with open(cache, "wb") as fh:
                pickle.dump({"head": head_arr, "eval": eval_arr}, fh)
            print(f"cached arrays -> {cache}", flush=True)

    # --- fit the mu_LM head on head pieces ----------------------------------
    H = np.concatenate([p["emb"] for p in head_arr]); Yh = np.concatenate([p["y"] for p in head_arr])
    W = lmfeat.fit_prior_mean_head(H, Yh, l2=args.l2)
    print(f"fit mu_LM head on {len(head_arr)} pieces, {len(H)} notes", flush=True)

    # --- cells: means x graph variants --------------------------------------
    variants = [(False, False, "marglik"), ("graph", True, "marglik")]
    if not args.no_calib:
        variants.append(("graph+calib", True, "calib"))
    mean_names = ["zero", "ridge", "LM"]
    # global pools (pooled metrics + reliability + PIT); per-piece pools (bootstrap)
    pool = {}      # (mean,label) -> [y[], pred[], std[]]
    per_piece = {} # (mean,label) -> {piece_idx -> [y[],pred[],std[]]}
    for mn in mean_names:
        for lab, _, _ in variants:
            pool[(mn, lab)] = [[], [], []]
            per_piece[(mn, lab)] = {}

    for s in range(args.seeds):
        seed_rng = np.random.default_rng(1000 + s)
        for pi, p in enumerate(eval_arr):
            score = piece_score(p); y = p["y"]
            mask = ie.random_mask(len(y), seed_rng, observed_frac=args.observed_frac)
            mu_lm = lmfeat.apply_prior_mean(p["emb"], W)
            means = {"zero": np.zeros_like(y), "ridge": ie.ridge_mean(score, y, mask), "LM": mu_lm}
            cells = ie.impute_methods(score, y, means, mask, fit_hyper=True,
                                      graph_variants=variants, rng=seed_rng)
            for key, cell in cells.items():
                pool[key][0].append(cell.y); pool[key][1].append(cell.pred); pool[key][2].append(cell.std)
                pp = per_piece[key].setdefault(pi, [[], [], []])
                pp[0].append(cell.y); pp[1].append(cell.pred); pp[2].append(cell.std)
        print(f"seed {s+1}/{args.seeds} done", flush=True)

    # --- pooled table + bootstrap CIs ---------------------------------------
    boot_rng = np.random.default_rng(7)
    print("\nHeld-out ASAP imputation (pooled over pieces x seeds x channels)")
    print(f"{'mean':6s} {'graph':12s} {'RMSE [95% CI]':>26s} {'NLL [95% CI]':>26s} {'cov@.9':>8s} {'cal':>7s}")
    rows = {}
    for mn in mean_names:
        for lab, _, _ in variants:
            key = (mn, lab)
            y = np.concatenate(pool[key][0]); pr = np.concatenate(pool[key][1]); sd = np.concatenate(pool[key][2])
            m = cell_metrics(y, pr, sd)
            pp_rmse = [cell_metrics(np.concatenate(v[0]), np.concatenate(v[1]), np.concatenate(v[2]))["rmse"]
                       for v in per_piece[key].values()]
            pp_nll = [cell_metrics(np.concatenate(v[0]), np.concatenate(v[1]), np.concatenate(v[2]))["nll"]
                      for v in per_piece[key].values()]
            r_mean, r_lo, r_hi = bootstrap_ci(pp_rmse, B=args.boot, rng=boot_rng)
            n_mean, n_lo, n_hi = bootstrap_ci(pp_nll, B=args.boot, rng=boot_rng)
            rows[key] = {"metrics": m, "pp_rmse": pp_rmse, "pp_nll": pp_nll,
                         "rmse_ci": (r_lo, r_hi), "nll_ci": (n_lo, n_hi)}
            glab = "off" if lab is False else lab
            print(f"{mn:6s} {glab:12s} "
                  f"{m['rmse']:7.4f} [{r_lo:.3f},{r_hi:.3f}]   "
                  f"{m['nll']:7.3f} [{n_lo:.2f},{n_hi:.2f}]   "
                  f"{m['coverage@0.90']:7.3f} {m['calibration_error']:6.3f}")

    # --- paired bootstrap: does the graph help? does LM beat ridge? ---------
    def paired_ci(key_a, key_b, field):
        pa = rows[key_a][field]; pb = rows[key_b][field]
        common = set(per_piece[key_a]) & set(per_piece[key_b])
        order = sorted(common)
        # rebuild per-piece in matching order
        def vals(key):
            d = {pi: cell_metrics(np.concatenate(v[0]), np.concatenate(v[1]), np.concatenate(v[2]))
                 ["rmse" if field == "pp_rmse" else "nll"] for pi, v in per_piece[key].items()}
            return np.array([d[pi] for pi in order])
        diff = vals(key_a) - vals(key_b)
        return bootstrap_ci(diff, B=args.boot, rng=boot_rng)

    print("\nPaired bootstrap (per-piece diff, 95% CI; negative = first is better):")
    contrasts = [
        (("LM", "graph"), ("LM", False), "LM+graph  vs  LM mean-only"),
        (("LM", "graph+calib"), ("LM", "graph"), "LM+graph+calib  vs  LM+graph"),
        (("LM", "graph"), ("ridge", False), "LM+graph  vs  ridge mean-only"),
        (("LM", "graph"), ("zero", "graph"), "LM+graph  vs  zero+graph"),
    ]
    contrasts = [c for c in contrasts if c[0] in per_piece and c[1] in per_piece]
    for field, name in [("pp_rmse", "RMSE"), ("pp_nll", "NLL")]:
        for a, b, desc in contrasts:
            d, lo, hi = paired_ci(a, b, field)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            print(f"  {name:4s}  {desc:34s} {d:+7.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}")

    # --- figures ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from score_bundle.metrics import coverage as _coverage, pit_values as _pit

    levels = np.linspace(0.1, 0.95, 18)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    show = [("zero", False, "zero (no graph)"), ("LM", False, "LM (no graph)"),
            ("LM", "graph", "LM + graph"), ("LM", "graph+calib", "LM + graph + calib")]
    show = [s for s in show if (s[0], s[1]) in pool]
    for mn, lab, name in show:
        y = np.concatenate(pool[(mn, lab)][0]); pr = np.concatenate(pool[(mn, lab)][1]); sd = np.concatenate(pool[(mn, lab)][2])
        emp = [_coverage(y, pr, sd, level=L) for L in levels]
        ax.plot(levels, emp, "o-", ms=3, label=name)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability diagram (held-out ASAP)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "reliability_diagram.png"), dpi=150)
    print(f"\nwrote {os.path.join(args.out_dir, 'reliability_diagram.png')}", flush=True)

    pit_cells = [("LM", "graph", "LM + graph"), ("LM", "graph+calib", "LM + graph + calib")]
    pit_cells = [c for c in pit_cells if (c[0], c[1]) in pool]
    fig, axes = plt.subplots(1, len(pit_cells), figsize=(4.5 * len(pit_cells), 3.6), squeeze=False)
    axes = axes[0]
    for ax, (mn, lab, name) in zip(axes, pit_cells):
        y = np.concatenate(pool[(mn, lab)][0]); pr = np.concatenate(pool[(mn, lab)][1]); sd = np.concatenate(pool[(mn, lab)][2])
        u = _pit(y, pr, sd)
        ax.hist(u, bins=20, range=(0, 1), color="#4477aa", edgecolor="white")
        ax.axhline(len(u) / 20, color="k", ls="--", lw=1)
        ax.set_title(f"PIT — {name}"); ax.set_xlabel("PIT value")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "pit_histogram.png"), dpi=150)
    print(f"wrote {os.path.join(args.out_dir, 'pit_histogram.png')}", flush=True)

    print(f"\nDone: {len(eval_arr)} eval pieces x {args.seeds} seeds.")


if __name__ == "__main__":
    main()
