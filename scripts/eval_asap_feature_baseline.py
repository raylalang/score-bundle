#!/usr/bin/env python
"""Does the pretrained LM mean beat a strong hand-built score-feature mean?

The honest vulnerability check on Phase 0: the LM's marginal accuracy value over a
*zero* mean is small, and the existing per-piece ridge baseline is weak (it refits on
each piece's observed notes and extrapolates timing badly).  The fair rival is a rich
score-only feature representation fit under the **identical cross-piece protocol** as
the LM head: fit once on the head pieces, applied out-of-sample to eval pieces, same
ridge head, same l2 selection.  If features match the LM, the claim "the pretrained LM
earns its place as the prior mean" must be reworded.

Grid: means {zero, feat, LM, feat+LM} x graph {off, on}; the published robust protocol
(random 60%-observed masks, seeds 1000+s, EB noise floor 5%).  Paired bootstrap answers:
(1) LM vs feat, alone and under the graph; (2) does the LM add anything *on top of*
features (feat+LM vs feat), and vice versa.  Numpy-only (reads the named array cache):

    python scripts/eval_asap_feature_baseline.py --arrays-cache .cache/asap_arrays_named.pkl
"""
from __future__ import annotations

import argparse

import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.baselines import rich_score_features
from score_bundle.downstream import load_piece_arrays, piece_score
from score_bundle.lm import features as lmfeat
from score_bundle.metrics import evaluate


def bootstrap_ci(vals, B=2000, rng=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cv_l2(reps_by_piece, ys_by_piece, grid, k=5):
    """Grouped k-fold CV over head pieces; returns (best l2, {l2: rmse})."""
    n = len(reps_by_piece)
    folds = [list(range(i, n, k)) for i in range(k)]
    scores = {}
    for l2 in grid:
        errs = []
        for fold in folds:
            tr = [i for i in range(n) if i not in fold]
            H = np.concatenate([reps_by_piece[i] for i in tr])
            Y = np.concatenate([ys_by_piece[i] for i in tr])
            W = lmfeat.fit_prior_mean_head(H, Y, l2=l2)
            for i in fold:
                mu = lmfeat.apply_prior_mean(reps_by_piece[i], W)
                errs.append(float(np.sqrt(np.mean((mu - ys_by_piece[i]) ** 2))))
        scores[l2] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    return best, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--rff-dim", type=int, default=256)
    ap.add_argument("--noise-floor-frac", type=float, default=0.05)
    ap.add_argument("--embeddings", default="emb_leakfree",
                    choices=["emb", "emb_scoreonly", "emb_leakfree"])
    ap.add_argument("--l2-grid", type=float, nargs="+",
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--force-l2", type=float, default=None,
                    help="skip CV l2 selection and use this l2 for every head "
                         "(e.g. 10 = the published protocol; the CV-selected l2=100 "
                         "run exposed the piece-28 EB tau collapse)")
    args = ap.parse_args()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    print(f"{len(head)} head + {len(ev)} eval pieces | embeddings={args.embeddings} "
          f"| rff_dim={args.rff_dim} | noise_floor_frac={args.noise_floor_frac}", flush=True)

    # --- representations (all score-only except the LM embedding) -----------
    def feats(p, rff):
        return rich_score_features(piece_score(p), rff_dim=rff)

    reps = {
        "LM": lambda p: np.asarray(p[args.embeddings], dtype=float),
        "feat-lin": lambda p: feats(p, 0),
        "feat-rff": lambda p: feats(p, args.rff_dim),
    }

    # --- identical head protocol per representation: 5-fold CV l2 on head ---
    ys_head = [np.asarray(p["y"], dtype=float) for p in head]
    l2_grid = [args.force_l2] if args.force_l2 is not None else args.l2_grid
    chosen = {}
    print("\nHead-split 5-fold CV (pooled-RMSE by l2; identical protocol per rep):"
          if args.force_l2 is None else
          f"\nForced l2={args.force_l2:g} for every head (CV skipped):")
    for name, fn in reps.items():
        Hs = [fn(p) for p in head]
        best, scores = cv_l2(Hs, ys_head, l2_grid)
        chosen[name] = best
        row = "  ".join(f"l2={l2:g}: {scores[l2]:.4f}" for l2 in l2_grid)
        print(f"  {name:9s} -> l2={best:g}   ({row})", flush=True)

    # pick the better feature variant by CV to carry into the grid
    def cv_best(name):
        Hs = [reps[name](p) for p in head]
        _, sc = cv_l2(Hs, ys_head, [chosen[name]])
        return sc[chosen[name]]

    feat_name = min(("feat-lin", "feat-rff"), key=cv_best)
    print(f"feature variant for the grid: {feat_name}", flush=True)

    def rep_feat(p):
        return reps[feat_name](p)

    def rep_cat(p):
        return np.concatenate([rep_feat(p), reps["LM"](p)], axis=1)

    grid_reps = {"feat": rep_feat, "LM": reps["LM"], "feat+LM": rep_cat}
    l2_cat, _ = cv_l2([rep_cat(p) for p in head], ys_head, l2_grid)
    grid_l2 = {"feat": chosen[feat_name], "LM": chosen["LM"], "feat+LM": l2_cat}
    print(f"l2 per grid mean: {grid_l2}", flush=True)

    heads = {}
    for name, fn in grid_reps.items():
        H = np.concatenate([fn(p) for p in head])
        heads[name] = lmfeat.fit_prior_mean_head(H, np.concatenate(ys_head),
                                                 l2=grid_l2[name])
    print("heads fit on head split", flush=True)

    # --- eval loop: published robust protocol --------------------------------
    mean_names = ["zero", "feat", "LM", "feat+LM"]
    pool = {}       # (mean, graph) -> [y, pred, std, channel]
    per_piece = {}  # (mean, graph) -> {pi -> [y, pred, std]}
    for s in range(args.seeds):
        seed_rng = np.random.default_rng(1000 + s)
        for pi, p in enumerate(ev):
            score = piece_score(p)
            y = np.asarray(p["y"], dtype=float)
            mask = ie.random_mask(len(y), seed_rng, observed_frac=args.observed_frac)
            means = {"zero": np.zeros_like(y)}
            for name, fn in grid_reps.items():
                means[name] = lmfeat.apply_prior_mean(fn(p), heads[name])
            cells = ie.impute_methods(score, y, means, mask, fit_hyper=True,
                                      rng=seed_rng,
                                      noise_floor_frac=args.noise_floor_frac)
            for key, cell in cells.items():
                buf = pool.setdefault(key, [[], [], [], []])
                buf[0].append(cell.y); buf[1].append(cell.pred); buf[2].append(cell.std)
                buf[3].append(np.asarray(cell.channel))
                pp = per_piece.setdefault(key, {}).setdefault(pi, [[], [], []])
                pp[0].append(cell.y); pp[1].append(cell.pred); pp[2].append(cell.std)
        print(f"seed {s + 1}/{args.seeds} done", flush=True)

    # --- report ---------------------------------------------------------------
    def pooled_metrics(key, ch=None):
        y, pr, sd, c = (np.concatenate(pool[key][i]) for i in range(4))
        if ch is not None:
            sel = c == ch
            y, pr, sd = y[sel], pr[sel], sd[sel]
        return evaluate(y, pr, sd, level=0.9)

    boot_rng = np.random.default_rng(7)
    print(f"\nPooled ({len(ev)} pieces x {args.seeds} seeds, identical masks):")
    print(f"{'mean':9s} {'graph':6s} {'RMSE':>8s} {'NLL':>9s} {'cov@.9':>8s} {'cal-err':>8s}")
    for mn in mean_names:
        for g in (False, True):
            if (mn, g) not in pool:
                continue
            m = pooled_metrics((mn, g))
            print(f"{mn:9s} {('on' if g else 'off'):6s} {m['rmse']:8.4f} "
                  f"{m['nll']:9.4f} {m['coverage@0.90']:8.3f} "
                  f"{m['calibration_error']:8.3f}")
    for ci, cname in enumerate(["tau", "log r", "v"]):
        print(f"\n[{cname}]")
        print(f"{'mean':9s} {'graph':6s} {'RMSE':>8s} {'NLL':>9s} {'cov@.9':>8s}")
        for mn in mean_names:
            for g in (False, True):
                if (mn, g) not in pool:
                    continue
                m = pooled_metrics((mn, g), ch=ci)
                print(f"{mn:9s} {('on' if g else 'off'):6s} {m['rmse']:8.4f} "
                      f"{m['nll']:9.4f} {m['coverage@0.90']:8.3f}")

    def pp_vals(key, field):
        out = {}
        for pi, v in per_piece[key].items():
            m = evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                         np.concatenate(v[2]), level=0.9)
            out[pi] = m[field]
        return out

    def paired(key_a, key_b, field):
        va, vb = pp_vals(key_a, field), pp_vals(key_b, field)
        order = sorted(set(va) & set(vb))
        d = np.array([va[pi] - vb[pi] for pi in order])
        return bootstrap_ci(d, rng=boot_rng)

    contrasts = [
        (("LM", False), ("feat", False), "LM vs feat            (mean only)"),
        (("LM", True), ("feat", True), "LM vs feat            (graph on) "),
        (("feat+LM", True), ("feat", True), "feat+LM vs feat       (graph on) "),
        (("feat+LM", True), ("LM", True), "feat+LM vs LM         (graph on) "),
        (("feat", True), ("zero", True), "feat vs zero          (graph on) "),
    ]
    print("\nPaired bootstrap, per-piece diff (negative = first better):")
    for field in ("rmse", "nll"):
        for a, b, desc in contrasts:
            if a not in per_piece or b not in per_piece:
                continue
            d, lo, hi = paired(a, b, field)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            print(f"  {field.upper():4s} {desc} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
    print("\nReading: if 'LM vs feat' spans 0 and 'feat+LM vs feat' spans 0, the "
          "pretrained LM adds nothing beyond hand-built score features and the claim "
          "must be reworded; if LM wins or adds on top, Phase 0 earns its place.")


if __name__ == "__main__":
    main()
