#!/usr/bin/env python
"""Downstream tasks re-validated under the GP-first model (graph-gp-first branch).

The six downstream demonstrations were originally run under the two-stage pipeline;
their verdicts do not automatically transfer to the GP-first model.  This script
re-runs each task's decisive contrasts with IDENTICAL rng for the old rows and the
GP-first rows, so old-vs-new is paired:

    anomaly     old {zero, LM} x graph on/off (scripts/eval_asap_anomaly.py recipe)
                + GP-feat / GP-featlm: joint MOGP fit on the corrupted piece,
                joint leave-one-out predictive NLL as the suspicion score
    denoise     old {zero, LM} x {graph, graph-oracle} (known-noise setting)
                + GP-feat blind / oracle (noise pinned at the true level)
    completion  prefix/block masks x fracs; old LM x graph on/off
                + GP-feat / GP-featlm on the same masks
    selective   pure analysis: risk-coverage from the v2 imputation cells
                (results/graphgp_v2/b_featlm*) vs the adopted-headline cells
    era         style summaries from raw y vs old-graph-denoised vs GP-denoised

Embeddings: the cached NON-strict `emb_leakfree` for both old and GP rows (the
condition the original downstream runs used — apples to apples; noted caveat).

    python scripts/eval_downstream_gpfirst.py --task anomaly --shard 0/4
    python scripts/eval_downstream_gpfirst.py --task report
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

OUT_DIR = "results/downstream_gpfirst"
CHANNELS = ["tau", "log r", "v"]


def bootstrap_ci(vals, B=2000, rng=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def zscore_cols(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / np.maximum(sd, 1e-9)


def gp_for(p, variant):
    """(gp, feats meta) for one piece under a GP-first variant (full-context embs)."""
    from score_bundle.baselines import rich_score_features
    from score_bundle.downstream import piece_score
    from score_bundle.gp import MultiOutputGraphGP
    from score_bundle.graph import build_adjacency, laplacian

    score = piece_score(p)
    X = zscore_cols(rich_score_features(score, rff_dim=0))
    feats = [np.concatenate([X, np.ones((len(X), 1))], axis=1)]
    if variant == "GP-featlm":
        feats.append(zscore_cols(np.asarray(p["emb_leakfree"], dtype=float)))
    nu, U = np.linalg.eigh(laplacian(build_adjacency(score)))
    return MultiOutputGraphGP(nu, U, kernel="additive", features=feats)


def shard_pieces(ev, shard):
    k, n = map(int, shard.split("/"))
    return [(pi, p) for pi, p in enumerate(ev) if pi % n == k]


# --------------------------------------------------------------------------- anomaly
def task_anomaly(args, head, ev):
    from score_bundle.downstream import (anomaly_scores, auroc, average_precision,
                                         inject_anomalies, piece_score)
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.lm import features as lmfeat

    W = lmfeat.fit_prior_mean_head(
        np.concatenate([p["emb_leakfree"] for p in head]),
        np.concatenate([p["y"] for p in head]), l2=10.0)
    old_methods = [("zero", False), ("zero", True), ("LM", False), ("LM", True)]
    gp_variants = ["GP-feat", "GP-featlm"]
    rows = {}
    for s in range(args.seeds):
        rng = np.random.default_rng(100 + s)
        for pi, p in shard_pieces(ev, args.shard):
            score = piece_score(p)
            L = laplacian(build_adjacency(score))
            mu_lm = lmfeat.apply_prior_mean(p["emb_leakfree"], W)
            Y = np.asarray(p["y"], dtype=float)
            # identical corruption stream to the published recipe: per channel, in order
            Y_bad = Y.copy()
            labels = {}
            for ci in range(3):
                Y_bad[:, ci], labels[ci] = inject_anomalies(
                    Y[:, ci], rng, frac=args.anomaly_frac, scale=args.scale)
            for ci, cname in enumerate(CHANNELS):
                for mname, use_graph in old_methods:
                    mean = np.zeros(len(Y)) if mname == "zero" else mu_lm[:, ci]
                    sc = anomaly_scores(L, Y_bad[:, ci], mean, use_graph=use_graph)
                    key = (cname, f"{mname}{'+graph' if use_graph else ''}")
                    rows.setdefault(key, []).append(
                        (auroc(labels[ci], sc), average_precision(labels[ci], sc)))
            for variant in gp_variants:
                gp = gp_for(p, variant)
                floor = 0.05 * np.array([float(np.var(Y_bad[:, c])) for c in range(3)])
                x_hat, _ = gp.fit(Y_bad, np.ones(len(Y), dtype=bool),
                                  noise_floor=floor, maxiter=args.maxiter)
                loo_m, loo_v = gp.loo_predictive(Y_bad, x_hat)
                nll = 0.5 * (np.log(2 * np.pi * loo_v)
                             + (Y_bad - loo_m) ** 2 / loo_v)
                for ci, cname in enumerate(CHANNELS):
                    rows.setdefault((cname, variant), []).append(
                        (auroc(labels[ci], nll[:, ci]),
                         average_precision(labels[ci], nll[:, ci])))
            print(f"[anomaly] seed {s} piece {pi} done", flush=True)
    return rows


# --------------------------------------------------------------------------- denoise
def task_denoise(args, head, ev):
    from score_bundle.downstream import denoise_channel, piece_score
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.lm import features as lmfeat
    from score_bundle.metrics import evaluate

    W = lmfeat.fit_prior_mean_head(
        np.concatenate([p["emb_leakfree"] for p in head]),
        np.concatenate([p["y"] for p in head]), l2=10.0)
    rows = {}
    for level in args.levels:
        rng = np.random.default_rng(int(1000 * level))
        for pi, p in shard_pieces(ev, args.shard):
            score = piece_score(p)
            L = laplacian(build_adjacency(score))
            mu_lm = lmfeat.apply_prior_mean(p["emb_leakfree"], W)
            Y = np.asarray(p["y"], dtype=float)
            stds = Y.std(axis=0)
            noise = rng.standard_normal(Y.shape) * (level * stds)
            Yn = Y + noise
            # old rows (per channel; LM mean, the published win condition)
            for method in ("graph", "graph-oracle"):
                yt, pr, sd = [], [], []
                for ci in range(3):
                    m, s_ = denoise_channel(L, Yn[:, ci], mu_lm[:, ci],
                                            level * stds[ci], method)
                    yt.append(Y[:, ci]); pr.append(m); sd.append(s_)
                mm = evaluate(np.concatenate(yt), np.concatenate(pr),
                              np.concatenate(sd), level=0.9)
                rows.setdefault((level, f"old LM {method}"), []).append(
                    (mm["rmse"], mm["nll"], mm["coverage@0.90"]))
            # GP-first rows: blind and oracle noise
            for variant, oracle in (("GP-feat blind", False), ("GP-feat oracle", True)):
                gp = gp_for(p, "GP-feat")
                nf = ((level * stds) ** 2 if oracle else None)
                floor = (None if oracle
                         else np.full(3, 1e-8))
                x_hat, _ = gp.fit(Yn, np.ones(len(Y), dtype=bool),
                                  noise_floor=floor, noise_fixed=nf,
                                  maxiter=args.maxiter)
                M, S = gp.posterior(Yn, np.ones(len(Y), dtype=bool), x_hat)
                mm = evaluate(np.concatenate([Y[:, c] for c in range(3)]),
                              np.concatenate([M[:, c] for c in range(3)]),
                              np.concatenate([S[:, c] for c in range(3)]), level=0.9)
                rows.setdefault((level, variant), []).append(
                    (mm["rmse"], mm["nll"], mm["coverage@0.90"]))
            print(f"[denoise] level {level} piece {pi} done", flush=True)
    return rows


# ------------------------------------------------------------------------ completion
def task_completion(args, head, ev):
    from score_bundle import imputation_eval as ie
    from score_bundle.downstream import block_mask, piece_score, prefix_mask
    from score_bundle.gp import MultiOutputGraphGP
    from score_bundle.lm import features as lmfeat
    from score_bundle.metrics import evaluate

    W = lmfeat.fit_prior_mean_head(
        np.concatenate([p["emb_leakfree"] for p in head]),
        np.concatenate([p["y"] for p in head]), l2=10.0)
    rows = {}
    for kind in args.kinds:
        for frac in args.fracs:
            rng = np.random.default_rng(0)
            for pi, p in shard_pieces(ev, args.shard):
                score = piece_score(p)
                Y = np.asarray(p["y"], dtype=float)
                mask = (prefix_mask(len(Y), frac) if kind == "prefix"
                        else block_mask(len(Y), rng, observed_frac=frac))
                held = ~mask
                mu_lm = lmfeat.apply_prior_mean(p["emb_leakfree"], W)
                cells = ie.impute_methods(score, Y, {"LM": mu_lm}, mask,
                                          fit_hyper=True, rng=rng,
                                          noise_floor_frac=0.05)
                for (mn, g), cell in cells.items():
                    mm = evaluate(cell.y, cell.pred, cell.std, level=0.9)
                    rows.setdefault((kind, frac, f"old {mn}{'+graph' if g else ''}"),
                                    []).append((mm["rmse"], mm["nll"]))
                for variant in ("GP-feat", "GP-featlm"):
                    gp = gp_for(p, variant)
                    floor = 0.05 * np.array([max(float(np.var(Y[mask, c])), 1e-10)
                                             for c in range(3)])
                    x_hat, _ = gp.fit(Y, mask, noise_floor=floor,
                                      maxiter=args.maxiter)
                    M, S = gp.posterior(Y, mask, x_hat)
                    nv = gp.unpack(x_hat)["noise"]
                    yt = np.concatenate([Y[held, c] for c in range(3)])
                    pr = np.concatenate([M[held, c] for c in range(3)])
                    sd = np.concatenate([np.sqrt(S[held, c] ** 2 + nv[c])
                                         for c in range(3)])
                    mm = evaluate(yt, pr, sd, level=0.9)
                    rows.setdefault((kind, frac, variant), []).append(
                        (mm["rmse"], mm["nll"]))
                print(f"[completion] {kind} {frac} piece {pi} done", flush=True)
    return rows


# -------------------------------------------------------------------------- era
def task_era(args, head, ev):
    from score_bundle.downstream import (era_of, loo_nearest_centroid, piece_score,
                                         style_features)

    rows = {}
    labels, feats_raw, feats_gp = [], [], []
    for pi, p in shard_pieces(ev, args.shard):
        era = era_of(p.get("composer", ""))
        if era is None:
            continue
        Y = np.asarray(p["y"], dtype=float)
        order = np.argsort(np.asarray(p["onset"]), kind="stable")
        labels.append(era)
        feats_raw.append(style_features(Y, order=order))
        gp = gp_for(p, "GP-feat")
        floor = 0.05 * np.array([float(np.var(Y[:, c])) for c in range(3)])
        x_hat, _ = gp.fit(Y, np.ones(len(Y), dtype=bool), noise_floor=floor,
                          maxiter=args.maxiter)
        M, _ = gp.posterior(Y, np.ones(len(Y), dtype=bool), x_hat)
        feats_gp.append(style_features(M, order=order))
        print(f"[era] piece {pi} ({era}) done", flush=True)
    rows[("era", "raw")] = (labels, feats_raw)
    rows[("era", "GP-denoised")] = (labels, feats_gp)
    return rows


# ------------------------------------------------------------------------ selective
def task_selective(args):
    """Pure analysis of existing imputation cells — no new fits."""
    import glob
    from score_bundle.downstream import selective_report

    def load_cells(pattern, mean_name=None):
        cells = {}
        for f in glob.glob(pattern):
            blob = pickle.load(open(f, "rb"))
            for k, v in blob["cells"].items():
                if mean_name is None or k[0] == mean_name:
                    cells[(f, *k[1:])] = v[:3] if len(v) > 3 else v
        return cells

    systems = {
        "old headline": load_cells("results/kernels_featlm/harmonic_vl.pkl", "LM"),
        "GP-first b_featlm": load_cells("results/graphgp_v2/b_featlm*.pkl", "GP"),
    }
    out = {}
    for name, cells in systems.items():
        if not cells:
            print(f"[selective] no cells for {name} (run R1 first?)"); continue
        yt = np.concatenate([c[0] for c in cells.values()])
        pr = np.concatenate([c[1] for c in cells.values()])
        sd = np.concatenate([c[2] for c in cells.values()])
        out[name] = selective_report(yt, pr, sd)
    return out


# --------------------------------------------------------------------------- report
def report(args):
    import glob
    from score_bundle.downstream import loo_nearest_centroid

    def merged(task):
        rows = {}
        for f in glob.glob(os.path.join(args.out_dir, f"{task}.shard*.pkl")):
            for k, v in pickle.load(open(f, "rb")).items():
                if task == "era":
                    la, fe = rows.get(k, ([], []))
                    rows[k] = (la + v[0], fe + v[1])
                else:
                    rows.setdefault(k, []).extend(v)
        return rows

    boot = np.random.default_rng(7)
    r = merged("anomaly")
    if r:
        print("\n=== ANOMALY (AUROC [95% CI] / AP), per channel ===")
        methods = sorted({k[1] for k in r})
        for cname in CHANNELS:
            for m in methods:
                vals = r.get((cname, m))
                if not vals:
                    continue
                a, lo, hi = bootstrap_ci([v[0] for v in vals], rng=boot)
                ap = float(np.mean([v[1] for v in vals]))
                print(f"  {cname:8s} {m:12s} {a:.3f} [{lo:.3f},{hi:.3f}]  AP {ap:.3f}")

    r = merged("denoise")
    if r:
        print("\n=== DENOISE (RMSE / NLL / cov@.9) ===")
        for key in sorted(r, key=str):
            vals = np.array(r[key])
            print(f"  level {key[0]:.1f} {key[1]:18s} "
                  f"RMSE {vals[:,0].mean():.4f}  NLL {vals[:,1].mean():+.3f}  "
                  f"cov {vals[:,2].mean():.3f}")

    r = merged("completion")
    if r:
        print("\n=== COMPLETION (RMSE / NLL) ===")
        for key in sorted(r, key=str):
            vals = np.array(r[key])
            print(f"  {key[0]:7s} frac {key[1]:.2f} {key[2]:14s} "
                  f"RMSE {vals[:,0].mean():.4f}  NLL {vals[:,1].mean():+.3f}")

    r = merged("era")
    if r:
        print("\n=== ERA (LOO nearest-centroid accuracy) ===")
        for key, (labels, feats) in sorted(r.items(), key=str):
            if len(labels) >= 8:
                acc, per = loo_nearest_centroid(np.array(feats), labels)
                maj = max(np.bincount([hash(l) % 97 for l in labels])) / len(labels)
                print(f"  {key[1]:14s} acc {acc:.3f} (n={len(labels)}, "
                      f"majority~{max(labels.count(l) for l in set(labels))/len(labels):.3f})")

    sel = task_selective(args)
    if sel:
        print("\n=== SELECTIVE (risk-coverage) ===")
        for name, m in sel.items():
            print(f"  {name:20s} rmse {m['rmse']:.4f}  aurc {m['aurc']:.4f}  "
                  f"excess {m['excess']:+.4f}  rmse@50% {m['rmse_at_50']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    choices=["anomaly", "denoise", "completion", "era",
                             "selective", "report"])
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--anomaly-frac", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--kinds", nargs="+", default=["prefix", "block"])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--maxiter", type=int, default=120)
    args = ap.parse_args()

    if args.task == "report":
        report(args); return
    if args.task == "selective":
        for name, m in task_selective(args).items():
            print(name, m)
        return

    from score_bundle.downstream import load_piece_arrays
    head, ev, _ = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    os.makedirs(args.out_dir, exist_ok=True)
    rows = {"anomaly": task_anomaly, "denoise": task_denoise,
            "completion": task_completion, "era": task_era}[args.task](args, head, ev)
    k = args.shard.replace("/", "_")
    path = os.path.join(args.out_dir, f"{args.task}.shard{k}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(rows, fh)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
