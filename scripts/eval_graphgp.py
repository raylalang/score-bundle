#!/usr/bin/env python
"""GP-first evaluation: the orthodox multi-output graph GP under the strict protocol.

Runs the staged orthodoxy ladder of docs/graphgp_first_design.md on the SAME cached
masks as every published number (.cache/kernel_sweep_inputs.pkl), so paired per-piece
comparisons against the current pipeline's cells are valid:

    a_diag    3 independent single-channel GPs (nested special case — the GATE:
              must land near the published zero+graph cell)
    a_icm     + coregionalization B (cross-channel coupling), additive shape
    a_icm_m2  same, Matern-2 shape
    b_feat    + linear kernel on score features (marginalized Bayesian linear mean)
    b_featlm  + linear kernel on mask-aware LM embeddings as well
    c_graph   b_feat with the graph's own (ell_b, ell_p) learned by evidence
    c_harm    b_feat on the harmonic graph with (ell_b, ell_p, chord, vl) learned
    d_corpus  b_feat with ONE corpus-level hyperparameter set fit on head pieces
              (frozen for eval — canonical train/test, no per-piece fitting)

    python scripts/eval_graphgp.py --stage run --configs a_diag
    python scripts/eval_graphgp.py --stage run --configs c_graph --shard 0/6
    python scripts/eval_graphgp.py --stage report

Numpy core; scipy used by the fit when available.  Results pickles mirror the
kernel-sweep schema so the report stage can pair against results/kernels*/ baselines.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

INPUTS = ".cache/kernel_sweep_inputs.pkl"
OUT_DIR = "results/graphgp"
CHANNELS = ["tau", "log r", "v"]

CONFIGS = ["a_diag", "a_icm", "a_icm_m2", "b_feat", "b_featlm",
           "c_graph", "c_harm", "d_corpus"]

BASELINES = {  # label -> (pickle path, mean-block name)
    "zero+graph": ("results/kernels/additive.pkl", "zero"),
    "LM+graph": ("results/kernels/additive.pkl", "LM"),
    "feat+LM+graph": ("results/kernels_featlm/additive.pkl", "LM"),
    "headline (feat+LM+harm)": ("results/kernels_featlm/harmonic_vl.pkl", "LM"),
}


def bootstrap_ci(vals, B=2000, rng=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def zscore_cols(X: np.ndarray) -> np.ndarray:
    """Per-piece column standardization (score-only inputs — no target content)."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / np.maximum(sd, 1e-9)


def piece_setup(p, config: str, emb=None):
    """(features list, graph builder over graph-params) for one piece."""
    from score_bundle.baselines import rich_score_features
    from score_bundle.downstream import piece_score
    from score_bundle.graph import (build_adjacency, build_adjacency_harmonic,
                                    laplacian)

    score = piece_score(p)
    feats = []
    if config.startswith(("b_", "c_", "d_")):
        X = zscore_cols(rich_score_features(score, rff_dim=0))
        feats.append(np.concatenate([X, np.ones((len(X), 1))], axis=1))
    if config == "b_featlm":
        assert emb is not None, "b_featlm needs the mask-aware embedding dump"
        feats.append(zscore_cols(emb))

    if config == "c_graph":
        def graph_eig(gp_params):  # gp_params = [log ell_b, log ell_p]
            eb, ep = np.exp(gp_params)
            return np.linalg.eigh(laplacian(build_adjacency(score, ell_b=eb, ell_p=ep)))
        n_graph = 2
        g0 = np.log([2.0, 4.0])
    elif config == "c_harm":
        def graph_eig(gp_params):  # [log ell_b, log ell_p, log chord_w, log vl_w]
            eb, ep, cw, vw = np.exp(gp_params)
            return np.linalg.eigh(laplacian(build_adjacency_harmonic(
                score, ell_b=eb, ell_p=ep, chord_weight=cw, vl_weight=vw)))
        n_graph = 4
        g0 = np.log([2.0, 4.0, 1.0, 1.0])
    else:
        eig = np.linalg.eigh(laplacian(build_adjacency(score)))

        def graph_eig(gp_params):
            return eig
        n_graph = 0
        g0 = np.zeros(0)
    return feats, graph_eig, n_graph, g0


def fit_and_predict(Y, mask, feats, graph_eig, n_graph, g0, kernel, x_init=None,
                    frozen_x=None, maxiter=200):
    """One joint fit (or frozen apply) + posterior; returns (yt, pr, sd, ch, info)."""
    from score_bundle.gp import MultiOutputGraphGP

    held = ~mask
    floor = 0.05 * np.array([float(np.var(Y[mask, c])) for c in range(Y.shape[1])])

    def make_gp(gparams):
        nu, U = graph_eig(gparams)
        return MultiOutputGraphGP(nu, U, kernel=kernel, features=feats,
                                  n_channels=Y.shape[1])

    if n_graph == 0:
        gp = make_gp(g0)
        if frozen_x is not None:
            x_hat, info = frozen_x, {"optimizer": "frozen"}
        else:
            x_hat, info = gp.fit(Y, mask, x0=x_init, noise_floor=floor, maxiter=maxiter)
        M, S = gp.posterior(Y, mask, x_hat)
        nv = gp.unpack(x_hat)["noise"]
    else:
        # joint optimization over [graph params | gp params]
        from score_bundle.optimize import nelder_mead
        gp0 = make_gp(g0)
        x0 = np.concatenate([g0, gp0.x0() if x_init is None else x_init])
        floor_log = np.log(np.maximum(floor, 1e-12))
        k = Y.shape[1]

        def neg(z):
            try:
                gp = make_gp(z[:n_graph])
                xg = z[n_graph:].copy()
                xg[-k:] = np.maximum(xg[-k:], floor_log)
                v = -gp.log_marginal_likelihood(Y, mask, xg)
            except (np.linalg.LinAlgError, ValueError):
                return 1e12
            return v if np.isfinite(v) else 1e12

        best = None
        try:
            from scipy.optimize import minimize
            res = minimize(neg, x0, method="L-BFGS-B",
                           options={"maxiter": maxiter, "eps": 1e-5})
            best = res.x
        except ImportError:
            pass
        if best is None:
            best = nelder_mead(neg, x0, max_iter=1500)
        polished = nelder_mead(neg, best, max_iter=300)
        if neg(polished) < neg(best):
            best = polished
        gp = make_gp(best[:n_graph])
        x_hat = best[n_graph:].copy()
        x_hat[-k:] = np.maximum(x_hat[-k:], floor_log)
        M, S = gp.posterior(Y, mask, x_hat)
        nv = gp.unpack(x_hat)["noise"]
        info = {"optimizer": "joint", "graph_params": np.exp(best[:n_graph]).tolist()}

    yt, pr, sd, ch = [], [], [], []
    for c in range(Y.shape[1]):
        yt.append(Y[held, c]); pr.append(M[held, c])
        sd.append(np.sqrt(S[held, c] ** 2 + nv[c]))
        ch.append(np.full(int(held.sum()), c, dtype=int))
    return (np.concatenate(yt), np.concatenate(pr), np.concatenate(sd),
            np.concatenate(ch)), info


# ------------------------------------------------------------------------- run
def stage_run(args) -> None:
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.gp import MultiOutputGraphGP

    with open(args.inputs, "rb") as fh:
        inputs = pickle.load(fh)
    masks, imeta = inputs["masks"], inputs["meta"]
    emb_dump = None
    if os.path.exists(args.emb_dump):
        with open(args.emb_dump, "rb") as fh:
            emb_dump = pickle.load(fh)["emb_ma"]
    head, ev, _ = load_piece_arrays(args.arrays_cache)
    ev = ev[: imeta["n_eval_pieces"]]
    seeds = imeta["seeds"]
    os.makedirs(args.out_dir, exist_ok=True)
    shard_k, shard_n = map(int, args.shard.split("/")) if "/" in args.shard else (0, 1)

    for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if config not in CONFIGS:
            print(f"unknown config {config!r}; known: {CONFIGS}"); sys.exit(1)
        kernel = "matern2" if config == "a_icm_m2" else "additive"
        print(f"=== {config}: kernel={kernel} shard {shard_k}/{shard_n}", flush=True)

        frozen = None
        if config == "d_corpus":
            frozen = fit_corpus_params(head, kernel, args)
            print(f"[d_corpus] corpus params fit on {args.corpus_pieces} head pieces",
                  flush=True)

        cells, infos = {}, []
        for s in range(seeds):
            for pi, p in enumerate(ev):
                if (s * len(ev) + pi) % shard_n != shard_k:
                    continue
                Y = np.asarray(p["y"], dtype=float)
                mask = masks[(pi, s)]
                emb = emb_dump[(pi, s)] if (config == "b_featlm") else None
                base_cfg = "b_feat" if config == "d_corpus" else config
                feats, graph_eig, n_graph, g0 = piece_setup(p, base_cfg, emb=emb)
                if config == "a_diag":
                    # three single-channel fits through the same machinery
                    yt, pr, sd, ch = [], [], [], []
                    for c in range(3):
                        cell, info = fit_and_predict(
                            Y[:, c:c + 1], mask, feats, graph_eig, n_graph, g0,
                            kernel, maxiter=args.maxiter)
                        yt.append(cell[0]); pr.append(cell[1]); sd.append(cell[2])
                        ch.append(np.full(len(cell[0]), c, dtype=int))
                    cell = (np.concatenate(yt), np.concatenate(pr),
                            np.concatenate(sd), np.concatenate(ch))
                else:
                    cell, info = fit_and_predict(
                        Y, mask, feats, graph_eig, n_graph, g0, kernel,
                        frozen_x=frozen, maxiter=args.maxiter)
                    infos.append(info)
                cells[("GP", pi, s)] = cell
            print(f"[{config}] seed {s + 1}/{seeds} done ({len(cells)} cells)", flush=True)

        suffix = f".shard{shard_k}" if shard_n > 1 else ""
        path = os.path.join(args.out_dir, f"{config}{suffix}.pkl")
        with open(path, "wb") as fh:
            pickle.dump({"row": config, "cells": cells,
                         "meta": {**imeta, "kernel": kernel, "shard": args.shard}}, fh)
        print(f"[{config}] wrote {path}", flush=True)


def fit_corpus_params(head, kernel, args):
    """ONE hyperparameter set by summed marginal likelihood over head pieces,
    each under a fixed random 60% observation mask (the deployment regime)."""
    from score_bundle import imputation_eval as ie
    from score_bundle.gp import MultiOutputGraphGP
    from score_bundle.optimize import nelder_mead

    rng = np.random.default_rng(7)
    problems = []
    for p in head[: args.corpus_pieces]:
        Y = np.asarray(p["y"], dtype=float)
        mask = ie.random_mask(len(Y), rng, observed_frac=0.6)
        feats, graph_eig, _, g0 = piece_setup(p, "b_feat")
        nu, U = graph_eig(g0)
        problems.append((MultiOutputGraphGP(nu, U, kernel=kernel, features=feats),
                         Y, mask))
    gp0 = problems[0][0]
    floors = [0.05 * np.array([float(np.var(Y[m, c])) for c in range(3)])
              for _, Y, m in problems]
    k = 3

    def neg(x):
        tot = 0.0
        for (gp, Y, m), fl in zip(problems, floors):
            xg = x.copy()
            xg[-k:] = np.maximum(xg[-k:], np.log(np.maximum(fl, 1e-12)))
            try:
                tot -= gp.log_marginal_likelihood(Y, m, xg)
            except (np.linalg.LinAlgError, ValueError):
                return 1e12
        return tot

    best = None
    try:
        from scipy.optimize import minimize
        best = minimize(neg, gp0.x0(), method="L-BFGS-B",
                        options={"maxiter": args.maxiter, "eps": 1e-5}).x
    except ImportError:
        pass
    if best is None:
        best = nelder_mead(neg, gp0.x0(), max_iter=1500)
    return best


# ---------------------------------------------------------------------- report
def stage_report(args) -> None:
    from score_bundle import imputation_eval as ie
    from score_bundle.metrics import evaluate

    def load_config(config):
        merged = {}
        base = os.path.join(args.out_dir, f"{config}.pkl")
        if os.path.exists(base):
            with open(base, "rb") as fh:
                merged.update(pickle.load(fh)["cells"])
        for k in range(64):
            sp = os.path.join(args.out_dir, f"{config}.shard{k}.pkl")
            if os.path.exists(sp):
                with open(sp, "rb") as fh:
                    merged.update(pickle.load(fh)["cells"])
        return merged or None

    def load_baseline(path, mean_name):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        return {("GP", pi, s): (yt, pr, sd, ch)
                for (mn, pi, s), (yt, pr, sd, ch) in blob["cells"].items()
                if mn == mean_name}

    def pooled(cells):
        acc = ie.MetricAccumulator()
        for (_, pi, s), (yt, pr, sd, ch) in cells.items():
            acc.add({("k", True): ie.CellResult(yt, pr, sd, ch)})
        return acc

    def per_piece(cells, field):
        by = {}
        for (_, pi, s), (yt, pr, sd, ch) in cells.items():
            b = by.setdefault(pi, [[], [], []])
            b[0].append(yt); b[1].append(pr); b[2].append(sd)
        return {pi: evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                             np.concatenate(v[2]), level=0.9)[field]
                for pi, v in by.items()}

    rows = {}
    for config in CONFIGS:
        cells = load_config(config)
        if cells:
            rows[config] = cells
    for label, (path, mn) in BASELINES.items():
        if os.path.exists(path):
            rows[label] = load_baseline(path, mn)
    print(f"rows: {list(rows)}\n")

    boot_rng = np.random.default_rng(11)
    base_label = args.baseline
    base_pp = {f: per_piece(rows[base_label], f) for f in ("rmse", "nll")} \
        if base_label in rows else None
    hdr = (f"{'row':26s} {'RMSE':>8s} {'NLL':>9s} {'cov@.9':>8s} {'med-cell':>9s} "
           f"{'worst':>8s}  dRMSE vs {base_label} [95% CI]   dNLL [95% CI]")
    print(hdr)
    for label, cells in rows.items():
        m = pooled(cells).report(level=0.9)[("k", True)]
        txt = {}
        for f in ("rmse", "nll"):
            if base_pp is None or label == base_label:
                txt[f] = "—"
                continue
            mine = per_piece(cells, f)
            common = sorted(set(mine) & set(base_pp[f]))
            d = np.array([mine[pi] - base_pp[f][pi] for pi in common])
            mu, lo, hi = bootstrap_ci(d, B=2000, rng=boot_rng)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            txt[f] = f"{mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}"
        print(f"{label:26s} {m['rmse']:8.4f} {m['nll']:9.4f} {m['coverage@0.90']:8.3f} "
              f"{m['rmse_median_cell']:9.4f} {m['rmse_worst_cell']:8.4f}  "
              f"{txt['rmse']:>30s} {txt['nll']:>28s}")

    print("\nper-channel (RMSE / cov@.9):")
    print(f"{'row':26s}" + "".join(f" {c + ' RMSE':>10s} {c + ' cov':>9s}" for c in CHANNELS))
    for label, cells in rows.items():
        rep = pooled(cells).report_by_channel(CHANNELS, level=0.9)
        line = f"{label:26s}"
        for c in CHANNELS:
            mm = rep.get(("k", True, c))
            line += (f" {mm['rmse']:10.4f} {mm['coverage@0.90']:9.3f}" if mm
                     else f" {'—':>10s} {'—':>9s}")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["run", "report"])
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--inputs", default=INPUTS)
    ap.add_argument("--emb-dump", default=".cache/kernel_sweep_emb_ma.pkl")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--configs", default="a_diag")
    ap.add_argument("--shard", default="0/1", help="k/n shard of the (piece, seed) cells")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--corpus-pieces", type=int, default=20)
    ap.add_argument("--baseline", default="headline (feat+LM+harm)")
    args = ap.parse_args()
    {"run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
