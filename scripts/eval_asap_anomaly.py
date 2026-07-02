#!/usr/bin/env python
"""Downstream task 2 — performance-error (anomaly) detection on held-out ASAP.

Inject controlled errors into a fraction of notes (a +-``scale``-sigma shift on one
channel) and rank all notes by model surprise; a calibrated, structured posterior should
separate corrupted from clean notes better than an unstructured residual z-score.  This
is the downstream task that directly cashes in *calibration*: ranking quality (AUROC /
average precision) depends on per-note predictive uncertainty, not just the mean.

Methods per channel:
    zero / LM mean,  graph off  — homoscedastic z-score around the mean;
    zero / LM mean,  graph on   — EB-fit GMRF + leave-one-out predictive NLL.

Caveat recorded in the output: with ``--embeddings emb`` the LM saw the *clean* performed
velocities at the leaky read-out, so LM rows on the ``v`` channel are oracle-ish; the
default leak-free embeddings avoid that (published tables used the ``emb_scoreonly``
band-aid — conservative for LM rows, irrelevant for the zero-mean rows that carry the
verdict).  Runs from the named array cache (numpy-only):

    python scripts/eval_asap_anomaly.py --arrays-cache .cache/asap_arrays_named.pkl
"""
from __future__ import annotations

import argparse

import numpy as np

from score_bundle.downstream import (
    anomaly_scores,
    auroc,
    average_precision,
    inject_anomalies,
    load_piece_arrays,
    piece_score,
)
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.lm import features as lmfeat


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
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--anomaly-frac", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=3.0, help="error size in channel stds")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--embeddings", default="emb_leakfree",
                    choices=["emb", "emb_scoreonly", "emb_leakfree"])
    args = ap.parse_args()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    print(f"{len(head)} head + {len(ev)} eval pieces | frac={args.anomaly_frac} "
          f"scale={args.scale} sigma | embeddings={args.embeddings}")

    H = np.concatenate([p[args.embeddings] for p in head])
    W = lmfeat.fit_prior_mean_head(H, np.concatenate([p["y"] for p in head]), l2=args.l2)

    channels = ["tau", "log r", "v"]
    methods = [("zero", False), ("zero", True), ("LM", False), ("LM", True)]
    # per (channel, method): list of per-(piece x seed) AUROC / AP
    res_auroc = {(c, m): [] for c in channels for m in methods}
    res_ap = {(c, m): [] for c in channels for m in methods}

    for s in range(args.seeds):
        rng = np.random.default_rng(100 + s)
        for p in ev:
            score = piece_score(p)
            L = laplacian(build_adjacency(score))
            mu_lm = lmfeat.apply_prior_mean(p[args.embeddings], W)
            for ci, cname in enumerate(channels):
                y = p["y"][:, ci]
                y_bad, labels = inject_anomalies(y, rng, frac=args.anomaly_frac,
                                                 scale=args.scale)
                for mname, use_graph in methods:
                    mean = np.zeros_like(y) if mname == "zero" else mu_lm[:, ci]
                    sc = anomaly_scores(L, y_bad, mean, use_graph=use_graph)
                    res_auroc[(cname, (mname, use_graph))].append(auroc(labels, sc))
                    res_ap[(cname, (mname, use_graph))].append(average_precision(labels, sc))
        print(f"seed {s + 1}/{args.seeds} done", flush=True)

    boot_rng = np.random.default_rng(7)
    print(f"\nAnomaly detection ({len(ev)} pieces x {args.seeds} seeds; "
          f"chance AUROC=0.5, chance AP={args.anomaly_frac:.2f})")
    print(f"{'channel':8s} {'mean':5s} {'graph':6s} {'AUROC [95% CI]':>24s} {'AP':>8s}")
    pooled = {}
    for cname in channels:
        for mname, use_graph in methods:
            a, lo, hi = bootstrap_ci(res_auroc[(cname, (mname, use_graph))], rng=boot_rng)
            ap_m = float(np.nanmean(res_ap[(cname, (mname, use_graph))]))
            pooled.setdefault((mname, use_graph), []).extend(
                res_auroc[(cname, (mname, use_graph))])
            print(f"{cname:8s} {mname:5s} {('on' if use_graph else 'off'):6s} "
                  f"{a:7.3f} [{lo:.3f},{hi:.3f}]   {ap_m:8.3f}")
    print("\npooled over channels:")
    for (mname, use_graph), vals in pooled.items():
        a, lo, hi = bootstrap_ci(vals, rng=boot_rng)
        print(f"  {mname:5s} {('on' if use_graph else 'off'):4s}  "
              f"AUROC {a:.3f} [{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
