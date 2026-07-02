#!/usr/bin/env python
"""Downstream task 5 — composer-era style classification from inferred expression.

Tests a different kind of claim than the regression tasks: does the graph-structured
posterior yield expression *features* that are more useful downstream? We classify each
piece's composer era (Baroque / Classical / Romantic / Modern) from **expression alone**
— per-note timing/articulation/dynamics aggregated into style descriptors
(`downstream.style_features`), never pitch or score identity — and compare three feature
sources:

    raw    — aggregates of the *observed* per-note expression y;
    graph  — aggregates of the graph-denoised expression (full-observation GMRF posterior
             mean, EB-fit + noise floor), i.e. the structured prior applied as a feature
             cleaner;
    lm     — aggregates of the LM prior mean mu_LM (a pure score-conditioned guess; a
             lower bound — it has no access to the actual performance).

Leave-one-piece-out nearest-centroid accuracy / macro-recall over head+eval pieces (this
is a piece-level task, so the head/eval imputation split does not apply; the LM-head fit
does not leak era labels). Honest baseline: majority-class accuracy. Numpy-only:

    python scripts/eval_asap_style.py --arrays-cache .cache/asap_arrays_named.pkl
"""
from __future__ import annotations

import argparse

import numpy as np

from score_bundle.downstream import (
    denoise_channel,
    era_of,
    load_piece_arrays,
    loo_nearest_centroid,
    piece_score,
    style_features,
)
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.lm import features as lmfeat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--embeddings", default="emb_scoreonly", choices=["emb", "emb_scoreonly"])
    args = ap.parse_args()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    pieces = head + ev
    # fit the LM head on head pieces only (for the 'lm' feature source)
    H = np.concatenate([p[args.embeddings] for p in head])
    W = lmfeat.fit_prior_mean_head(H, np.concatenate([p["y"] for p in head]), l2=args.l2)

    X_raw, X_graph, X_lm, labels, composers = [], [], [], [], []
    for p in pieces:
        era = era_of(p.get("composer", ""))
        if era is None:
            continue
        y = p["y"]
        order = np.argsort(p["onset"])
        score = piece_score(p)
        L = laplacian(build_adjacency(score))
        # graph-denoised expression: per-channel full-observation posterior mean
        y_dn = np.column_stack([
            denoise_channel(L, y[:, c], np.zeros(len(y)),
                            max(float(np.std(y[:, c])), 1e-6), "graph")[0]
            for c in range(3)
        ])
        mu_lm = lmfeat.apply_prior_mean(p[args.embeddings], W)
        X_raw.append(style_features(y, order))
        X_graph.append(style_features(y_dn, order))
        X_lm.append(style_features(mu_lm, order))
        labels.append(era)
        composers.append((p.get("composer") or "?").split()[0])

    X_raw = np.array(X_raw); X_graph = np.array(X_graph); X_lm = np.array(X_lm)
    labels = np.array(labels)
    n = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    majority = counts.max() / n
    print(f"{n} pieces with era labels | classes: "
          + ", ".join(f"{c}={k}" for c, k in zip(classes, counts)))
    print(f"majority-class accuracy = {majority:.3f}\n")

    print(f"{'features':8s} {'LOO acc':>8s}   per-class recall")
    for name, X in [("lm", X_lm), ("raw", X_raw), ("graph", X_graph)]:
        acc, recall = loo_nearest_centroid(X, labels)
        rec = "  ".join(f"{c}:{recall[c]:.2f}" for c in classes)
        print(f"{name:8s} {acc:8.3f}   {rec}")

    print("\nReading: 'raw' vs 'graph' asks whether graph-denoising the expression yields "
          "cleaner\nstyle features; 'lm' (score-only guess, no performance) is a lower "
          "bound. All use\nexpression aggregates only — never pitch/score identity — so "
          "the composer is not\ntrivially readable. Confounded (expression correlates with "
          "meter/tempo, hence era);\nreport as a feature-quality probe, not a style-ID "
          "system.")


if __name__ == "__main__":
    main()
