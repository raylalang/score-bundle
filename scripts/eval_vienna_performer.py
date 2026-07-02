#!/usr/bin/env python
"""Downstream task 6 — performer identification on the Vienna 4x22 corpus.

The task ASAP/MAESTRO cannot support: 22 pianists each play the **same four** excerpts, so
the score is held constant and the only signal is expression. We classify the performer
(1-of-22, chance 0.045) from expression-style features, with **leave-one-piece-out** cross
-validation (train on 3 pieces, test on the 4th) so the classifier must generalize a
performer's style across repertoire, not memorize a rendition.

On-thesis question (mirrors the ASAP style probe, but with real labels and *no* score
confound): does graph-denoising the expression yield more performer-discriminative
features than the raw observed expression?

    raw    — style aggregates of the observed per-note y = [tau, log r, v];
    graph  — style aggregates of the graph-denoised expression (EB-fit GMRF posterior mean);
    both   — segment-level variant (each performance split into K contiguous segments →
             more samples per performer; per-segment accuracy).

Needs a local checkout of the corpus and partitura:

    git clone https://github.com/CPJKU/vienna4x22 ../data/vienna4x22
    pip install partitura
    python scripts/eval_vienna_performer.py --root ../data/vienna4x22

If the corpus or partitura is absent, prints how to obtain them and exits 0 (so CI stays
green); see docs/vienna_4x22_scoping.md.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _style_features_segmented(y, order, n_seg):
    """List of style-feature vectors, one per contiguous score-order segment."""
    from score_bundle.downstream import style_features

    idx = np.asarray(order)
    feats = []
    bounds = np.linspace(0, len(idx), n_seg + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 8:
            feats.append(style_features(y[idx[a:b]]))
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="../data/vienna4x22", help="corpus checkout")
    ap.add_argument("--n-seg", type=int, default=4, help="segments/performance (segment task)")
    ap.add_argument("--max-notes", type=int, default=250,
                    help="cap notes/performance (the per-piece graph EB fit is O(N^3); the "
                         "prefix is a fixed score excerpt shared across performers, so the "
                         "comparison stays fair)")
    args = ap.parse_args()

    try:
        import partitura  # noqa: F401
    except Exception:
        print("partitura not installed. `pip install partitura` (>=1.2.0).")
        print("See docs/vienna_4x22_scoping.md."); sys.exit(0)

    from score_bundle import vienna
    from score_bundle.downstream import (
        denoise_channel,
        grouped_nearest_centroid,
        style_features,
    )
    from score_bundle.graph import build_adjacency, laplacian
    from score_bundle.score import Score

    try:
        recs = vienna.load_vienna_meta(args.root)
    except FileNotFoundError as exc:
        print(exc)
        print("\nObtain it:  git clone https://github.com/CPJKU/vienna4x22 "
              f"{args.root}\nSee docs/vienna_4x22_scoping.md."); sys.exit(0)

    print(f"{len(recs)} performances | "
          f"{len(set(r.piece for r in recs))} pieces x "
          f"{len(set(r.performer for r in recs))} performers")

    # per-performance features (raw and graph-denoised), plus segment-level features
    Xraw_p, Xgraph_p, perf_p, piece_p = [], [], [], []
    Xraw_s, perf_s, piece_s = [], [], []
    for r in recs:
        try:
            score, y = vienna.load_vienna_performance(r.match_path)
        except Exception as exc:
            print(f"  skip {r.piece} {r.performer}: {exc}"); continue
        if args.max_notes and len(score) > args.max_notes:
            score = Score(score.notes[: args.max_notes]); y = y[: args.max_notes]
        order = np.argsort(score.onset)
        L = laplacian(build_adjacency(score))
        y_dn = np.column_stack([
            denoise_channel(L, y[:, c], np.zeros(len(y)),
                            max(float(np.std(y[:, c])), 1e-6), "graph")[0]
            for c in range(3)
        ])
        Xraw_p.append(style_features(y, order))
        Xgraph_p.append(style_features(y_dn, order))
        perf_p.append(r.performer); piece_p.append(r.piece)
        for f in _style_features_segmented(y, order, args.n_seg):
            Xraw_s.append(f); perf_s.append(r.performer); piece_s.append(r.piece)

    if not Xraw_p:
        print("no performances loaded; aborting."); sys.exit(0)

    n_perf = len(set(perf_p))
    chance = 1.0 / n_perf
    print(f"\nPerformer ID (leave-one-piece-out, chance = {chance:.3f})")
    print(f"{'features':16s} {'accuracy':>9s} {'n':>5s}")
    for name, X in [("raw (per-perf)", Xraw_p), ("graph (per-perf)", Xgraph_p)]:
        acc, n = grouped_nearest_centroid(np.array(X), perf_p, piece_p)
        print(f"{name:16s} {acc:9.3f} {n:5d}")
    acc_s, n_s = grouped_nearest_centroid(np.array(Xraw_s), perf_s, piece_s)
    print(f"{'raw (per-seg)':16s} {acc_s:9.3f} {n_s:5d}")

    print("\nReading: chance is 1/22 = 0.045. 'raw' vs 'graph' asks whether graph-denoised "
          "expression\nis a more performer-discriminative feature; the per-segment row is a "
          "higher-N variant.\nLeave-one-piece-out means piece identity cannot be exploited "
          "— only performer style.")


if __name__ == "__main__":
    main()
