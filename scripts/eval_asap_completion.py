#!/usr/bin/env python
"""Downstream task 1 — expressive rendering / performance completion on held-out ASAP.

Predict the expression of *unheard* notes from a small observed excerpt.  Unlike the
Phase-1 benchmark (random 40% held out, so every gap has observed neighbours =
interpolation), completion uses structured masks:

    prefix  — the performer played the opening ``observed_frac`` of the piece; predict
              the rest (pure extrapolation; the rendering use-case: adapt to an excerpt).
    block   — one contiguous held-out span at a random position (gap extrapolation).
    random  — the Phase-1 mask, as the reference point.

Cells are the Phase-1 grid: mean in {zero, ridge, LM} x graph residual in {off, on}, with
recovery (RMSE) and calibration (NLL / coverage / cal-err) per configuration.  Runs from
the named array cache (numpy-only; no torch needed):

    python scripts/eval_asap_completion.py --arrays-cache .cache/asap_arrays_named.pkl
"""
from __future__ import annotations

import argparse

import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.downstream import block_mask, load_piece_arrays, piece_score, prefix_mask
from score_bundle.lm import features as lmfeat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--kinds", nargs="+", default=["prefix", "block", "random"])
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--embeddings", default="emb_scoreonly", choices=["emb", "emb_scoreonly"])
    ap.add_argument("--noise-floor-frac", type=float, default=0.05,
                    help="EB noise_var floor (fraction of observed residual variance); "
                         "0 disables — expect degenerate overconfident fits on some cells")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    print(f"{len(head)} head + {len(ev)} eval pieces | embeddings={args.embeddings} "
          f"| noise_floor_frac={args.noise_floor_frac}")

    H = np.concatenate([p[args.embeddings] for p in head])
    W = lmfeat.fit_prior_mean_head(H, np.concatenate([p["y"] for p in head]), l2=args.l2)

    channels = ["tau", "log r", "v"]
    for kind in args.kinds:
        for frac in args.fracs:
            rng = np.random.default_rng(args.seed)
            acc = ie.MetricAccumulator()
            for p in ev:
                score = piece_score(p)
                y = p["y"]
                n = len(y)
                if kind == "prefix":
                    mask = prefix_mask(n, frac)
                elif kind == "block":
                    mask = block_mask(n, rng, observed_frac=frac)
                else:
                    mask = ie.random_mask(n, rng, observed_frac=frac)
                mu_lm = lmfeat.apply_prior_mean(p[args.embeddings], W)
                means = {
                    "zero": np.zeros_like(y),
                    "ridge": ie.ridge_mean(score, y, mask),
                    "LM": mu_lm,
                }
                cells = ie.impute_methods(score, y, means, mask, fit_hyper=True,
                                          rng=rng, noise_floor_frac=args.noise_floor_frac)
                acc.add(cells)
            print(f"\n=== mask={kind}  observed_frac={frac} "
                  f"({len(ev)} pieces, pooled over channels) ===")
            print(ie.format_report(acc.report(level=0.9)))
    print("\nReading: lower RMSE/NLL/cal-err better; coverage closer to 0.90 better.")
    print("prefix/block are extrapolation; random (the Phase-1 setting) is interpolation.")


if __name__ == "__main__":
    main()
