#!/usr/bin/env python
"""Per-channel breakdown + per-channel variance rescaling for the corrected headline.

Runs the Phase-1 imputation grid from the named array cache (score-only embeddings,
optional EB noise floor) and reports the per-channel table.  With ``--var-rescale``,
a per-(cell, channel) std scale factor is fit on the *head* pieces (the split already
used to fit the LM head — never the eval notes) via the conformal-style
``metrics.std_rescale_factor``, and the eval metrics are reported before/after scaling.

    python scripts/eval_asap_channelwise.py --noise-floor-frac 0.05 --var-rescale
"""
from __future__ import annotations

import argparse

import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.downstream import load_piece_arrays, piece_score
from score_bundle.lm import features as lmfeat
from score_bundle.metrics import std_rescale_factor


def run_split(pieces, W, ekey, args, seeds, acc=None, collect=None):
    """Run the imputation grid over ``pieces``; fill a MetricAccumulator and/or raw pools."""
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        for p in pieces:
            score = piece_score(p)
            y = p["y"]
            mask = ie.random_mask(len(y), rng, observed_frac=args.observed_frac)
            mu_lm = lmfeat.apply_prior_mean(p[ekey], W)
            means = {"zero": np.zeros_like(y), "ridge": ie.ridge_mean(score, y, mask),
                     "LM": mu_lm}
            cells = ie.impute_methods(score, y, means, mask, fit_hyper=True, rng=rng,
                                      noise_floor_frac=args.noise_floor_frac)
            if acc is not None:
                acc.add(cells)
            if collect is not None:
                for key, cell in cells.items():
                    buf = collect.setdefault(key, [[], [], [], []])
                    buf[0].append(cell.y); buf[1].append(cell.pred)
                    buf[2].append(cell.std); buf[3].append(cell.channel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--observed-frac", type=float, default=0.6)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--embeddings", default="emb_scoreonly", choices=["emb", "emb_scoreonly"])
    ap.add_argument("--noise-floor-frac", type=float, default=0.0)
    ap.add_argument("--var-rescale", action="store_true",
                    help="fit per-(cell,channel) std scales on head pieces, apply to eval")
    args = ap.parse_args()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    channels = ["tau", "log r", "v"]
    print(f"{len(head)} head + {len(ev)} eval pieces | embeddings={args.embeddings} "
          f"| noise_floor_frac={args.noise_floor_frac} | seeds={args.seeds}")

    H = np.concatenate([p[args.embeddings] for p in head])
    W = lmfeat.fit_prior_mean_head(H, np.concatenate([p["y"] for p in head]), l2=args.l2)

    # --- eval pieces: per-channel table --------------------------------------
    acc = ie.MetricAccumulator()
    eval_pool = {}
    run_split(ev, W, args.embeddings, args, args.seeds, acc=acc, collect=eval_pool)
    print("\nPooled:")
    print(ie.format_report(acc.report(level=0.9)))
    print("\nPer-channel breakdown:")
    print(ie.format_report_by_channel(acc.report_by_channel(channels, level=0.9), channels))

    if not args.var_rescale:
        return

    # --- head pieces (1 seed): fit per-(cell, channel) std scales ------------
    head_pool = {}
    run_split(head, W, args.embeddings, args, 1, collect=head_pool)
    scales = {}
    for key, buf in head_pool.items():
        y = np.concatenate(buf[0]); pr = np.concatenate(buf[1])
        sd = np.concatenate(buf[2]); ch = np.concatenate(buf[3])
        for ci in range(3):
            sel = ch == ci
            scales[(key, ci)] = std_rescale_factor(y[sel], pr[sel], sd[sel], level=0.9)
    print("\nPer-(cell, channel) std scales fit on head pieces:")
    for (key, ci), s in sorted(scales.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        print(f"  {str(key):24s} {channels[ci]:6s} s={s:.3f}")

    # --- apply to eval predictions and re-report ------------------------------
    acc2 = ie.MetricAccumulator()
    for key, buf in eval_pool.items():
        y = np.concatenate(buf[0]); pr = np.concatenate(buf[1])
        sd = np.concatenate(buf[2]).copy(); ch = np.concatenate(buf[3])
        for ci in range(3):
            sel = ch == ci
            sd[sel] = sd[sel] * scales[(key, ci)]
        acc2.add({key: ie.CellResult(y, pr, sd, ch)})
    print("\nAfter per-channel variance rescaling (fit on head split):")
    print(ie.format_report(acc2.report(level=0.9)))
    print("\nPer-channel breakdown (rescaled):")
    print(ie.format_report_by_channel(acc2.report_by_channel(channels, level=0.9), channels))


if __name__ == "__main__":
    main()
