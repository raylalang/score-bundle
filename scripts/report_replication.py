#!/usr/bin/env python
"""Report for the post-hoc REPLICATION set (30 fresh pieces, positions 50-79).

Fresh pieces never touched by any selection decision (identity gate: positions
0-49 of the extended cache are byte-identical to the published cache, so the
dev and confirmation sets are untouched). NOT confirmation-grade — no
preregistration — but an independent out-of-selection replication of the dev
ladder: does the ordering hold, and do the paired ingredient contributions
stay significant, on pieces nobody ever looked at?

    PYTHONPATH=src:scripts python scripts/report_replication.py \
        | tee logs/replication_report.log
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import bootstrap_ci  # noqa: E402
from report_mask_sweep import per_piece, pooled  # noqa: E402

ROOT = "results/graphgp_repl"
CONFIGS = ["b_feat", "b_featlm", "b_featlm_nograph"]
DEV = {  # published dev values for side-by-side (logs/graphgp_v2_report.log)
    "b_feat": (0.3683, -0.370), "b_featlm": (0.3601, -0.404),
    "b_featlm_nograph": (0.3755, -0.337)}


def load(config):
    merged = {}
    for path in sorted(glob.glob(os.path.join(ROOT, f"{config}.shard*.pkl"))):
        with open(path, "rb") as fh:
            merged.update(pickle.load(fh)["cells"])
    return merged


def main() -> None:
    cells = {c: load(c) for c in CONFIGS}
    print("== REPLICATION set (30 fresh pieces x 4 seeds, guard on) ==")
    print(f"{'config':<20} {'RMSE':>8} {'NLL':>8} {'cov@.9':>7}   "
          f"(dev value for reference)")
    for c in CONFIGS:
        m = pooled(cells[c])
        print(f"{c:<20} {m['rmse']:8.4f} {m['nll']:8.3f} "
              f"{m['coverage@0.90']:7.3f}   (dev {DEV[c][0]:.4f} / {DEV[c][1]:+.3f})")

    rng = np.random.default_rng(11)
    print("\npaired per-piece contrasts (bootstrap 95% CI over 30 fresh pieces):")
    for name, a, b in [("graph value", "b_featlm", "b_featlm_nograph"),
                       ("embedding value", "b_featlm", "b_feat")]:
        for f in ("rmse", "nll"):
            pa, pb = per_piece(cells[a], f), per_piece(cells[b], f)
            common = sorted(set(pa) & set(pb))
            d = np.array([pa[k] - pb[k] for k in common])
            assert np.isfinite(d).all()
            mu, lo, hi = bootstrap_ci(d, B=2000, rng=rng)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            neg = int((d < 0).sum())
            print(f"  {name:<16} d{f.upper():<5} {mu:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]{sig}  ({neg}/30 pieces negative)")

    order_ok = (pooled(cells['b_featlm'])['rmse']
                < min(pooled(cells['b_feat'])['rmse'],
                      pooled(cells['b_featlm_nograph'])['rmse']))
    print(f"\nordering (proposed < both ablations): "
          f"{'REPLICATES' if order_ok else 'DOES NOT REPLICATE'}")


if __name__ == "__main__":
    main()
