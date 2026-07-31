#!/usr/bin/env python
"""Report: music-theory features across masking rates (DEV pieces).

Aggregates the 2026-07-16 run of ``scripts/run_theoryfeat_rates.sh``
(``results/graphgp_theoryfeat_obs{0.50,0.70,0.80,0.90}/``) against the
mask-sweep baseline cells (``results/graphgp_masksweep/obsX/``), completing
the 40%-hidden anchor A/B of ``docs/theory_features_results.md`` at the other
four masking levels.  Contrasts (paired per-piece bootstrap 95% CI):

* theory value (plain)   = b_theoryfeat   - b_feat     (25+14 vs 25 cols)
* theory value (with LM) = b_theoryfeatlm - b_featlm

DEV set only; the confirmation set is untouched.

    PYTHONPATH=src python scripts/report_theoryfeat_rates.py | tee logs/theoryfeat_rates_report.log
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import bootstrap_ci  # noqa: E402
from report_mask_sweep import pooled, per_piece  # noqa: E402

SWEEP_ROOT = "results/graphgp_masksweep"
TAGS = ["obs0.50", "obs0.70", "obs0.80", "obs0.90"]
CONTRASTS = [("theory (plain)", "b_theoryfeat", "b_feat"),
             ("theory (with LM)", "b_theoryfeatlm", "b_featlm")]


def load_cells(root: str, tag: str, config: str) -> dict:
    merged = {}
    for path in sorted(glob.glob(os.path.join(root, tag, f"{config}.shard*.pkl"))):
        with open(path, "rb") as fh:
            merged.update(pickle.load(fh)["cells"])
    return merged


def main() -> None:
    rng = np.random.default_rng(13)
    print("== Theory features across masking rates (DEV, 30 x 4 seeds, guard on) ==\n")
    print(f"{'hidden':>7} {'config':<18} {'RMSE':>8} {'NLL':>8} {'cov@.9':>7}")
    all_cells = {}
    for tag in TAGS:
        hidden = 1.0 - float(tag[3:7])
        for config in ("b_feat", "b_theoryfeat", "b_featlm", "b_theoryfeatlm"):
            root = SWEEP_ROOT if config in ("b_feat", "b_featlm") else \
                f"results/graphgp_theoryfeat_{tag}"
            t = tag if root == SWEEP_ROOT else ""
            cells = load_cells(root, t, config)
            if not cells:
                print(f"{hidden:7.0%} {config:<18} MISSING"); continue
            all_cells[(tag, config)] = cells
            m = pooled(cells)
            print(f"{hidden:7.0%} {config:<18} {m['rmse']:8.4f} {m['nll']:8.3f} "
                  f"{m['coverage@0.90']:7.3f}")
        print()

    print("== Paired per-piece contrasts (bootstrap 95% CI over 30 dev pieces) ==\n")
    print(f"{'hidden':>7} {'contrast':<18} {'dRMSE':>28} {'dNLL':>28}")
    for tag in TAGS:
        hidden = 1.0 - float(tag[3:7])
        for name, ca, cb in CONTRASTS:
            if (tag, ca) not in all_cells or (tag, cb) not in all_cells:
                continue
            row = []
            for f in ("rmse", "nll"):
                a = per_piece(all_cells[(tag, ca)], f)
                b = per_piece(all_cells[(tag, cb)], f)
                common = sorted(set(a) & set(b))
                d = np.array([a[pi] - b[pi] for pi in common])
                mu, lo, hi = bootstrap_ci(d, B=2000, rng=rng)
                sig = "*" if (lo > 0) or (hi < 0) else " "
                row.append(f"{mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}")
            print(f"{hidden:7.0%} {name:<18} {row[0]:>28} {row[1]:>28}")
        print()


if __name__ == "__main__":
    main()
