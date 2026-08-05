#!/usr/bin/env python
"""Report: harmonic learned graph vs plain graph under the final model, by rate.

Pairs c_harm_lm (results/graphgp_charmlm_obsX/) against the mask-sweep
b_featlm cells (results/graphgp_masksweep/obsX/) at every completed rate,
plus the 40%-hidden anchor tie from the adoption record. Tail cells are
reported by per-piece counts and medians, never averaged away.

    PYTHONPATH=src python scripts/report_charmlm_rates.py | tee logs/charmlm_rates_report.log
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import bootstrap_ci  # noqa: E402
from report_mask_sweep import per_piece, pooled  # noqa: E402
from report_theoryfeat_rates import load_cells  # noqa: E402

SWEEP_ROOT = "results/graphgp_masksweep"
TAGS = ["obs0.50", "obs0.70", "obs0.80", "obs0.90"]


def main() -> None:
    rng = np.random.default_rng(19)
    print("== c_harm_lm vs b_featlm by masking rate (DEV, 30 x 4 seeds, guard on) ==\n")
    print(f"{'hidden':>7} {'system':<12} {'RMSE':>8} {'NLL':>8} {'cov@.9':>7}")
    rows = {}
    for tag in TAGS:
        hidden = 1.0 - float(tag[3:7])
        harm = load_cells(f"results/graphgp_charmlm_{tag}", "", "c_harm_lm")
        base = load_cells(SWEEP_ROOT, tag, "b_featlm")
        if not harm or not base:
            print(f"{hidden:7.0%} MISSING ({len(harm)}/{len(base)} cells)")
            continue
        rows[tag] = (harm, base)
        for name, cells in (("c_harm_lm", harm), ("b_featlm", base)):
            m = pooled(cells)
            print(f"{hidden:7.0%} {name:<12} {m['rmse']:8.4f} {m['nll']:8.3f} "
                  f"{m['coverage@0.90']:7.3f}")
        print()

    print("== Paired per-piece contrasts, c_harm_lm - b_featlm "
          "(bootstrap 95% CI, 30 dev pieces) ==\n")
    print(f"{'hidden':>7} {'dRMSE':>28} {'dNLL':>28} {'nll<0':>6} {'med dNLL':>9}")
    for tag in TAGS:
        if tag not in rows:
            continue
        hidden = 1.0 - float(tag[3:7])
        harm, base = rows[tag]
        cols = []
        extra = ""
        for f in ("rmse", "nll"):
            a, b = per_piece(harm, f), per_piece(base, f)
            common = sorted(set(a) & set(b))
            d = np.array([a[pi] - b[pi] for pi in common])
            mu, lo, hi = bootstrap_ci(d, B=2000, rng=rng)
            sig = "*" if (lo > 0) or (hi < 0) else " "
            cols.append(f"{mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}")
            if f == "nll":
                extra = f"{(d < 0).sum():>4}/{len(d)} {np.median(d):+9.4f}"
        print(f"{hidden:7.0%} {cols[0]:>28} {cols[1]:>28} {extra}")
    print("\nAnchor (40% hidden) record: c_harm_lm 0.3561 ties b_featlm 0.3590 "
          "(adoption record, logs/graphgp_final_report.log).")


if __name__ == "__main__":
    main()
