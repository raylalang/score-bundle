#!/usr/bin/env python
"""Collapse-exclusion companion to report_charmlm_rates.py (obs 0.50 row).

The 50%-hidden c_harm_lm vs b_featlm contrast is dominated by one collapsed
cell (piece 18, seed 2; RMSE 5e4 — see docs/kernel_multirate_results.md).
The headline row in that doc excludes the collapse; this script IS that
computation, at both exclusion granularities, so the doc row has a logged,
re-runnable provenance instead of an ad-hoc recompute:

  - drop the whole PIECE (n=29): the doc's row;
  - drop only the CELL (piece 18 seed 2; n=30 pieces, 119 cells): the
    sensitivity — the dNLL verdict flips to a small significant harm here,
    which the doc reports alongside the piece-level row.

    PYTHONPATH=src:scripts python scripts/report_charmlm_obs050_excl.py \
        | tee logs/charmlm_obs050_excl_report.log
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import bootstrap_ci  # noqa: E402
from report_charmlm_rates import SWEEP_ROOT  # noqa: E402
from report_mask_sweep import per_piece  # noqa: E402
from report_theoryfeat_rates import load_cells  # noqa: E402

COLLAPSE_PIECE, COLLAPSE_SEED = 18, 2


def paired(harm_cells, base_cells, label):
    print(f"-- {label} --")
    for f in ("rmse", "nll"):
        a, b = per_piece(harm_cells, f), per_piece(base_cells, f)
        common = sorted(set(a) & set(b))
        d = np.array([a[p] - b[p] for p in common])
        rng = np.random.default_rng(19)
        mu, lo, hi = bootstrap_ci(d, B=2000, rng=rng)
        sig = "*" if (lo > 0) or (hi < 0) else " "
        print(f"  d{f.upper():<5} {mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig} "
              f"n={len(d)}  median {np.median(d):+.4f}  <0: {(d < 0).sum()}/{len(d)}")
    print()


def main() -> None:
    harm = load_cells("results/graphgp_charmlm_obs0.50", "", "c_harm_lm")
    base = load_cells(SWEEP_ROOT, "obs0.50", "b_featlm")
    rp = per_piece(harm, "rmse")
    worst = max(rp, key=lambda p: rp[p])
    print(f"collapse cell: piece {COLLAPSE_PIECE} seed {COLLAPSE_SEED} "
          f"(worst per-piece RMSE: piece {worst}, {rp[worst]:.1f})")
    assert worst == COLLAPSE_PIECE, "collapse piece moved — re-inspect cells"
    print()

    def drop_piece(cells):
        return {k: v for k, v in cells.items() if k[1] != COLLAPSE_PIECE}

    def drop_cell(cells):
        return {k: v for k, v in cells.items()
                if not (k[1] == COLLAPSE_PIECE and k[2] == COLLAPSE_SEED)}

    paired(drop_piece(harm), drop_piece(base),
           "excluding the collapse PIECE entirely (doc row, n=29)")
    paired(drop_cell(harm), drop_cell(base),
           "excluding only the collapse CELL (sensitivity, n=30)")


if __name__ == "__main__":
    main()
