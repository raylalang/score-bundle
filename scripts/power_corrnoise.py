#!/usr/bin/env python
"""Power check for a correlated-timing-noise registration (DEV ONLY).

The tonal-draft method (docs/phase2_tonal_prereg_DRAFT.md), applied to the
correlated-tau result: subsample the existing development cells
(results/corrnoise_tau/cells.shard*of6.pkl, 149 (track, seed) cells over
URMP pieces) at the PIECE level, and count how often the registered star
criterion (95% cluster-bootstrap CI over pieces excluding zero) fires at
each candidate pool size, for the tau NLL and tau RMSE contrasts.  No new
data, no refits; deterministic (seeded).

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/power_corrnoise.py
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIZES = (4, 6, 8, 10, 13, 15, 20)
R = 400          # subsample draws per size
B = 1000         # bootstrap replicates per draw


def main() -> None:
    recs = []
    for f in sorted(glob.glob("results/corrnoise_tau/cells.shard*of6.pkl")):
        recs += pickle.load(open(f, "rb"))
    by_piece = {}
    for r in recs:
        by_piece.setdefault(r["key"][0], []).append(r)
    pieces = sorted(by_piece)
    print(f"{len(recs)} cells over {len(pieces)} URMP pieces "
          f"(median {int(np.median([len(v) for v in by_piece.values()]))} "
          f"cells/piece)\n")

    def star(chosen, met, rng):
        d = [r["corr"][met] - r["base"][met]
             for pc in chosen for r in by_piece[pc]]
        d = np.asarray(d)
        per = [np.asarray([r["corr"][met] - r["base"][met]
                           for r in by_piece[pc]]) for pc in chosen]
        means = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(per), len(per))
            means[b] = np.concatenate([per[j] for j in pick]).mean()
        lo, hi = np.percentile(means, [2.5, 97.5])
        return hi < 0.0 or lo > 0.0

    rng = np.random.default_rng(0)
    print(f"{'pieces':>7s} {'P(tau NLL stars)':>17s} {'P(tau RMSE stars)':>18s}")
    for n in SIZES:
        hits = {"nll": 0, "rmse": 0}
        for _ in range(R):
            chosen = rng.choice(pieces, size=n, replace=False)
            for met in ("nll", "rmse"):
                hits[met] += star(chosen, met, rng)
        print(f"{n:7d} {hits['nll'] / R:17.2f} {hits['rmse'] / R:18.2f}")
    print("\nCaveat (as in the tonal draft): dev pieces here carry a "
          "median 4 (track, seed) cells each (~2 tracks x 2 seeds); a "
          "candidate corpus with more cells per piece sits above its "
          "row (Bach10: 4 stems x 2 seeds = 8), fewer sits below.")


if __name__ == "__main__":
    main()
