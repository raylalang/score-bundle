#!/usr/bin/env python
"""Power check for the motif-study registration (docs/motif_prereg_design.md).

Per-piece paired differences (seeds averaged) of one contrast are read from a
motif results directory (cells.shard*.pkl of eval_motif_pilot_dev.py), then
pieces are subsampled WITH replacement to n pieces and the 95% percentile
bootstrap (B = 2000, the report's rule) is applied; the table reports how often
the interval excludes zero (a star), at the measured effect and at two thirds
and half of it.

    PYTHONPATH=src:scripts python scripts/power_motif.py results/motif_pilot2 \
        --arm time_motif_pc --scheme occ --channel 0 --metric nll [--reps 1000]
"""
from __future__ import annotations

import argparse
import glob
import pickle

import numpy as np


def per_piece_diffs(out, arm, scheme, channel, metric, ref="time", subset="motif"):
    pieces = pickle.load(open(f"{out}/pieces.pkl", "rb"))
    cells = {}
    for f in glob.glob(f"{out}/cells.shard*.pkl"):
        cells.update(pickle.load(open(f, "rb")))
    seeds = sorted({k[2] for k in cells})

    def score(r):
        yt, pr, sd, ch = r["cell"]
        sel = ch == channel
        if subset == "motif":
            sel &= np.tile(r["motif_sel"], 3)
        if sel.sum() < 3:
            return None
        z = (yt[sel] - pr[sel]) / sd[sel]
        if metric == "nll":
            return float(np.mean(0.5 * z ** 2 + np.log(sd[sel]) + 0.5 * np.log(2 * np.pi)))
        return float(np.sqrt(np.mean((yt[sel] - pr[sel]) ** 2)))

    diffs, names = [], []
    for pi, p in enumerate(pieces):
        a, b = [], []
        for s in seeds:
            ra, rb = cells.get((pi, scheme, s, arm)), cells.get((pi, scheme, s, ref))
            if ra is None or rb is None:
                continue
            va, vb = score(ra), score(rb)
            if va is None or vb is None:
                continue
            a.append(va); b.append(vb)
        if a:
            diffs.append(np.mean(a) - np.mean(b)); names.append(p["folder"])
    return np.asarray(diffs), names


def power_table(d, ns, reps=1000, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        row = [n]
        for shrink in (1.0, 2 / 3, 0.5):
            hits = 0
            for _ in range(reps):
                s = rng.choice(d, n, replace=True) - d.mean() + d.mean() * shrink
                bs = rng.choice(s, (B, n), replace=True).mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                hits += (hi < 0) if d.mean() < 0 else (lo > 0)
            row.append(hits / reps)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arm", default="time_motif_pc")
    ap.add_argument("--ref", default="time")
    ap.add_argument("--scheme", default="occ")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--metric", choices=["nll", "rmse"], default="nll")
    ap.add_argument("--ns", default="10,14,18,21")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    d, names = per_piece_diffs(a.out, a.arm, a.scheme, a.channel, a.metric, a.ref)
    print(f"{a.arm} vs {a.ref}, scheme {a.scheme}, channel {a.channel}, {a.metric}: "
          f"n = {d.size} pieces, mean {d.mean():+.4f}, sd {d.std(ddof=1):.4f}")
    print("| n pieces | P(star) at the measured effect | at two thirds | at half |")
    print("|---|---|---|---|")
    for n, p1, p2, p3 in power_table(d, [int(x) for x in a.ns.split(",")], reps=a.reps):
        print(f"| {n} | {p1:.2f} | {p2:.2f} | {p3:.2f} |")


if __name__ == "__main__":
    main()
