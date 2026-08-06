#!/usr/bin/env python
"""Tracker-calibration study (URMP DEVELOPMENT side only).

For every development track: run extract_f0 (pyin) on the separate-track
audio, align to URMP's ground-truth F0 grid (both 10 ms hops), and measure
on co-voiced frames: cents deviation (median |dev|, 90th pct, gross-error
rate > 50 cents) and voicing recall.  Then the question the Phase-2
observation noise hinges on: is pyin's per-frame voicing probability
informative about |error| (confidence calibration)?  Confirmation pieces are
refused by construction.

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/eval_tracker_calibration.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/home/ray/Research/data/urmp/Dataset"
OUT = "results/tracker_calibration_dev.md"

# pyin range per instrument (generous, instrument-appropriate)
RANGES = {"vn": (180, 3000), "va": (120, 1500), "vc": (60, 1000),
          "db": (35, 500), "fl": (240, 2500), "ob": (220, 1800),
          "cl": (140, 1800), "bn": (55, 700), "sax": (100, 1200),
          "tpt": (160, 1200), "hn": (85, 800), "tbn": (70, 700),
          "tba": (40, 400)}


def track_stats(track):
    import soundfile as sf

    from score_bundle.phase2.intonation import extract_f0
    from score_bundle.phase2.urmp import read_f0_annotation

    t_gt, f0_gt = read_f0_annotation(track.f0s)
    audio, sr = sf.read(track.audio)
    fmin, fmax = RANGES[track.instrument]
    out = extract_f0(np.asarray(audio, dtype=float), sr,
                     fmin=float(fmin), fmax=float(fmax))
    f0_py = np.interp(t_gt, out["t"], np.where(np.isnan(out["f0"]), 0.0,
                                               out["f0"]))
    v_py = np.interp(t_gt, out["t"], out["voiced"].astype(float)) > 0.5
    prob = np.interp(t_gt, out["t"], np.nan_to_num(out["prob"]))
    voiced_gt = f0_gt > 0
    both = voiced_gt & v_py & (f0_py > 0)
    if both.sum() < 50:
        return None
    cents = 1200.0 * np.log2(f0_py[both] / f0_gt[both])
    return {"cents": cents, "prob": prob[both],
            "recall": float((v_py & voiced_gt).sum() / max(voiced_gt.sum(), 1)),
            "n": int(both.sum())}


def main() -> None:
    from score_bundle.phase2.splits import CONF_PIECES, urmp_split
    from score_bundle.phase2.urmp import load_urmp_meta

    dev, conf = urmp_split(load_urmp_meta(ROOT))
    assert all(p.index not in CONF_PIECES for p in dev)
    print(f"development: {len(dev)} pieces; confirmation untouched "
          f"({len(conf)} pieces refused)", flush=True)

    per_inst: dict = {}
    all_cents, all_prob = [], []
    for p in dev:
        for tr in p.tracks:
            if not (tr.audio and tr.f0s):
                continue
            s = track_stats(tr)
            if s is None:
                continue
            d = per_inst.setdefault(tr.instrument,
                                    {"cents": [], "recall": [], "n": 0})
            d["cents"].append(s["cents"])
            d["recall"].append(s["recall"])
            d["n"] += s["n"]
            all_cents.append(s["cents"])
            all_prob.append(s["prob"])
            print(f"  {p.index:02d} {p.name:<12} tr{tr.number} "
                  f"{tr.instrument:<4} med|dev| "
                  f"{np.median(np.abs(s['cents'])):5.1f}c  "
                  f"recall {s['recall']:.2f}  n={s['n']}", flush=True)

    lines = ["# Tracker calibration — URMP development side (pyin vs GT F0)",
             "",
             "| instrument | tracks | med \\|dev\\| (c) | 90th pct | >50c | voicing recall |",
             "|---|---|---|---|---|---|"]
    for inst in sorted(per_inst):
        d = per_inst[inst]
        c = np.concatenate(d["cents"])
        lines.append(
            f"| {inst} | {len(d['cents'])} | {np.median(np.abs(c)):.1f} "
            f"| {np.percentile(np.abs(c), 90):.1f} "
            f"| {(np.abs(c) > 50).mean():.1%} "
            f"| {np.mean(d['recall']):.2f} |")

    # confidence calibration: |error| by pyin voicing-probability quintile
    c = np.abs(np.concatenate(all_cents))
    q = np.concatenate(all_prob)
    edges = np.quantile(q, [0, .2, .4, .6, .8, 1.0])
    lines += ["", "## Does pyin confidence predict error?",
              "", "| prob quintile | med \\|dev\\| (c) | >50c |", "|---|---|---|"]
    meds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q >= lo) & (q <= hi)
        meds.append(np.median(c[m]))
        lines.append(f"| [{lo:.2f}, {hi:.2f}] | {meds[-1]:.1f} "
                     f"| {(c[m] > 50).mean():.1%} |")
    mono = all(a >= b for a, b in zip(meds[:-1], meds[1:]))
    lines += ["", f"Median error monotonically decreasing with confidence: "
                  f"**{mono}**; Spearman(prob, |dev|) = "
                  f"{_spearman(q, c):+.3f}."]
    table = "\n".join(lines)
    with open(OUT, "w") as fh:
        fh.write(table + "\n")
    print("\n" + table)
    print(f"\nwrote {OUT}")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


if __name__ == "__main__":
    main()
