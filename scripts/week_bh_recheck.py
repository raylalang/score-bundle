#!/usr/bin/env python
"""Statistical hardening of the exploration week's contrasts (DEV ONLY).

Benjamini-Hochberg across every starred (or adverse-starred) contrast the
2026-09 exploration week produced, plus the robustness battery
(percentile/basic/BCa CIs, Wilcoxon, sign test) per contrast --- the
robustness_recheck.py treatment applied to the week's record-style
pickles.  Everything offline from committed run artifacts; no refits; the
confirmation sets are never read.

Per-piece deltas: URMP records are grouped by piece index (key[0], the
composition-level cluster of splits.py); ASAP records by piece index.
The overall cell-level dmean is printed first and must reproduce the
shipped reports (sanity pin) before any correction is applied.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/week_bh_recheck.py | tee logs/week_bh_recheck.log
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))

from robustness_recheck import boot_pvalue, full_battery  # noqa: E402


def _records(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += pickle.load(open(f, "rb"))
    return rows


def _per_group(deltas, groups):
    """(cell deltas, group labels) -> per-group mean deltas (1-D array)."""
    out = {}
    for d, g in zip(deltas, groups):
        out.setdefault(g, []).append(d)
    return np.array([np.mean(v) for g, v in sorted(out.items())])


def corrnoise_contrasts():
    """(name, cell_dmean, per-piece deltas) for the three corrnoise runs."""
    out = []
    for tag, label in (("", "corrnoise s01"), ("_s23", "corrnoise s23"),
                       ("_joint", "corrnoise joint")):
        recs = _records(f"results/corrnoise_tau/cells{tag}.shard*.pkl")
        if not recs:
            continue
        for met in ("nll", "rmse"):
            d = np.array([r["corr"][met] - r["base"][met] for r in recs])
            g = [r["key"][0] for r in recs]
            out.append((f"{label} tau {met}", float(d.mean()),
                        _per_group(d, g)))
    return out


def tem_contrasts():
    recs = _records("results/t_noise_em/cells.shard*.pkl")
    out = []
    if not recs:
        return out
    NU = 5.0
    d_t = np.array([r["tem"][0]["tnll"] - r["gauss"][0]["tnll"]
                    for r in recs])
    d_r = np.array([r["tem"][1]["rmse"] - r["gauss"][1]["rmse"]
                    for r in recs])
    g = [r["pi"] for r in recs]
    out.append(("t-EM tau t-NLL", float(d_t.mean()), _per_group(d_t, g)))
    out.append(("t-EM log-r RMSE", float(d_r.mean()), _per_group(d_r, g)))
    return out


def bump_contrasts():
    """Study E raw cells vs the published b_feat cells (recomputed scores)."""
    bump = {}
    for f in sorted(glob.glob("results/spectral_bump/cells.shard*.pkl")):
        bump.update(pickle.load(open(f, "rb")))
    base = {}
    for f in sorted(glob.glob("results/graphgp_v2/b_feat.shard*.pkl")):
        base.update(pickle.load(open(f, "rb"))["cells"])
    common = sorted(set(bump) & set(base))
    if not common:
        return []

    def scores(cell):
        yt, pr, sd, ch = cell
        z = (yt - pr) / sd
        return (float(np.sqrt(np.mean((yt - pr) ** 2))),
                float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2)
                              + 0.5 * z ** 2)))

    d_r, d_n, g = [], [], []
    for k in common:
        r1, n1 = scores(bump[k]["cell"])
        r0, n0 = scores(base[k])
        d_r.append(r1 - r0)
        d_n.append(n1 - n0)
        g.append(k[1])
    d_r, d_n = np.array(d_r), np.array(d_n)
    return [("bump-additive RMSE", float(d_r.mean()), _per_group(d_r, g)),
            ("bump-additive NLL", float(d_n.mean()), _per_group(d_n, g))]


def smprior_contrast():
    rows = _records("results/phase3_smprior/cells.shard*.pkl")
    if not rows:
        return []
    d = np.array([abs(r["sm"][0] - r["gt_c"]) - abs(r["bump"][0] - r["gt_c"])
                  for r in rows])
    g = [r["key"][0] for r in rows]
    return [("smprior |err| (Slot B)", float(d.mean()), _per_group(d, g))]


def sm_estimator_contrasts():
    rows = _records(".cache/sm_dev_shard_*_4.pkl")
    both = [r for r in rows if r.get("ident_tr") and r.get("ident_gt")
            and "sm_tr" in r and "sm_gt" in r
            and np.isfinite(r["nl_tr"]["lf"]) and np.isfinite(r["nl_gt"]["lf"])]
    out = []
    if not both:
        return out
    for q, label in (("c", "SM-est c |err|"), ("lg", "SM-est log-gamma |err|"),
                     ("lf", "SM-est log-f |err|")):
        d = np.array([abs(r["sm_tr"][q] - r["sm_gt"][q])
                      - abs(r["nl_tr"][q] - r["nl_gt"][q]) for r in both])
        g = [r["key"][0] for r in both]
        out.append((label, float(d.mean()), _per_group(d, g)))
    return out


def main() -> None:
    family = (corrnoise_contrasts() + tem_contrasts() + bump_contrasts()
              + smprior_contrast() + sm_estimator_contrasts())
    print(f"family of {len(family)} contrasts\n")

    print("== sanity pins (cell-level dmeans must match shipped reports) ==")
    for name, cell_dmean, _ in family:
        if name == "corrnoise s01 tau nll":
            assert abs(cell_dmean - (-0.1108)) < 5e-4, cell_dmean
        if name == "corrnoise s01 tau rmse":
            assert abs(cell_dmean - (-0.0192)) < 5e-4, cell_dmean
    print("corrnoise s01 pins OK\n")

    print("== per-contrast battery (per-piece deltas; URMP clustered at "
          "composition level by construction) ==")
    for name, cell_dmean, d in family:
        print(f"[cell dmean {cell_dmean:+.4f}; n_pieces {d.size}]")
        full_battery(name, d)

    print("\n== Benjamini-Hochberg across the week (q = 0.05, bootstrap "
          "p-values on per-piece deltas) ==")
    ps = []
    for name, _, d in family:
        p = boot_pvalue(d)
        ps.append((p, name, float(d.mean())))
    ps.sort()
    m = len(ps)
    thresh = 0.0
    for i, (p, _, _) in enumerate(ps, start=1):
        if p <= 0.05 * i / m:
            thresh = p
    for p, name, dm in ps:
        verdict = "SURVIVES" if p <= thresh else "n.s. after BH"
        print(f"  p={p:8.5f}  dmean {dm:+8.4f}  {name:26s} {verdict}")


if __name__ == "__main__":
    main()
