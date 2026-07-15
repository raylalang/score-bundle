#!/usr/bin/env python
"""Report for the masking-level sweep (DEV pieces; results/graphgp_masksweep/).

Aggregates the sweep run by ``scripts/run_mask_sweep.sh``: hidden fraction
50/40/30/20/10% (plus the LOO limit from ``scripts/eval_gp_loo.py``) for
``b_feat`` / ``b_featlm`` / ``b_featlm_nograph``.  For every fraction it prints
pooled metrics (published convention: all held-out notes concatenated) and the
two paired per-piece contrasts with bootstrap 95% CIs:

* LM value    = b_featlm - b_feat            (what the embeddings add)
* graph value = b_featlm - b_featlm_nograph  (what the graph adds)

It also scans every (config, piece, seed) cell for NLL outliers (the known
Gaussian-tail failure mode) so no anomaly is averaged away.

DEV set only — every number here is development-labeled; the confirmation set
is untouched.

    PYTHONPATH=src python scripts/report_mask_sweep.py | tee logs/masksweep_report.log
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import bootstrap_ci  # noqa: E402

ROOT = "results/graphgp_masksweep"
TAGS = ["obs0.50", "obs0.60_anchor", "obs0.70", "obs0.80", "obs0.90"]
SWEEP_CONFIGS = ["b_feat", "b_featlm", "b_featlm_nograph"]
CONTRASTS = [("LM value", "b_featlm", "b_feat"),
             ("graph value", "b_featlm", "b_featlm_nograph")]


def load_cells(tag: str, config: str) -> dict:
    merged = {}
    for path in sorted(glob.glob(os.path.join(ROOT, tag, f"{config}.shard*.pkl"))) \
            + [p for p in [os.path.join(ROOT, tag, f"{config}.pkl")]
               if os.path.exists(p)]:
        with open(path, "rb") as fh:
            merged.update(pickle.load(fh)["cells"])
    return merged


def pooled(cells) -> dict:
    from score_bundle.metrics import evaluate
    yt = np.concatenate([c[0] for c in cells.values()])
    pr = np.concatenate([c[1] for c in cells.values()])
    sd = np.concatenate([c[2] for c in cells.values()])
    return evaluate(yt, pr, sd, level=0.9)


def per_piece(cells, field: str) -> dict:
    from score_bundle.metrics import evaluate
    by: dict = {}
    for (_, pi, s), (yt, pr, sd, ch) in cells.items():
        b = by.setdefault(pi, [[], [], []])
        b[0].append(yt); b[1].append(pr); b[2].append(sd)
    return {pi: evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                         np.concatenate(v[2]), level=0.9)[field]
            for pi, v in by.items()}


def contrast_txt(cells_a, cells_b, rng) -> dict:
    out = {}
    for f in ("rmse", "nll"):
        a, b = per_piece(cells_a, f), per_piece(cells_b, f)
        common = sorted(set(a) & set(b))
        d = np.array([a[pi] - b[pi] for pi in common])
        mu, lo, hi = bootstrap_ci(d, B=2000, rng=rng)
        sig = "*" if (lo > 0) or (hi < 0) else " "
        out[f] = f"{mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}"
    return out


def main() -> None:
    from score_bundle.metrics import evaluate

    rng = np.random.default_rng(11)
    all_cells = {}          # (tag, config) -> cells
    print("== Masking-level sweep (DEV pieces, 30 x 4 seeds, guard on) ==\n")
    print(f"{'hidden':>7} {'config':<20} {'RMSE':>8} {'NLL':>8} {'cov@.9':>7} "
          f"{'cal-err':>8}")
    for tag in TAGS:
        hidden = 1.0 - float(tag[3:7])
        for config in SWEEP_CONFIGS:
            cells = load_cells(tag, config)
            if not cells:
                print(f"{hidden:7.0%} {config:<20} MISSING"); continue
            all_cells[(tag, config)] = cells
            m = pooled(cells)
            print(f"{hidden:7.0%} {config:<20} {m['rmse']:8.4f} {m['nll']:8.3f} "
                  f"{m['coverage@0.90']:7.3f} {m['calibration_error']:8.3f}")
        print()

    print("== Paired per-piece contrasts (bootstrap 95% CI over 30 dev pieces) ==\n")
    print(f"{'hidden':>7} {'contrast':<12} {'dRMSE':>28} {'dNLL':>28}")
    for tag in TAGS:
        hidden = 1.0 - float(tag[3:7])
        for name, ca, cb in CONTRASTS:
            if (tag, ca) not in all_cells or (tag, cb) not in all_cells:
                continue
            t = contrast_txt(all_cells[(tag, ca)], all_cells[(tag, cb)], rng)
            print(f"{hidden:7.0%} {name:<12} {t['rmse']:>28} {t['nll']:>28}")
        print()

    # ---- LOO limit --------------------------------------------------------
    loo_path = os.path.join(ROOT, "loo.pkl")
    if os.path.exists(loo_path):
        with open(loo_path, "rb") as fh:
            loo = pickle.load(fh)["results"]
        print("== LOO limit (hyperparams fit on the full piece; see eval_gp_loo) ==\n")
        print(f"{'config':<20} {'RMSE':>8} {'NLL':>8} {'cov@.9':>7}   (note-pooled)")
        loo_pp = {}
        for config in SWEEP_CONFIGS:
            r = loo.get(config)
            if r is None:
                continue
            yt = np.concatenate([c[0].ravel() for c in r["cells"].values()])
            pr = np.concatenate([c[1].ravel() for c in r["cells"].values()])
            sd = np.sqrt(np.concatenate([np.maximum(c[2], 1e-12).ravel()
                                         for c in r["cells"].values()]))
            m = evaluate(yt, pr, sd, level=0.9)
            loo_pp[config] = {row["piece"]: (row["rmse"], row["nll"])
                              for row in r["per_piece"]}
            print(f"{config:<20} {m['rmse']:8.4f} {m['nll']:8.3f} "
                  f"{m['coverage@0.90']:7.3f}")
        print()
        for name, ca, cb in CONTRASTS:
            if ca not in loo_pp or cb not in loo_pp:
                continue
            common = sorted(set(loo_pp[ca]) & set(loo_pp[cb]))
            for fi, f in enumerate(("rmse", "nll")):
                d = np.array([loo_pp[ca][pi][fi] - loo_pp[cb][pi][fi]
                              for pi in common])
                mu, lo, hi = bootstrap_ci(d, B=2000, rng=rng)
                sig = "*" if (lo > 0) or (hi < 0) else " "
                print(f"    LOO {name:<12} d{f.upper():<5} "
                      f"{mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}")
        print()

    # ---- per-cell NLL outlier scan ----------------------------------------
    print("== Per-cell NLL outlier scan (worst 10 cells across the sweep) ==\n")
    rows = []
    for (tag, config), cells in all_cells.items():
        for (_, pi, s), (yt, pr, sd, ch) in cells.items():
            nll = evaluate(yt, pr, sd, level=0.9)["nll"]
            zmax = float(np.max(np.abs((yt - pr) / sd)))
            rows.append((nll, tag, config, pi, s, zmax))
    rows.sort(reverse=True)
    print(f"{'NLL':>9} {'hidden':>7} {'config':<20} {'piece':>5} {'seed':>4} "
          f"{'max|z|':>8}")
    for nll, tag, config, pi, s, zmax in rows[:10]:
        hidden = 1.0 - float(tag[3:7])
        print(f"{nll:9.3f} {hidden:7.0%} {config:<20} {pi:5d} {s:4d} {zmax:8.1f}")
    med = float(np.median([r[0] for r in rows]))
    print(f"\nmedian cell NLL {med:.3f}; cells above median+2: "
          f"{sum(1 for r in rows if r[0] > med + 2)}/{len(rows)}")


if __name__ == "__main__":
    main()
