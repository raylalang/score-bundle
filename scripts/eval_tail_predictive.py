#!/usr/bin/env python
"""Deploy-time heavy-tailed timing predictive: defusing the Gaussian tail (DEV).

The one measured Phase-1 failure mode (draft sec:guard-gp): a very steady
piece gives timing a tiny scale, a handful of held-out outliers land many
sigma out, and the Gaussian log score's quadratic tail amplifies them into
the pooled NLL (dev reproduction: piece 28, seed 2 at 30% hidden, tau NLL
+73). Future Work names two remedies; this study measures the cheap one,
entirely offline from the published per-note predictions -- the fits, the
posterior means, and the RMSE are untouched by construction:

  gauss        the published Gaussian predictive (baseline)
  t3/t5/t10    Student-t predictive on tau, variance-matched
               (scale^2 = s^2 (nu-2)/nu), Gaussian elsewhere
  floor50/100  predictive-std floor on tau: s' = max(s, beta * median s
               of the same cell), beta = 0.5 / 1.0 -- deploy-legal (uses
               only the model's own outputs)
  t5+floor50   the combination

Scored on the proposed model's cells (b_featlm): the published dev run
(results/graphgp_v2, 40% hidden) and the masking-level sweep
(results/graphgp_masksweep, 50/30/20/10% hidden).  Metrics per variant,
tau channel: pooled mean cell NLL, median cell NLL, worst cell, coverage
at the variant's own nominal 90% interval, PIT-KS.  Development only,
exploratory, no claims.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_tail_predictive.py
"""
from __future__ import annotations

import glob
import pickle

import numpy as np

try:
    from scipy import stats
except ImportError as exc:  # script-level dependency, core untouched
    raise SystemExit("this study script needs scipy (t CDF/quantiles)") from exc

Z90 = 1.6448536269514722
SETS = [
    ("40% hidden (published dev)", "results/graphgp_v2/b_featlm.shard*.pkl"),
    ("50% hidden", "results/graphgp_masksweep/obs0.50/b_featlm.shard*.pkl"),
    ("30% hidden", "results/graphgp_masksweep/obs0.70/b_featlm.shard*.pkl"),
    ("20% hidden", "results/graphgp_masksweep/obs0.80/b_featlm.shard*.pkl"),
    ("10% hidden", "results/graphgp_masksweep/obs0.90/b_featlm.shard*.pkl"),
]


def load_cells(pattern):
    cells = {}
    for f in sorted(glob.glob(pattern)):
        cells.update(pickle.load(open(f, "rb"))["cells"])
    return cells


def variants(y, m, s):
    """Per-note tau scores under each predictive.  Returns
    {name: (nll_array, hit_array, pit_array)}."""
    out = {}
    z = (y - m) / s
    out["gauss"] = (0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * z ** 2,
                    np.abs(z) <= Z90, stats.norm.cdf(z))
    for nu in (3, 5, 10):
        sc = s * np.sqrt((nu - 2) / nu)          # variance-matched t scale
        zt = (y - m) / sc
        q = stats.t.ppf(0.95, nu)
        out[f"t{nu}"] = (-stats.t.logpdf(zt, nu) + np.log(sc),
                         np.abs(zt) <= q, stats.t.cdf(zt, nu))
    for beta, name in ((0.5, "floor50"), (1.0, "floor100")):
        sf = np.maximum(s, beta * np.median(s))
        zf = (y - m) / sf
        out[name] = (0.5 * np.log(2 * np.pi * sf ** 2) + 0.5 * zf ** 2,
                     np.abs(zf) <= Z90, stats.norm.cdf(zf))
    sf = np.maximum(s, 0.5 * np.median(s))
    sc = sf * np.sqrt(3 / 5)
    zt = (y - m) / sc
    q = stats.t.ppf(0.95, 5)
    out["t5+floor50"] = (-stats.t.logpdf(zt, 5) + np.log(sc),
                         np.abs(zt) <= q, stats.t.cdf(zt, 5))
    return out


def main() -> None:
    names = None
    for label, pattern in SETS:
        cells = load_cells(pattern)
        if not cells:
            print(f"--- {label}: no files ({pattern}), skipped")
            continue
        per = {}
        for key, (yt, pr, sd, ch) in sorted(cells.items()):
            tau = ch == 0
            v = variants(yt[tau], pr[tau], sd[tau])
            names = list(v)
            for name, (nll, hit, pit) in v.items():
                per.setdefault(name, []).append(
                    (float(nll.mean()), hit, pit))
        print(f"\n=== {label} ({len(cells)} cells, tau channel) ===")
        print(f"{'variant':12s} {'pooled':>8s} {'median':>8s} {'worst':>8s} "
              f"{'cov@90':>7s} {'PIT-KS':>7s}")
        for name in names:
            rows = per[name]
            nlls = np.array([r[0] for r in rows])
            hits = np.concatenate([r[1] for r in rows])
            pits = np.sort(np.concatenate([r[2] for r in rows]))
            ks = float(np.max(np.abs(pits - (np.arange(pits.size) + 0.5)
                                     / pits.size)))
            print(f"{name:12s} {nlls.mean():8.3f} {np.median(nlls):8.3f} "
                  f"{nlls.max():8.2f} {hits.mean():7.3f} {ks:7.3f}")

    # the documented blow-up cell, before and after
    cells = load_cells(SETS[2][1])
    if cells:
        key = ("GP", 28, 2)
        if key in cells:
            yt, pr, sd, ch = cells[key]
            tau = ch == 0
            v = variants(yt[tau], pr[tau], sd[tau])
            print(f"\n=== the documented tail cell {key} (30% hidden) ===")
            for name, (nll, hit, _) in v.items():
                print(f"{name:12s} cell NLL {nll.mean():8.3f}  "
                      f"cov {hit.mean():.2f}")


if __name__ == "__main__":
    main()
