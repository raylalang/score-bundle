#!/usr/bin/env python
"""Per-piece bootstrap 95% CIs for the six thesis headline-table rows.

The draft's Table `tab:headline-exec` carries point estimates; this script computes
the matching per-piece percentile-bootstrap CIs (RMSE and NLL), all under the strict
protocol with the identical kernel-sweep masks:

    zero mean, graph off        computed here (mean prediction, homoscedastic std —
    LM mean,   graph off        the no-graph branch of imputation_eval._predict_channel)
    zero + additive graph       results/kernels/additive.pkl        (zero block)
    LM   + additive graph       results/kernels/additive.pkl        (LM block)
    feat+LM + additive graph    results/kernels_featlm/additive.pkl (LM block)
    feat+LM + harmonic graph    results/kernels_featlm/harmonic_vl.pkl (LM block)

Pooled point estimates must reproduce the published cells (0.5664 / 0.4041 / 0.3930 /
0.3879 / 0.3795) — that is the gate for trusting the CIs.  NumPy-only.

    python scripts/report_headline_cis.py > logs/headline_cis.log
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


def bootstrap_ci(vals, B=2000, rng=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--inputs-lm", default=".cache/kernel_sweep_inputs.pkl")
    ap.add_argument("--inputs-featlm", default=".cache/kernel_sweep_inputs_featlm.pkl")
    ap.add_argument("--kernels-dir", default="results/kernels")
    ap.add_argument("--featlm-dir", default="results/kernels_featlm")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    from score_bundle.downstream import load_piece_arrays
    from score_bundle.metrics import evaluate

    _, ev, _ = load_piece_arrays(args.arrays_cache)
    with open(args.inputs_lm, "rb") as fh:
        in_lm = pickle.load(fh)
    ev = ev[: in_lm["meta"]["n_eval_pieces"]]
    seeds = in_lm["meta"]["seeds"]

    # ---- graph-off rows: mean prediction, per-channel homoscedastic residual std ----
    def mean_only_cells(mu_of):
        """{(pi, s): (y, pred, std)} exactly as imputation_eval._predict_channel's
        no-graph branch scores a mean: predict the mean at held-out notes with a
        per-channel std from the observed residuals."""
        cells = {}
        for s in range(seeds):
            for pi, p in enumerate(ev):
                y = np.asarray(p["y"], dtype=float)
                mask = in_lm["masks"][(pi, s)]
                held = ~mask
                M = mu_of(pi, s, y)
                yt, pr, sd = [], [], []
                for c in range(y.shape[1]):
                    resid = (y[:, c] - M[:, c])[mask]
                    sc = float(np.std(resid)) if resid.size > 1 else 1.0
                    yt.append(y[held, c]); pr.append(M[held, c])
                    sd.append(np.full(int(held.sum()), max(sc, 1e-6)))
                cells[(pi, s)] = tuple(np.concatenate(a) for a in (yt, pr, sd))
        return cells

    def graph_cells(path, mean_name):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        return {(pi, s): (yt, pr, sd)
                for (mn, pi, s), (yt, pr, sd, ch) in blob["cells"].items()
                if mn == mean_name}

    rows = [
        ("zero mean, graph off",
         mean_only_cells(lambda pi, s, y: np.zeros_like(y))),
        ("zero + graph",
         graph_cells(os.path.join(args.kernels_dir, "additive.pkl"), "zero")),
        ("LM mean, graph off",
         mean_only_cells(lambda pi, s, y: np.asarray(in_lm["mu_lm"][(pi, s)], dtype=float))),
        ("LM + graph",
         graph_cells(os.path.join(args.kernels_dir, "additive.pkl"), "LM")),
        ("feat+LM + graph",
         graph_cells(os.path.join(args.featlm_dir, "additive.pkl"), "LM")),
        ("feat+LM + harmonic graph (headline)",
         graph_cells(os.path.join(args.featlm_dir, "harmonic_vl.pkl"), "LM")),
    ]

    boot_rng = np.random.default_rng(11)
    print(f"Headline-table rows, strict protocol, identical masks "
          f"({len(ev)} pieces x {seeds} seeds); per-piece bootstrap 95% CIs\n")
    print(f"{'system':38s} {'RMSE [95% CI]':>24s} {'NLL [95% CI]':>26s} {'cov@.9':>8s}")
    for name, cells in rows:
        yt = np.concatenate([c[0] for c in cells.values()])
        pr = np.concatenate([c[1] for c in cells.values()])
        sd = np.concatenate([c[2] for c in cells.values()])
        m = evaluate(yt, pr, sd, level=0.9)
        by_piece = {}
        for (pi, s), (a, b, c) in cells.items():
            bp = by_piece.setdefault(pi, [[], [], []])
            bp[0].append(a); bp[1].append(b); bp[2].append(c)
        pp = {f: [evaluate(np.concatenate(v[0]), np.concatenate(v[1]),
                           np.concatenate(v[2]), level=0.9)[f]
                  for v in by_piece.values()]
              for f in ("rmse", "nll")}
        r, rlo, rhi = bootstrap_ci(pp["rmse"], B=args.boot, rng=boot_rng)
        n, nlo, nhi = bootstrap_ci(pp["nll"], B=args.boot, rng=boot_rng)
        print(f"{name:38s} {m['rmse']:7.4f} [{rlo:.3f},{rhi:.3f}] "
              f"{m['nll']:9.4f} [{nlo:.3f},{nhi:.3f}] {m['coverage@0.90']:8.3f}")


if __name__ == "__main__":
    main()
