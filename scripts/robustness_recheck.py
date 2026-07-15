#!/usr/bin/env python
"""Statistical robustness recheck of the DEV-set evidence (presentation prep).

Implements the shoring analyses from the 2026-07-15 statistical review, all from
committed result pickles — no refits, DEV only, the confirmation set is never
read:

  A. per-channel + variance-standardized re-pooling of the dev headline contrast
     (proposed model vs the strongest two-stage configuration);
  B. CI-method invariance for the three key dev contrasts: percentile, basic,
     and BCa bootstrap, Wilcoxon signed-rank, exact sign test, delta skewness;
  C. Benjamini-Hochberg across the starred dev contrasts (kernel-study wins,
     ladder ingredient contrasts, masking-sweep contrasts, overlap trend);
  D. piece-level coverage distribution + bootstrap CI for the proposed model;
  E. composer-clustered bootstrap (resample composers, keep their pieces) for
     the key contrasts;
  F. non-finiteness audit: assert every per-piece value entering any reported
     contrast is finite.

    PYTHONPATH=src:scripts python scripts/robustness_recheck.py \
        | tee logs/robustness_recheck.log
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import bootstrap_ci  # noqa: E402

Z90 = 1.6448536269514722


# ---------------------------------------------------------------- loading
def load_cells(pattern):
    merged = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        cells = blob["cells"]
        merged.update(cells)
    return merged


def load_baseline(path, mean_name):
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    return {("GP", pi, s): v for (mn, pi, s), v in blob["cells"].items()
            if mn == mean_name}


def per_piece_metric(cells, metric, channel=None):
    from score_bundle.metrics import evaluate
    by = {}
    for (_, pi, s), (yt, pr, sd, ch) in cells.items():
        b = by.setdefault(pi, [[], [], [], []])
        b[0].append(yt); b[1].append(pr); b[2].append(sd); b[3].append(ch)
    out = {}
    for pi, v in by.items():
        yt = np.concatenate(v[0]); pr = np.concatenate(v[1])
        sd = np.concatenate(v[2]); ch = np.concatenate(v[3])
        if channel is not None:
            m = ch == channel
            yt, pr, sd = yt[m], pr[m], sd[m]
        out[pi] = evaluate(yt, pr, sd)[metric]
    return out


def deltas(cells_a, cells_b, metric, channel=None):
    pa = per_piece_metric(cells_a, metric, channel)
    pb = per_piece_metric(cells_b, metric, channel)
    common = sorted(set(pa) & set(pb))
    d = np.array([pa[k] - pb[k] for k in common])
    assert np.isfinite(d).all(), "non-finite per-piece delta (audit F fails)"
    return d, common


# ---------------------------------------------------------------- inference
def bca_ci(d, B=4000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(7)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    theta = d.mean()
    z0 = _phi_inv((boots < theta).mean())
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum()) ** 1.5
    a = num / den if den > 0 else 0.0
    lo_q = _phi(z0 + (z0 + _phi_inv(alpha / 2)) / (1 - a * (z0 + _phi_inv(alpha / 2))))
    hi_q = _phi(z0 + (z0 + _phi_inv(1 - alpha / 2)) / (1 - a * (z0 + _phi_inv(1 - alpha / 2))))
    return (float(np.quantile(boots, np.clip(lo_q, 0, 1))),
            float(np.quantile(boots, np.clip(hi_q, 0, 1))))


def _phi(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def _phi_inv(p):
    from scipy.stats import norm
    return float(norm.ppf(np.clip(p, 1e-9, 1 - 1e-9)))


def basic_ci(d, B=4000, rng=None):
    rng = rng or np.random.default_rng(7)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    theta = d.mean()
    lo_p, hi_p = np.percentile(boots, [2.5, 97.5])
    return float(2 * theta - hi_p), float(2 * theta - lo_p)


def boot_pvalue(d, B=20000, rng=None):
    """Two-sided bootstrap p for mean(delta)=0 (shift method)."""
    rng = rng or np.random.default_rng(7)
    n = len(d)
    centered = d - d.mean()
    boots = np.array([centered[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(min(1.0, 2 * min((boots <= -abs(d.mean())).mean()
                                  + 1 / B, (boots >= abs(d.mean())).mean() + 1 / B)))


def full_battery(name, d):
    from scipy.stats import binomtest, skew, wilcoxon
    mu, lo, hi = bootstrap_ci(d, B=2000, rng=np.random.default_rng(11))
    blo, bhi = basic_ci(d)
    clo, chi = bca_ci(d)
    w = wilcoxon(d, alternative="two-sided")
    npos = int((d > 0).sum()); nneg = int((d < 0).sum())
    sign_p = binomtest(min(npos, nneg), npos + nneg, 0.5).pvalue
    print(f"  {name}")
    print(f"    mean {mu:+.4f}  percentile [{lo:+.4f},{hi:+.4f}]  "
          f"basic [{blo:+.4f},{bhi:+.4f}]  BCa [{clo:+.4f},{chi:+.4f}]")
    print(f"    Wilcoxon p={w.pvalue:.4f}  sign test {nneg}/{len(d)} pieces "
          f"negative, p={sign_p:.4f}  skew {skew(d):+.2f}")


def clustered_ci(d, groups, B=4000, rng=None):
    rng = rng or np.random.default_rng(7)
    uniq = sorted(set(groups))
    idx_by = {g: np.flatnonzero(np.array(groups) == g) for g in uniq}
    boots = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[uniq[k]] for k in pick])
        boots.append(d[sel].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- main
def main() -> None:
    from score_bundle.downstream import load_piece_arrays

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    composers = [p.get("composer", "?") for p in ev[:30]]

    V2 = "results/graphgp_v2"
    gp = load_cells(f"{V2}/b_featlm.shard*.pkl") or load_cells(f"{V2}/b_featlm.pkl")
    nograph = load_cells(f"{V2}/b_featlm_nograph.shard*.pkl") \
        or load_cells(f"{V2}/b_featlm_nograph.pkl")
    feat = load_cells(f"{V2}/b_feat.shard*.pkl") or load_cells(f"{V2}/b_feat.pkl")
    oldhead = load_baseline("results/kernels_featlm/harmonic_vl.pkl", "LM")
    names = ["tau", "log r", "v"]

    print("== A. dev headline contrast, per channel + standardized pooling ==")
    d_pool, pieces = deltas(gp, oldhead, "rmse")
    full_battery("pooled (native units): GP - two-stage headline", d_pool)
    for c in range(3):
        dc, _ = deltas(gp, oldhead, "rmse", channel=c)
        mu, lo, hi = bootstrap_ci(dc, B=2000, rng=np.random.default_rng(11))
        sig = "*" if (lo > 0) or (hi < 0) else " "
        print(f"    channel {names[c]:<6} dRMSE {mu:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}")
    # variance-standardized pooling: per-channel deltas scaled by the zero-mean
    # no-graph per-channel RMSE (the natural per-channel scale), then averaged
    scale = {}
    zero = load_cells(f"{V2}/a_diag.shard*.pkl") or load_cells(f"{V2}/a_diag.pkl")
    for c in range(3):
        scale[c] = np.mean(list(per_piece_metric(zero, "rmse", channel=c).values()))
    d_std = np.zeros(len(pieces))
    for c in range(3):
        dc, common_c = deltas(gp, oldhead, "rmse", channel=c)
        assert common_c == pieces
        d_std += dc / scale[c]
    d_std /= 3.0
    full_battery("standardized pooling (per-channel scale = indep. baseline RMSE)",
                 d_std)

    print("\n== B. CI-method invariance, key ingredient contrasts ==")
    d_graph, _ = deltas(gp, nograph, "rmse")
    d_emb, _ = deltas(gp, feat, "rmse")
    full_battery("graph value (dRMSE)", d_graph)
    full_battery("embedding value (dRMSE)", d_emb)
    d_graph_nll, _ = deltas(gp, nograph, "nll")
    full_battery("graph value (dNLL)", d_graph_nll)

    print("\n== C. Benjamini-Hochberg across starred dev contrasts ==")
    family = {}
    family["gp_vs_head_rmse"] = d_pool
    family["graph_rmse"] = d_graph
    family["graph_nll"] = d_graph_nll
    family["emb_rmse"] = d_emb
    family["emb_nll"] = deltas(gp, feat, "nll")[0]
    # kernel-study wins (mu_LM block, vs additive)
    add = load_baseline("results/kernels/additive.pkl", "LM")
    for row, path in [("chord", "results/kernels/harmonic.pkl"),
                      ("chord_vl", "results/kernels/harmonic_vl.pkl"),
                      ("tonal", "results/kernels/tonal.pkl"),
                      ("temporal", "results/kernels/temporal.pkl")]:
        if not os.path.exists(path):
            print(f"    (skip {row}: {path} missing)")
            continue
        cells = load_baseline(path, "LM")
        for metric in ("rmse", "nll"):
            family[f"{row}_{metric}"] = deltas(cells, add, metric)[0]
    # masking sweep contrasts
    sys.path.insert(0, "scripts")
    from report_mask_sweep import load_cells as ms_load
    for tag, hid in [("obs0.50", 50), ("obs0.60_anchor", 40), ("obs0.70", 30),
                     ("obs0.80", 20), ("obs0.90", 10)]:
        a = ms_load(tag, "b_featlm"); b = ms_load(tag, "b_featlm_nograph")
        family[f"sweep{hid}_graph_rmse"] = deltas(a, b, "rmse")[0]
    # theory features + overlap
    tf = load_cells("results/graphgp_theoryfeat/b_theoryfeat.shard*.pkl")
    anchor_feat = ms_load("obs0.60_anchor", "b_feat")
    family["theoryfeat_rmse"] = deltas(tf, anchor_feat, "rmse")[0]
    ov = load_cells("results/graphgp_overlap/c_overlap.shard*.pkl")
    cg = load_cells("results/graphgp_overlap/c_graph.shard*.pkl")
    family["overlap_rmse"] = deltas(ov, cg, "rmse")[0]
    family["overlap_nll"] = deltas(ov, cg, "nll")[0]

    pvals = {k: boot_pvalue(d) for k, d in family.items()}
    m = len(pvals)
    order = sorted(pvals, key=lambda k: pvals[k])
    print(f"    {m} contrasts; BH at q=0.05:")
    thresh = 0.0
    for i, k in enumerate(order, 1):
        if pvals[k] <= 0.05 * i / m:
            thresh = pvals[k]
    for i, k in enumerate(order, 1):
        surv = "SURVIVES" if pvals[k] <= thresh else "   --   "
        print(f"    {k:<22} p={pvals[k]:.4f}  BH crit {0.05 * i / m:.4f}  {surv}")

    print("\n== D. piece-level coverage, proposed model (dev) ==")
    cov = per_piece_metric(gp, "coverage@0.90")
    vals = np.array([cov[k] for k in sorted(cov)])
    mu, lo, hi = bootstrap_ci(vals, B=2000, rng=np.random.default_rng(11))
    print(f"    mean {mu:.3f}  piece-level 95% CI [{lo:.3f},{hi:.3f}]  "
          f"min {vals.min():.3f}  max {vals.max():.3f}  "
          f"pieces<0.85: {(vals < 0.85).sum()}/{len(vals)}")

    print("\n== E. composer-clustered bootstrap (key contrasts) ==")
    groups = [composers[pi] for pi in pieces]
    from collections import Counter
    print(f"    composers: {dict(Counter(groups))}")
    for name, d in [("GP - two-stage headline", d_pool),
                    ("graph value", d_graph), ("embedding value", d_emb)]:
        lo, hi = clustered_ci(d, groups)
        sig = "*" if (lo > 0) or (hi < 0) else " "
        print(f"    {name:<24} mean {d.mean():+.4f} clustered "
              f"[{lo:+.4f},{hi:+.4f}]{sig}")

    print("\n== F. non-finiteness audit ==")
    print("    every per-piece delta above passed a finite-ness assertion; "
          "no value was dropped from any contrast.")


if __name__ == "__main__":
    main()
