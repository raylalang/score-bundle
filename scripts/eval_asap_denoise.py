#!/usr/bin/env python
"""Downstream task 3 — transcription denoising on held-out ASAP.

Observe *every* note through synthetic i.i.d. observation noise (simulating a noisy AMT
transcription, as in the ATEPP / Aria-MIDI scaling corpora) and recover the clean
expressive values by posterior shrinkage.  Metrics compare the recovered latent field to
the clean targets: RMSE, plus NLL / coverage of the *latent* posterior std.

Methods (per channel, noise std = ``level`` x channel std):
    identity          the noisy observation itself (std = oracle noise level; calibrated
                      by construction but inaccurate — the floor to beat on RMSE);
    independent       scalar Wiener shrinkage toward the mean, *oracle* noise (without
                      coupling the noise level is unidentifiable, so it must be given);
    graph             GMRF posterior, *blind* — (lam, eta, noise_var) all EB-fit; the
                      graph structure is what makes blind noise estimation possible;
    graph-oracle      GMRF posterior with the true noise variance fixed.

Means: zero and LM (score-only embeddings by default).  Runs from the named cache
(numpy-only):

    python scripts/eval_asap_denoise.py --arrays-cache .cache/asap_arrays_named.pkl
"""
from __future__ import annotations

import argparse

import numpy as np

from score_bundle.downstream import denoise_channel, load_piece_arrays, piece_score
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.lm import features as lmfeat
from score_bundle.metrics import evaluate


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
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.5, 1.0],
                    help="noise std in units of the per-channel clean std")
    ap.add_argument("--methods", nargs="+",
                    default=["identity", "independent", "graph", "graph-oracle"])
    ap.add_argument("--means", nargs="+", default=["zero", "LM"])
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--embeddings", default="emb_scoreonly", choices=["emb", "emb_scoreonly"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    head, ev, meta = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]
    print(f"{len(head)} head + {len(ev)} eval pieces | levels={args.levels} "
          f"| embeddings={args.embeddings}")

    H = np.concatenate([p[args.embeddings] for p in head])
    W = lmfeat.fit_prior_mean_head(H, np.concatenate([p["y"] for p in head]), l2=args.l2)

    channels = ["tau", "log r", "v"]
    # (level, mean, method) -> pooled arrays + per-piece RMSE/NLL for CIs
    pool = {}
    per_piece = {}
    rng = np.random.default_rng(args.seed)
    for p in ev:
        score = piece_score(p)
        L = laplacian(build_adjacency(score))
        mu_lm = lmfeat.apply_prior_mean(p[args.embeddings], W)
        for ci, cname in enumerate(channels):
            y_clean = p["y"][:, ci]
            sd_c = float(np.std(y_clean)) or 1.0
            for level in args.levels:
                noise_std = level * sd_c
                y_noisy = y_clean + rng.normal(scale=noise_std, size=y_clean.size)
                for mname in args.means:
                    mean = np.zeros_like(y_clean) if mname == "zero" else mu_lm[:, ci]
                    for method in args.methods:
                        if method in ("identity",) and mname != args.means[0]:
                            continue  # identity ignores the mean; report it once
                        pred, std = denoise_channel(L, y_noisy, mean, noise_std, method)
                        key = (level, mname if method != "identity" else "-", method)
                        buf = pool.setdefault(key, [[], [], [], []])
                        buf[0].append(y_clean); buf[1].append(pred); buf[2].append(std)
                        buf[3].append(np.full(y_clean.size, ci))
                        m = evaluate(y_clean, pred, std, level=0.9)
                        pp = per_piece.setdefault(key, {"rmse": [], "nll": []})
                        pp["rmse"].append(m["rmse"]); pp["nll"].append(m["nll"])
    boot_rng = np.random.default_rng(7)
    print(f"\nDenoising ({len(ev)} pieces; latent posterior vs clean targets, "
          f"pooled over channels)")
    print(f"{'level':>6s} {'mean':5s} {'method':13s} {'RMSE [95% CI]':>24s} "
          f"{'NLL [95% CI]':>24s} {'cov@.9':>7s}")
    for key in sorted(pool, key=lambda k: (k[0], k[1], k[2])):
        level, mname, method = key
        y = np.concatenate(pool[key][0]); pr = np.concatenate(pool[key][1])
        sd = np.concatenate(pool[key][2])
        m = evaluate(y, pr, sd, level=0.9)
        r_m, r_lo, r_hi = bootstrap_ci(per_piece[key]["rmse"], rng=boot_rng)
        n_m, n_lo, n_hi = bootstrap_ci(per_piece[key]["nll"], rng=boot_rng)
        print(f"{level:6.2f} {mname:5s} {method:13s} "
              f"{m['rmse']:7.4f} [{r_lo:.3f},{r_hi:.3f}]  "
              f"{m['nll']:7.3f} [{n_lo:.2f},{n_hi:.2f}]  {m['coverage@0.90']:7.3f}")
    print("\nReading: identity is calibrated by construction (oracle noise std) but has "
          "the worst RMSE;\na good structured denoiser must beat it on RMSE while keeping "
          "coverage near 0.90.\n'graph' is fully blind (noise level estimated); "
          "'independent'/'graph-oracle' are told the true noise.")


if __name__ == "__main__":
    main()
