#!/usr/bin/env python
"""Leave-one-out (LOO) evaluation of the GP-first model on the DEV pieces.

The limit point of the masking-level sweep (requested 2026-07-13): every note is
predicted from all the others.  Per piece, hyperparameters are fit once by exact
evidence on the fully observed piece, then the closed-form LOO predictive
(:meth:`score_bundle.gp.MultiOutputGraphGP.loo_predictive`; Rasmussen & Williams
S5.4.2) scores each observation.  Honesty notes, recorded in the output meta:

* hyperparameters see the whole piece — standard GP LOO-CV removes the
  observation from the prediction, not from the hyperparameter evidence;
* the LM feature matrix is the leak-free pre-velocity readout
  (``emb_leakfree``): note i's embedding never contains its own velocity, and
  tau / log r never enter the LM input at all, so the LOO limit stays leak-free
  without per-note re-embedding.

DEV pieces only (head of the eval list); the confirmation set (offset 30+) is
never read.

    OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/eval_gp_loo.py --guard
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import piece_setup  # noqa: E402

_LOG2PI = float(np.log(2.0 * np.pi))
_Z90 = 1.6448536269514722  # two-sided 90% normal quantile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays-cache", default=".cache/asap_arrays_named.pkl")
    ap.add_argument("--configs", default="b_feat,b_featlm,b_featlm_nograph")
    ap.add_argument("--n-eval-pieces", type=int, default=30)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--guard", action="store_true",
                    help="guarded evidence fits (matches the sweep's --guard runs)")
    ap.add_argument("--out", default="results/graphgp_masksweep/loo.pkl")
    args = ap.parse_args()

    from score_bundle.downstream import load_piece_arrays
    from score_bundle.gp import MultiOutputGraphGP

    _, ev, _ = load_piece_arrays(args.arrays_cache)
    ev = ev[: args.n_eval_pieces]  # DEV head of the eval list only
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    results: dict = {}
    for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
        kernel = "none" if config.endswith("_nograph") else "additive"
        cells, rows = {}, []
        for pi, p in enumerate(ev):
            Y = np.asarray(p["y"], dtype=float)
            emb = (np.asarray(p["emb_leakfree"], dtype=np.float32)
                   if "featlm" in config else None)
            feats, graph_eig, n_graph, g0 = piece_setup(p, config, emb=emb)
            assert n_graph == 0, "LOO evaluator covers fixed-graph configs only"
            nu, U = graph_eig(g0)
            gp = MultiOutputGraphGP(nu, U, kernel=kernel, features=feats,
                                    n_channels=Y.shape[1])
            mask = np.ones(len(Y), dtype=bool)
            floor = 0.05 * np.array([float(np.var(Y[:, c]))
                                     for c in range(Y.shape[1])])
            if args.guard:
                x_hat, info = gp.fit_guarded(Y, mask, noise_floor=floor,
                                             maxiter=args.maxiter,
                                             rng=np.random.default_rng(0))
            else:
                x_hat, info = gp.fit(Y, mask, noise_floor=floor,
                                     maxiter=args.maxiter)
            m, v = gp.loo_predictive(Y, x_hat)
            sd = np.sqrt(np.maximum(v, 1e-12))
            err = Y - m
            cells[(config, pi)] = (Y.astype(np.float32), m.astype(np.float32),
                                   v.astype(np.float32))
            rows.append({
                "piece": pi, "n": len(Y),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "nll": float(np.mean(0.5 * (_LOG2PI + 2 * np.log(sd)
                                            + (err / sd) ** 2))),
                "coverage@0.90": float(np.mean(np.abs(err) <= _Z90 * sd)),
                "info": {k: info[k] for k in ("optimizer", "nll") if k in info},
            })
            print(f"[{config}] piece {pi + 1}/{len(ev)} "
                  f"rmse {rows[-1]['rmse']:.4f} nll {rows[-1]['nll']:.3f} "
                  f"cov {rows[-1]['coverage@0.90']:.3f}", flush=True)
        pooled = {
            "rmse": float(np.mean([r["rmse"] for r in rows])),
            "nll": float(np.mean([r["nll"] for r in rows])),
            "coverage@0.90": float(np.mean([r["coverage@0.90"] for r in rows])),
        }
        print(f"=== {config} (LOO, mean over {len(rows)} dev pieces): "
              f"rmse {pooled['rmse']:.4f} nll {pooled['nll']:.3f} "
              f"cov {pooled['coverage@0.90']:.3f}", flush=True)
        results[config] = {"per_piece": rows, "pooled": pooled, "cells": cells}

    with open(args.out, "wb") as fh:
        pickle.dump({"results": results, "meta": {
            "protocol": "LOO limit of the masking sweep; DEV pieces only",
            "arrays_cache": args.arrays_cache,
            "n_eval_pieces": args.n_eval_pieces,
            "guard": bool(args.guard), "maxiter": args.maxiter,
            "hyperparams": "fit once per piece on the fully observed piece "
                           "(standard GP LOO-CV; evidence sees every note)",
            "embeddings": "emb_leakfree (pre-velocity readout; leak-free per "
                          "note by construction, no per-note re-embedding)",
        }}, fh)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
