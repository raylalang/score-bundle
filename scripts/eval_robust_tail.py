#!/usr/bin/env python
"""Student-t tau likelihood on a DEV replica of the confirmation tail.

The masking-level sweep found a development copy of the confirmation-set
Gaussian-tail failure: at 30% hidden (obs0.70), piece 28 / seed 2 scores
NLL ~ +35 for every config (four tau notes with ~3-beat onset residuals under
tight evidence-fitted intervals). The Student-t prototype (gp_robust) was
previously measured only as a no-harm check on tail-free data; this script
measures it where the tail actually exists — dev-only, the confirmation set
stays untouched.

Protocol: all 30 dev pieces of the obs0.70 fraction, seed 2 (the tail seed),
b_featlm information set (features + mask-aware embeddings), Gaussian fit vs
the EM Student-t tau fit (nu=4, 3 EM iters). Both scored two ways: Gaussian
NLL (the published metric — harsh on the t-variant by construction) and each
model's own predictive NLL (t-scoring for the t-variant on tau). RMSE and
coverage are likelihood-agnostic.

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/eval_robust_tail.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_graphgp import piece_setup  # noqa: E402

_LOG2PI = float(np.log(2.0 * np.pi))
_Z90 = 1.6448536269514722
SEED = 2
INPUTS = ".cache/masksweep_inputs_obs0.70.pkl"
EMB = ".cache/masksweep_emb_obs0.70.pkl"
OUT = "results/robust_tail_obs070.pkl"


def cell_metrics(yt, pr, sd):
    err = yt - pr
    nll = 0.5 * (_LOG2PI + 2 * np.log(sd) + (err / sd) ** 2)
    return (float(np.sqrt(np.mean(err ** 2))), float(np.mean(nll)),
            float(np.mean(np.abs(err) <= _Z90 * sd)))


def main() -> None:
    from score_bundle.downstream import load_piece_arrays
    from score_bundle.gp import MultiOutputGraphGP
    from score_bundle.gp_robust import fit_robust_tau, t_predictive_nll

    _, ev, _ = load_piece_arrays(".cache/asap_arrays_named.pkl")
    ev = ev[:30]
    with open(INPUTS, "rb") as fh:
        masks = pickle.load(fh)["masks"]
    with open(EMB, "rb") as fh:
        emb_dump = pickle.load(fh)["emb_ma"]

    rows = []
    for pi, p in enumerate(ev):
        Y = np.asarray(p["y"], dtype=float)
        mask = masks[(pi, SEED)]
        emb = emb_dump[(pi, SEED)]
        feats, graph_eig, n_graph, g0 = piece_setup(p, "b_featlm", emb=emb)
        nu, U = graph_eig(g0)
        held = ~mask
        floor = 0.05 * np.array([float(np.var(Y[mask, c])) for c in range(3)])

        # Gaussian reference fit
        gp = MultiOutputGraphGP(nu, U, kernel="additive", features=feats,
                                n_channels=3)
        xg, _ = gp.fit(Y, mask, noise_floor=floor, maxiter=200)
        M, S = gp.posterior(Y, mask, xg)
        nv = gp.unpack(xg)["noise"]
        yt = np.concatenate([Y[held, c] for c in range(3)])
        pr = np.concatenate([M[held, c] for c in range(3)])
        sd = np.concatenate([np.sqrt(S[held, c] ** 2 + nv[c]) for c in range(3)])
        g_rmse, g_nll, g_cov = cell_metrics(yt, pr, sd)

        # Student-t tau fit (EM), scored both ways
        gpt, xt, w = fit_robust_tau(nu, U, Y, mask, features=feats,
                                    kernel="additive", nu_t=4.0, em_iters=3,
                                    noise_floor=floor, maxiter=200)
        Mt, St = gpt.posterior(Y, mask, xt)
        nvt = gpt.unpack(xt)["noise"]
        yt2 = np.concatenate([Y[held, c] for c in range(3)])
        pr2 = np.concatenate([Mt[held, c] for c in range(3)])
        sd2 = np.concatenate([np.sqrt(St[held, c] ** 2
                                      + nvt[c] * gpt.noise_scale[held, c])
                              for c in range(3)])
        t_rmse, t_gnll, t_cov = cell_metrics(yt2, pr2, sd2)
        # own-likelihood scoring: t on tau, Gaussian elsewhere
        n_h = int(held.sum())
        own = []
        for c in range(3):
            y_c, m_c = Y[held, c], Mt[held, c]
            s_c = np.sqrt(St[held, c] ** 2 + nvt[c] * gpt.noise_scale[held, c])
            if c == 0:
                own.append(t_predictive_nll(y_c, m_c, s_c, nu_t=4.0))
            else:
                own.append(0.5 * (_LOG2PI + 2 * np.log(s_c)
                                  + ((y_c - m_c) / s_c) ** 2))
        t_ownnll = float(np.mean(np.concatenate(own)))
        rows.append({"piece": pi, "n_held": n_h,
                     "gauss": (g_rmse, g_nll, g_cov),
                     "t_gaussscored": (t_rmse, t_gnll, t_cov),
                     "t_own_nll": t_ownnll,
                     "min_w": float(w.min())})
        flag = "  <-- TAIL CELL" if pi == 28 else ""
        print(f"piece {pi:2d}  G: rmse {g_rmse:.3f} nll {g_nll:+7.3f} cov {g_cov:.3f}"
              f"  |  t: rmse {t_rmse:.3f} gnll {t_gnll:+7.3f} own {t_ownnll:+7.3f}"
              f" cov {t_cov:.3f}  min_w {w.min():.3f}{flag}", flush=True)

    g_nlls = np.array([r["gauss"][1] for r in rows])
    t_gnlls = np.array([r["t_gaussscored"][1] for r in rows])
    t_own = np.array([r["t_own_nll"] for r in rows])
    print("\n=== summary (30 dev pieces, obs0.70 seed 2) ===")
    print(f"Gaussian:   mean NLL {g_nlls.mean():+.3f}  median {np.median(g_nlls):+.3f}"
          f"  worst {g_nlls.max():+.3f}")
    print(f"t (G-scored): mean {t_gnlls.mean():+.3f}  median {np.median(t_gnlls):+.3f}"
          f"  worst {t_gnlls.max():+.3f}")
    print(f"t (own):    mean {t_own.mean():+.3f}  median {np.median(t_own):+.3f}"
          f"  worst {t_own.max():+.3f}")
    print(f"RMSE: G {np.mean([r['gauss'][0] for r in rows]):.4f}"
          f"  t {np.mean([r['t_gaussscored'][0] for r in rows]):.4f}")
    print(f"cov:  G {np.mean([r['gauss'][2] for r in rows]):.3f}"
          f"  t {np.mean([r['t_gaussscored'][2] for r in rows]):.3f}")
    tail = rows[28]
    print(f"\nTAIL CELL piece 28: Gaussian NLL {tail['gauss'][1]:+.3f} -> "
          f"t own-scored {tail['t_own_nll']:+.3f} "
          f"(G-scored {tail['t_gaussscored'][1]:+.3f}); min weight {tail['min_w']:.3f}")

    os.makedirs("results", exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump({"rows": rows, "meta": {
            "protocol": "obs0.70 seed 2, b_featlm info set, DEV only",
            "nu_t": 4.0, "em_iters": 3,
            "scoring": "Gaussian NLL for both + t-on-tau own scoring for the t fit",
        }}, fh)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
