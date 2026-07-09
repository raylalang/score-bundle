"""Phase 1, the thesis model: one multi-output graph GP on a synthetic score.

Demonstrates the GP-first formulation end to end (numpy-only, no torch needed):
a score graph, an ICM-coupled spectral kernel, score features entering the kernel
as a marginalized Bayesian linear mean, everything fit by the exact per-piece
evidence — then held-out imputation with calibrated per-note, per-channel
uncertainty, against its own nested ablations (no graph / no features).

    python examples/phase1_graphgp.py
"""
from __future__ import annotations

import numpy as np

from score_bundle import imputation_eval as ie
from score_bundle.baselines import rich_score_features
from score_bundle.gp import MultiOutputGraphGP
from score_bundle.graph import build_adjacency, laplacian
from score_bundle.metrics import evaluate
from score_bundle.synthetic import random_score


def zscore_cols(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)


def main() -> None:
    rng = np.random.default_rng(0)
    score = random_score(120, rng)
    nu, U = np.linalg.eigh(laplacian(build_adjacency(score)))
    n = len(score)

    # ground truth drawn FROM the model class: correlated channels sharing one
    # smooth graph field, plus a feature-driven component and channel noise
    X = np.concatenate([zscore_cols(rich_score_features(score, rff_dim=0)),
                        np.ones((n, 1))], axis=1)
    g = np.clip(1.0 / (1.0 + 1.5 * nu), 1e-12, None)
    f = U @ (np.sqrt(g) * rng.standard_normal(n))
    B_true = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
    Lb = np.linalg.cholesky(B_true)
    F = np.outer(f, np.ones(3)) @ Lb.T * 0.6
    beta = rng.standard_normal((X.shape[1], 3)) * 0.15
    Y = F + X @ beta + 0.1 * rng.standard_normal((n, 3))

    mask = ie.random_mask(n, rng, observed_frac=0.6)
    held = ~mask
    floor = 0.05 * Y[mask].var(axis=0)

    def run(name: str, kernel: str, feats):
        gp = MultiOutputGraphGP(nu, U, kernel=kernel, features=feats)
        x_hat, info = gp.fit(Y, mask, noise_floor=floor, maxiter=150)
        M, S = gp.posterior(Y, mask, x_hat)
        nv = gp.unpack(x_hat)["noise"]
        yt = np.concatenate([Y[held, c] for c in range(3)])
        pr = np.concatenate([M[held, c] for c in range(3)])
        sd = np.concatenate([np.sqrt(S[held, c] ** 2 + nv[c]) for c in range(3)])
        m = evaluate(yt, pr, sd, level=0.9)
        print(f"  {name:28s} RMSE {m['rmse']:.4f}  NLL {m['nll']:+.3f}  "
              f"cov@.9 {m['coverage@0.90']:.3f}")
        return m

    print(f"synthetic piece: {n} notes, {int(held.sum())} held out\n"
          f"one GP, everything by evidence — vs its nested ablations:")
    full = run("GP-first (graph + features)", "additive", [X])
    run("  ablation: no features", "additive", [])
    run("  ablation: no graph", "none", [X])
    print("\nThe full model should win on both axes: the features recover the "
          "note-level component,\nthe graph recovers the shared field and keeps "
          "the intervals honest.")


if __name__ == "__main__":
    main()
