"""Score graph construction (the discrete base of the score bundle).

The graph couples notes that are close in score time / pitch / voice, so that the
graph prior (:mod:`score_bundle.prior`) encourages musically related notes to share
expressive behaviour.  Implemented densely with NumPy for clarity; for large scores
the same objects are sparse GMRFs and should be assembled with ``scipy.sparse``.
"""
from __future__ import annotations

import numpy as np

from .score import Score


def build_adjacency(
    score: Score,
    ell_b: float = 2.0,
    ell_p: float = 4.0,
    voice_weight: float = 0.0,
    knn: int | None = None,
) -> np.ndarray:
    """Gaussian edge weights from score-time and pitch distance.

    W_ij = exp(-(b_i-b_j)^2 / 2 ell_b^2  -  (p_i-p_j)^2 / 2 ell_p^2),  i != j.

    Parameters
    ----------
    ell_b, ell_p: length scales (beats, semitones).
    voice_weight: optional additive bonus for same-voice pairs.
    knn:          if given, keep only the top-``knn`` neighbours per node
                  (then symmetrize).
    """
    b = score.onset
    p = score.pitch
    db = b[:, None] - b[None, :]
    dp = p[:, None] - p[None, :]
    W = np.exp(-(db ** 2) / (2 * ell_b ** 2) - (dp ** 2) / (2 * ell_p ** 2))
    np.fill_diagonal(W, 0.0)

    if voice_weight:
        same = (score.voice[:, None] == score.voice[None, :]).astype(float)
        W = W + voice_weight * same
        np.fill_diagonal(W, 0.0)

    if knn is not None and knn < len(score) - 1:
        W = _knn_sparsify(W, knn)
    return W


def _knn_sparsify(W: np.ndarray, k: int) -> np.ndarray:
    n = W.shape[0]
    out = np.zeros_like(W)
    for i in range(n):
        idx = np.argsort(W[i])[::-1][:k]
        out[i, idx] = W[i, idx]
    # symmetrize: keep an edge if it appears in either node's neighbourhood
    out = np.maximum(out, out.T)
    return out


def degree(W: np.ndarray) -> np.ndarray:
    """Diagonal degree matrix D = diag(sum_j W_ij)."""
    return np.diag(W.sum(axis=1))


def laplacian(W: np.ndarray, normalized: bool = False) -> np.ndarray:
    """Graph Laplacian.

    Combinatorial:  L = D - W.
    Normalized:     L = I - D^{-1/2} W D^{-1/2}.
    """
    d = W.sum(axis=1)
    if normalized:
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        return np.eye(W.shape[0]) - (dinv[:, None] * W * dinv[None, :])
    return np.diag(d) - W


def chain_adjacency(order: np.ndarray | None = None, n: int | None = None) -> np.ndarray:
    """Adjacency of a 1-D chain (nearest neighbours in a given order).

    Used by the *temporal-only* baseline: notes coupled only to their score-time
    neighbours (an AR(1)/random-walk-style smoother).
    """
    if order is not None:
        n = len(order)
        perm = np.asarray(order)
    else:
        assert n is not None, "provide either `order` or `n`"
        perm = np.arange(n)
    W = np.zeros((n, n))
    for a, b in zip(perm[:-1], perm[1:]):
        W[a, b] = W[b, a] = 1.0
    return W
