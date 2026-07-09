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


def fifths_distance(p_i: np.ndarray, p_j: np.ndarray) -> np.ndarray:
    """Circle-of-fifths distance between pitch *classes* (0..6 steps).

    Position on the circle is ``(pc * 7) mod 12`` (C, G, D, ... in fifths order);
    the distance is the shorter way around the ring.  A perfect fifth is 1 step,
    a major second 2, a semitone 5 — the music-theoretic notion of tonal proximity,
    deliberately different from raw semitone distance.
    """
    pos_i = (np.asarray(p_i, dtype=int) * 7) % 12
    pos_j = (np.asarray(p_j, dtype=int) * 7) % 12
    d = np.abs(pos_i - pos_j)
    return np.minimum(d, 12 - d).astype(float)


def build_adjacency_tonal(
    score: Score,
    ell_b: float = 2.0,
    ell_5: float = 2.0,
    ell_oct: float = 1.0,
    knn: int | None = None,
) -> np.ndarray:
    """Tonal-distance edges: music-theoretic pitch distance instead of semitones.

    Replaces the raw pitch term ``(p_i - p_j)^2 / 2 ell_p^2`` of
    :func:`build_adjacency` with a circle-of-fifths pitch-class distance
    (:func:`fifths_distance`, length scale ``ell_5`` in fifths steps) plus an octave
    displacement term ``((p_i - p_j)/12)^2 / 2 ell_oct^2`` that keeps register
    locality.  Under this metric an octave or a perfect fifth is *close* and a
    semitone is *far* — the Tier-C "music theory in the kernel" hypothesis of the
    kernel comparison (docs/kernel_comparison_experiment.md).  The score-time term
    is unchanged.
    """
    b = score.onset
    p = score.pitch
    db = b[:, None] - b[None, :]
    d5 = fifths_distance(p[:, None], p[None, :])
    doct = (p[:, None] - p[None, :]) / 12.0
    W = np.exp(
        -(db ** 2) / (2 * ell_b ** 2)
        - (d5 ** 2) / (2 * ell_5 ** 2)
        - (doct ** 2) / (2 * ell_oct ** 2)
    )
    np.fill_diagonal(W, 0.0)
    if knn is not None and knn < len(score) - 1:
        W = _knn_sparsify(W, knn)
    return W


def build_adjacency_harmonic(
    score: Score,
    ell_b: float = 2.0,
    ell_p: float = 4.0,
    chord_weight: float = 1.0,
    vl_weight: float = 0.0,
    vl_window: float = 2.0,
) -> np.ndarray:
    """Combinatorial base graph plus explicit harmonic / voice-leading edge families.

    Starts from :func:`build_adjacency` (unchanged defaults) and adds:

    - **chord edges** (``chord_weight``): notes with *identical* score onset —
      same-chord membership, the strongest harmonic-simultaneity signal available
      without a harmonic analysis;
    - **voice-leading edges** (``vl_weight``): stepwise motion — pairs within
      ``vl_window`` beats, at *different* onsets, whose pitches differ by 1–2
      semitones (melodic step), the voice-leading proximity family.

    Cadential/functional edges would need a harmonic analysis and are left as
    future work.  Each family ablates independently by setting its weight to 0.
    """
    W = build_adjacency(score, ell_b=ell_b, ell_p=ell_p)
    b = score.onset
    p = score.pitch
    db = np.abs(b[:, None] - b[None, :])
    if chord_weight:
        same_onset = (db < 1e-9).astype(float)
        W = W + chord_weight * same_onset
    if vl_weight:
        dp = np.abs(p[:, None] - p[None, :])
        step = (db > 1e-9) & (db <= vl_window) & (dp >= 1) & (dp <= 2)
        W = W + vl_weight * step.astype(float)
    np.fill_diagonal(W, 0.0)
    return W


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
