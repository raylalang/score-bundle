"""Baselines for the Phase-1 imputation/recovery task.

- Independent per-note prior   : diagonal precision, no graph coupling.
- Temporal smoothing (AR(1))   : chain graph along score time only.
- Score-feature regression     : predict expression from local score features
                                 (ridge; gradient-boosted trees if scikit-learn
                                 is installed). No inter-note coupling.

The *proposed* model is :class:`score_bundle.model.GraphGaussianField` with the
full score graph; a win requires beating both the independent and temporal-only
baselines on recovery *and* calibration.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .graph import chain_adjacency, laplacian
from .model import GraphGaussianField
from .prior import laplacian_precision
from .score import Score


def independent_field(n: int, prior_var: float = 1.0, mean=0.0) -> GraphGaussianField:
    """Diagonal prior precision (no coupling)."""
    Q = np.eye(n) / float(prior_var)
    return GraphGaussianField(Q, mean=mean)


def temporal_field(
    score: Score, lam: float = 1.0, eta: float = 1.0, mean=0.0
) -> GraphGaussianField:
    """AR(1)-style chain graph along score-time order."""
    order = np.argsort(score.onset)
    W = chain_adjacency(order=order)
    L = laplacian(W)
    Q = laplacian_precision(L, lam=lam, eta=eta)
    return GraphGaussianField(Q, mean=mean)


def score_features(score: Score) -> np.ndarray:
    """Simple local score features: [pitch, onset-in-bar proxy, beat strength, duration]."""
    p = score.pitch
    b = score.onset
    d = score.duration
    onset_frac = b - np.floor(b)            # position within the beat/bar (proxy)
    beat_strength = np.cos(2 * np.pi * onset_frac)  # high on strong beats
    X = np.stack([p, onset_frac, beat_strength, d], axis=1)
    # standardize
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    return (X - mu) / sd


def ridge_impute(
    score: Score,
    y_obs: np.ndarray,
    mask: np.ndarray,
    l2: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Predict masked notes from observed ones via ridge on score features.

    Returns (predictions over all notes, residual std estimated on observed set).
    """
    X = score_features(score)
    Xo, yo = X[mask], np.asarray(y_obs)[mask]
    n_feat = X.shape[1]
    A = Xo.T @ Xo + l2 * np.eye(n_feat)
    w = np.linalg.solve(A, Xo.T @ (yo - yo.mean()))
    b = yo.mean()
    pred = X @ w + b
    resid = yo - (Xo @ w + b)
    sigma = float(np.std(resid)) if resid.size > 1 else 1.0
    return pred, sigma


def gbm_impute(score: Score, y_obs: np.ndarray, mask: np.ndarray):
    """Gradient-boosted-trees baseline (requires scikit-learn)."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("gbm_impute requires scikit-learn (pip install score-bundle[baselines])") from exc

    X = score_features(score)
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X[mask], np.asarray(y_obs)[mask])
    return model.predict(X)


def _window_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of ``x`` over the +-``w``-note window around each position (cumsum trick)."""
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    lo = np.maximum(i - w, 0)
    hi = np.minimum(i + w + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def rich_score_features(
    score: Score,
    window: int = 8,
    rff_dim: int = 0,
    rff_seed: int = 0,
) -> np.ndarray:
    """Rich, score-only per-note features — the honest rival to the LM embedding.

    Everything here is computed from the score support alone (pitch / onset /
    duration / voice, never the performance), so a prior-mean head fit on these
    features under the *identical* cross-piece protocol as the LM head
    (:func:`score_bundle.lm.features.fit_prior_mean_head` on head pieces, applied
    out-of-sample) isolates what the pretrained transformer adds beyond hand-built
    local score structure.  Unlike :func:`score_features` (4 dims, fit per piece on
    the observed notes), these are designed to transfer across pieces.

    Base features (score-time order, mapped back to the input order): absolute and
    piece-relative pitch, local pitch deviation / trend / spread, melodic interval
    and contour, log-duration (absolute / piece-z / local-relative), metrical phase
    at beat, half-bar and bar periods (sin + cos), inter-onset intervals, local note
    density, chord size and within-chord pitch rank, piece position and edge
    proximity, voice.  With ``rff_dim > 0``, appends deterministic random Fourier
    features of the base features (seed ``rff_seed``) so a linear head can fit a
    smooth nonlinear function.
    """
    n = len(score)
    p = score.pitch.astype(float)
    b = score.onset.astype(float)
    d = score.duration.astype(float)
    v = score.voice.astype(float)
    order = np.lexsort((p, b))
    ps, bs, ds, vs = p[order], b[order], d[order], v[order]

    w = max(1, min(window, n - 1)) if n > 1 else 1
    eps = 1e-9

    def pz(x):  # per-piece z-score
        return (x - x.mean()) / (x.std() + eps)

    # pitch
    pitch_abs = (ps - 60.0) / 24.0
    pitch_z = pz(ps)
    m_p = _window_mean(ps, w)
    pitch_local_dev = (ps - m_p) / 12.0
    m_p2 = _window_mean(ps * ps, w)
    pitch_local_std = np.sqrt(np.maximum(m_p2 - m_p ** 2, 0.0)) / 12.0
    interval_prev = np.diff(ps, prepend=ps[:1]) / 12.0
    contour = np.sign(interval_prev)
    # local pitch-vs-time slope over the window (semitones per beat, squashed)
    m_b = _window_mean(bs, w)
    m_bp = _window_mean(bs * ps, w)
    m_b2 = _window_mean(bs * bs, w)
    slope = (m_bp - m_b * m_p) / (m_b2 - m_b ** 2 + eps)
    slope = np.tanh(slope / 12.0)
    # duration
    dur_log = np.log1p(ds)
    dur_z = pz(dur_log)
    dur_local_rel = dur_log - _window_mean(dur_log, w)
    # metrical phase (beat / half-bar / bar proxies; ASAP onsets are in beats)
    phases = []
    for period in (1.0, 2.0, 4.0):
        ph = 2.0 * np.pi * np.mod(bs, period) / period
        phases += [np.sin(ph), np.cos(ph)]
    # inter-onset intervals and local density
    ioi_prev = np.log1p(np.diff(bs, prepend=bs[:1]))
    ioi_next = np.log1p(np.diff(bs, append=bs[-1:]))
    span = 2.0
    dens = (np.searchsorted(bs, bs + span, side="right")
            - np.searchsorted(bs, bs - span, side="left")) / (2.0 * span)
    dens = np.log1p(dens)
    # chords (identical onsets): size and normalized pitch rank within the chord
    _, grp, counts = np.unique(np.round(bs, 6), return_inverse=True, return_counts=True)
    chord_size = np.log1p(counts[grp].astype(float))
    starts = np.concatenate([[0], np.cumsum(counts)])[grp]
    chord_rank = (np.arange(n) - starts) / np.maximum(counts[grp] - 1.0, 1.0)
    # position in the piece and edge proximity
    pos = np.arange(n) / max(n - 1, 1)
    edge_start = np.exp(-np.arange(n) / 8.0)
    edge_end = np.exp(-(n - 1 - np.arange(n)) / 8.0)
    voice_n = vs / (vs.max() + 1.0)

    X = np.stack(
        [pitch_abs, pitch_z, pitch_local_dev, pitch_local_std, interval_prev,
         contour, slope, dur_log, dur_z, dur_local_rel, *phases,
         ioi_prev, ioi_next, dens, chord_size, chord_rank,
         pos, edge_start, edge_end, voice_n],
        axis=1,
    )
    if rff_dim > 0:
        rng = np.random.default_rng(rff_seed)
        G = rng.normal(0.0, 1.0, size=(X.shape[1], rff_dim))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=rff_dim)
        Z = np.sqrt(2.0 / rff_dim) * np.cos(X @ G + phase)
        X = np.concatenate([X, Z], axis=1)

    out = np.empty_like(X)
    out[order] = X
    return out
