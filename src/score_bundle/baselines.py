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
