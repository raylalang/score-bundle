"""Graph priors over performance variables.

The prior precision Q_G is built from the graph Laplacian L_G.  Two interchangeable
forms (see the concept note, section 8.4):

    Laplacian:        Q_G = lambda I + alpha L_G
    Matern / SPDE:    Q_G = sigma^{-2} (kappa^2 I + L_G)^alpha

Intuition: Q_G penalizes differences between connected notes, so nearby/related
notes have correlated expressive deviations.  lambda (or kappa) controls the pull
toward the mean; alpha controls smoothness/range.  For integer alpha, Q_G is sparse
(a Gaussian Markov random field).
"""
from __future__ import annotations

import numpy as np


def laplacian_precision(L: np.ndarray, lam: float = 1.0, eta: float = 1.0) -> np.ndarray:
    """Additive graph precision  Q_G = lam I + eta L.

    ``lam`` is the ridge / pull-to-mean term; ``eta`` is the Laplacian (smoothing)
    weight.  Distinct from the Matern exponent ``alpha`` in :func:`matern_precision`.
    """
    if lam <= 0:
        raise ValueError("lam must be > 0 for a positive-definite precision")
    return lam * np.eye(L.shape[0]) + eta * L


def matern_precision(
    L: np.ndarray, kappa: float = 1.0, alpha: int = 2, sigma_g: float = 1.0
) -> np.ndarray:
    """Matern / SPDE graph precision  Q_G = sigma_g^{-2} (kappa^2 I + L)^alpha.

    ``alpha`` (positive integer) is the smoothness order; ``sigma_g`` the prior
    marginal scale (reserve plain ``sigma`` for the *posterior* standard deviation).
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if int(alpha) != alpha or alpha < 1:
        raise ValueError("alpha must be a positive integer for the GMRF form")
    base = kappa ** 2 * np.eye(L.shape[0]) + L
    M = np.linalg.matrix_power(base, int(alpha))
    return M / (sigma_g ** 2)


def is_positive_definite(Q: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(Q)
        return True
    except np.linalg.LinAlgError:
        return False
