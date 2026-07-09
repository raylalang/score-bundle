"""Graph priors over performance variables.

The prior precision Q_G is built from the graph Laplacian L_G.  Two interchangeable
forms (see the concept note, section 8.4):

    Laplacian:        Q_G = lambda I + eta L_G
    Matern / SPDE:    Q_G = sigma_g^{-2} (kappa^2 I + L_G)^alpha

Intuition: Q_G penalizes differences between connected notes, so nearby/related
notes have correlated expressive deviations.  lambda (or kappa) controls the pull
toward the mean; eta weights the Laplacian coupling and alpha (the Matern
exponent) controls smoothness/range.  For integer alpha, Q_G is sparse (a
Gaussian Markov random field).

For the kernel-comparison experiment (docs/kernel_comparison_experiment.md) every
kernel here is expressed *spectrally*: with L = U diag(nu) U^T, the prior covariance
is K = U diag(g(nu; theta)) U^T, where g > 0 maps Laplacian eigenvalues to
covariance eigenvalues.  This one form covers the additive Laplacian
(g = 1/(lam + eta nu)), the Matern family (g = sigma_g^2 (kappa^2 + nu)^-alpha) and
its alpha -> infinity limit, the diffusion / heat kernel (g = sigma_g^2 exp(-t nu)),
whose *precision* eigenvalues exp(t nu) overflow — the covariance form stays benign.
The p-step random walk (I + eta L)^p is the Matern family reparameterized
(kappa^2 = 1/eta, sigma_g^2 = eta^-p), so it is deliberately not a separate kernel.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Tuple

import numpy as np

# covariance eigenvalues are clipped to this range: keeps the diffusion kernel's
# underflow (exp(-t nu) -> 0) and near-zero ridge terms from producing singular /
# infinite kernels inside the EB objective
_G_MIN, _G_MAX = 1e-12, 1e12


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


# --------------------------------------------------------------------------- spectral kernels
class SpectralKernel(NamedTuple):
    """A graph-GP kernel as a spectral function of the Laplacian.

    ``cov_eigs(nu, params)`` maps Laplacian eigenvalues ``nu`` (ascending, >= 0) and a
    positive parameter vector to prior *covariance* eigenvalues g(nu).  ``x0`` is the
    log-space initial point for the EB fit (one entry per parameter, matching the
    ``(0, 0)`` start of :func:`~score_bundle.model.fit_laplacian_field`).
    """

    param_names: Tuple[str, ...]
    x0: Tuple[float, ...]
    cov_eigs: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _matern_cov_eigs(alpha: int):
    def g(nu: np.ndarray, params: np.ndarray) -> np.ndarray:
        sigma_g, kappa = params
        return sigma_g ** 2 / (kappa ** 2 + nu) ** alpha
    return g


SPECTRAL_KERNELS = {
    # Tier A: no coupling (the floor) and the current additive default
    "independent": SpectralKernel(
        ("lam",), (0.0,), lambda nu, p: np.full_like(nu, 1.0 / p[0])
    ),
    "additive": SpectralKernel(
        ("lam", "eta"), (0.0, 0.0), lambda nu, p: 1.0 / (p[0] + p[1] * nu)
    ),
    # Tier B: Matern / SPDE at fixed integer alpha (fit (sigma_g, kappa); alpha on a
    # discrete grid keeps it identifiable), and the diffusion / heat kernel limit
    "matern1": SpectralKernel(("sigma_g", "kappa"), (0.0, 0.0), _matern_cov_eigs(1)),
    "matern2": SpectralKernel(("sigma_g", "kappa"), (0.0, 0.0), _matern_cov_eigs(2)),
    "matern3": SpectralKernel(("sigma_g", "kappa"), (0.0, 0.0), _matern_cov_eigs(3)),
    "diffusion": SpectralKernel(
        ("sigma_g", "t"), (0.0, -1.0),
        lambda nu, p: p[0] ** 2 * np.exp(-p[1] * nu)
    ),
}


def spectral_cov_eigs(nu: np.ndarray, kernel: str, params) -> np.ndarray:
    """Clipped covariance eigenvalues g(nu; params) for a registered kernel."""
    spec = SPECTRAL_KERNELS[kernel]
    params = np.asarray(params, dtype=float)
    if params.shape != (len(spec.param_names),):
        raise ValueError(
            f"kernel {kernel!r} takes {len(spec.param_names)} params "
            f"{spec.param_names}, got shape {params.shape}"
        )
    g = spec.cov_eigs(np.asarray(nu, dtype=float), params)
    return np.clip(g, _G_MIN, _G_MAX)


def spectral_covariance(U: np.ndarray, nu: np.ndarray, kernel: str, params) -> np.ndarray:
    """Prior covariance K = U diag(g(nu)) U^T of a registered spectral kernel."""
    g = spectral_cov_eigs(nu, kernel, params)
    return (U * g) @ U.T
