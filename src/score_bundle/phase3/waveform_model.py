"""Waveform likelihood with exact amplitude marginalization (Phase 3).

Given the nonlinear positions z (here summarized by the design matrix Phi = Phi(z))
and a Gaussian amplitude prior a ~ N(mu_a, Sigma_a), the amplitude posterior is
closed-form and the amplitude-collapsed likelihood is Gaussian:

    p(a | x, z) = N(m_a, S_a),  S_a = (Sigma_a^{-1} + Phi^T K_n^{-1} Phi)^{-1}
    p(x | z)    = N(x; Phi mu_a, Phi Sigma_a Phi^T + K_n).

What remains (the open research work) is inference over the nonlinear z: a Laplace
approximation or per-performance VI, initialized from the nominal score f0 to handle
multimodality.  That outer loop is left as a documented stub.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_LOG2PI = np.log(2.0 * np.pi)


def amplitude_posterior(
    x: np.ndarray,
    Phi: np.ndarray,
    Sigma_a: np.ndarray,
    mu_a: Optional[np.ndarray] = None,
    noise_var: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form Gaussian posterior over the harmonic amplitudes a."""
    x = np.asarray(x, dtype=float)
    Phi = np.asarray(Phi, dtype=float)
    p = Phi.shape[1]
    if mu_a is None:
        mu_a = np.zeros(p)
    Kn_inv = np.eye(Phi.shape[0]) / float(noise_var)
    Sa_inv = np.linalg.inv(Sigma_a) + Phi.T @ Kn_inv @ Phi
    S_a = np.linalg.inv(Sa_inv)
    m_a = S_a @ (np.linalg.solve(Sigma_a, mu_a) + Phi.T @ Kn_inv @ x)
    return m_a, S_a


def collapsed_loglik(
    x: np.ndarray,
    Phi: np.ndarray,
    Sigma_a: np.ndarray,
    mu_a: Optional[np.ndarray] = None,
    noise_var: float = 1.0,
) -> float:
    """log p(x | z) with the amplitudes marginalized out."""
    x = np.asarray(x, dtype=float)
    Phi = np.asarray(Phi, dtype=float)
    m = Phi.shape[0]
    if mu_a is None:
        mu_a = np.zeros(Phi.shape[1])
    C = Phi @ Sigma_a @ Phi.T + noise_var * np.eye(m)
    r = x - Phi @ mu_a
    sign, logdet = np.linalg.slogdet(C)
    return float(-0.5 * (r @ np.linalg.solve(C, r) + logdet + m * _LOG2PI))


def infer_positions(*args, **kwargs):  # pragma: no cover - research stub
    """Inverse inference over the nonlinear positions z (Laplace / VI).

    TODO: build Phi(z) from score_bundle.phase3.synth, marginalize amplitudes via
    `amplitude_posterior`, optimize `collapsed_loglik(x, Phi(z)) + log p(z)` over z
    with a Gauss-Newton/Laplace approximation, initialized from the nominal score f0.
    """
    raise NotImplementedError(
        "Phase-3 position inference is the open research step; see the concept note, "
        "section 9 (Phase 3) and the docstring above."
    )
