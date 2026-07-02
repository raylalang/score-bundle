"""Phase-1 structured model: a Gaussian graph field over a per-note variable.

Given a prior precision Q (built from the score graph) and noisy observations
    y_tilde = y + e,   e ~ N(0, Sigma_e),
the posterior over the latent performance field y is closed-form Gaussian:

    p(y | y_tilde) = N(m, Sigma_y)
    Sigma_y = (Q + Sigma_e^{-1})^{-1}
    m = Sigma_y (Q mu + Sigma_e^{-1} y_tilde).

Unobserved (masked) notes simply carry no likelihood term, so they are predicted
from their neighbours through the conditional Gaussian -- this is the held-out
imputation task used to test whether the graph prior helps.

NumPy dense implementation for clarity and exactness; the production path uses the
sparse GMRF precision with sparse Cholesky / selected inversion.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .optimize import nelder_mead
from .prior import laplacian_precision

_LOG2PI = np.log(2.0 * np.pi)


def _as_vec(value, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"expected scalar or length-{n} vector, got shape {arr.shape}")
    return arr


class GraphGaussianField:
    """A Gaussian field y ~ N(mu, Q^{-1}) over the N score nodes (one channel)."""

    def __init__(self, Q: np.ndarray, mean=0.0):
        self.Q = np.asarray(Q, dtype=float)
        self.N = self.Q.shape[0]
        self.mean = _as_vec(mean, self.N)

    # --- generative -------------------------------------------------------
    def covariance(self) -> np.ndarray:
        return np.linalg.inv(self.Q)

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples from the prior. Returns shape (N,) if size==1 else (N, size)."""
        K = self.covariance()
        L = np.linalg.cholesky(K + 1e-12 * np.eye(self.N))
        z = rng.standard_normal((self.N, size))
        out = self.mean[:, None] + L @ z
        return out[:, 0] if size == 1 else out

    # --- inference --------------------------------------------------------
    def posterior(
        self,
        y_obs: np.ndarray,
        noise_var,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Closed-form posterior mean and per-note std.

        Parameters
        ----------
        y_obs:     observed targets (length N); values at masked nodes are ignored.
        noise_var: observation variance Sigma_e (scalar or length-N vector).
        mask:      boolean array, True where a note is observed (default: all).
        """
        y_obs = np.asarray(y_obs, dtype=float)
        if mask is None:
            mask = np.ones(self.N, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        nv = _as_vec(noise_var, self.N)

        d = np.where(mask, 1.0 / nv, 0.0)  # observation precision per node
        y_filled = np.where(mask, y_obs, 0.0)

        A = self.Q + np.diag(d)
        Sigma_y = np.linalg.inv(A)  # posterior covariance (note: Σ_y)
        m = Sigma_y @ (self.Q @ self.mean + d * y_filled)
        std = np.sqrt(np.clip(np.diag(Sigma_y), 0.0, None))
        return m, std

    def log_marginal_likelihood(
        self,
        y_obs: np.ndarray,
        noise_var,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """log p(y_obs_observed | theta) = N(y_o; mu_o, K_oo + Sigma_e)."""
        y_obs = np.asarray(y_obs, dtype=float)
        if mask is None:
            mask = np.ones(self.N, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return 0.0
        nv = _as_vec(noise_var, self.N)[idx]

        K = self.covariance()
        C = K[np.ix_(idx, idx)] + np.diag(nv)
        r = y_obs[idx] - self.mean[idx]

        sign, logdet = np.linalg.slogdet(C)
        Cinv_r = np.linalg.solve(C, r)
        n = idx.size
        return float(-0.5 * (r @ Cinv_r + logdet + n * _LOG2PI))


def fit_laplacian_field(
    L: np.ndarray,
    y_obs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mean=0.0,
    x0=(0.0, 0.0, -2.0),
    noise_floor: float = 0.0,
) -> Tuple[GraphGaussianField, dict]:
    """Empirical-Bayes fit of (lam, eta, noise_var) by marginal likelihood.

    ``lam`` and ``eta`` are the additive-form precision parameters Q_G = lam I + eta L.
    Returns the fitted :class:`GraphGaussianField` and a dict of hyperparameters.
    Parameters are optimized in log-space to keep them positive.

    ``noise_floor`` (an absolute variance, default 0 = off) clamps ``noise_var`` from
    below *inside the objective* and in the returned hyperparameters.  On a minority
    of mask realizations the unconstrained maximizer degenerates (noise_var -> 0,
    an overconfident posterior whose held-out NLL blows up); the floor is the same
    principle as the predictive-variance floor in ``imputation_eval._predict_channel``
    applied to the fit itself.  Callers typically pass a small fraction of the
    observed residual variance, e.g. ``0.05 * np.var((y - mean)[mask])``.
    """
    L = np.asarray(L, dtype=float)

    def neg_lml(logp: np.ndarray) -> float:
        lam, eta, nv = np.exp(logp)
        nv = max(nv, noise_floor)
        try:
            Q = laplacian_precision(L, lam=lam, eta=eta)
            field = GraphGaussianField(Q, mean=mean)
            return -field.log_marginal_likelihood(y_obs, nv, mask)
        except (np.linalg.LinAlgError, ValueError):
            return 1e12

    best = nelder_mead(neg_lml, np.asarray(x0, dtype=float))
    lam, eta, noise_var = np.exp(best)
    noise_var = max(noise_var, noise_floor)
    Q = laplacian_precision(L, lam=lam, eta=eta)
    field = GraphGaussianField(Q, mean=mean)
    return field, {"lam": float(lam), "eta": float(eta), "noise_var": float(noise_var)}


def fit_laplacian_field_calib(
    L: np.ndarray,
    y_obs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mean=0.0,
    calib_frac: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    x0=(0.0, 0.0, -2.0),
) -> Tuple[GraphGaussianField, dict]:
    """Pick (lam, eta, noise_var) by held-out predictive NLL on a *calibration split*.

    Unlike :func:`fit_laplacian_field` (which maximizes the in-sample marginal likelihood of
    all observed nodes), this splits the observed nodes into a fit subset and a calibration
    subset, then chooses the hyperparameters that minimize the predictive NLL of the
    calibration nodes given the fit nodes -- using the same observation-noise-floored
    predictive std (``diag(Sigma_y) + noise_var``) the eval scores against.  This optimizes
    *calibration* directly rather than in-sample fit, so coverage lands closer to nominal.

    Falls back to the marginal-likelihood fit if there are too few observed nodes to split.
    """
    L = np.asarray(L, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    n = L.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if rng is None:
        rng = np.random.default_rng(0)

    obs_idx = np.where(mask)[0]
    n_calib = int(round(calib_frac * obs_idx.size))
    if obs_idx.size < 4 or n_calib < 1 or obs_idx.size - n_calib < 1:
        return fit_laplacian_field(L, y_obs, mask=mask, mean=mean, x0=x0)

    perm = rng.permutation(obs_idx)
    calib_idx = perm[:n_calib]
    fit_idx = perm[n_calib:]
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[fit_idx] = True

    def neg_calib_nll(logp: np.ndarray) -> float:
        lam, eta, nv = np.exp(logp)
        try:
            Q = laplacian_precision(L, lam=lam, eta=eta)
            field = GraphGaussianField(Q, mean=mean)
            m, std = field.posterior(y_obs, nv, mask=fit_mask)
        except (np.linalg.LinAlgError, ValueError):
            return 1e12
        pred_var = std[calib_idx] ** 2 + nv  # observation-noise-floored predictive variance
        r = y_obs[calib_idx] - m[calib_idx]
        nll = 0.5 * (_LOG2PI + np.log(pred_var) + r ** 2 / pred_var)
        return float(np.mean(nll))

    best = nelder_mead(neg_calib_nll, np.asarray(x0, dtype=float))
    lam, eta, noise_var = np.exp(best)
    # refit the field on ALL observed nodes with the chosen hyperparameters
    Q = laplacian_precision(L, lam=lam, eta=eta)
    field = GraphGaussianField(Q, mean=mean)
    return field, {"lam": float(lam), "eta": float(eta), "noise_var": float(noise_var)}
