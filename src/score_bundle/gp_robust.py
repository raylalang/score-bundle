"""PROTOTYPE — Student-t timing noise for the GP-first model (future work, measured need).

Motivation (docs/graphgp_first_design.md, guard section): one confirmation piece
produced predictive NLL +27.5 on the timing channel with unremarkable RMSE and
nominal coverage — a few held-out outlier notes under tight intervals, amplified by
the Gaussian likelihood's quadratic tail. The principled fix is a heavy-tailed
observation model for tau. This module prototypes it as a Gaussian scale mixture:

    e_i ~ t_nu(0, varsigma_tau)   ==   e_i | w_i ~ N(0, varsigma_tau^2 / w_i),
    w_i ~ Gamma(nu/2, nu/2)

fit by EM on the observed block (E-step: w_i = (nu+1) / (nu + z_i^2) from the
standardized observed residuals; M-step: re-fit the GP with per-note tau noise
inflated by 1/w_i), and evaluated with a Student-t predictive density on held-out
notes (the latent variance stays Gaussian; only the noise is heavy-tailed, so the
exact predictive is not t — the t form with matched scale is the standard
approximation and is what we score).

STATUS: prototype. Not part of any published/confirmed number; dev-set-only A/B.
NumPy-only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .gp import MultiOutputGraphGP

_LOG2PI = np.log(2.0 * np.pi)


class WeightedNoiseGP(MultiOutputGraphGP):
    """MultiOutputGraphGP with a fixed per-(note, channel) noise multiplier.

    ``noise_scale[i, c] >= 1`` inflates channel ``c``'s observation noise at note
    ``i`` (the EM E-step's 1/w_i). The scale is FIXED data (not a parameter), so
    the evidence machinery is unchanged apart from the noisy diagonal.
    """

    def __init__(self, *args, noise_scale: Optional[np.ndarray] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = (np.ones((self.N, self.k)) if noise_scale is None
                            else np.asarray(noise_scale, dtype=float))

    def _noisy(self, K: np.ndarray, p: dict, idx: np.ndarray) -> np.ndarray:
        n = idx.size
        for c in range(self.k):
            K[c * n:(c + 1) * n, c * n:(c + 1) * n] += np.diag(
                p["noise"][c] * self.noise_scale[idx, c])
        return K

    def log_marginal_likelihood(self, Y, mask, x):
        p = self.unpack(x)
        obs = np.where(np.asarray(mask, dtype=bool))[0]
        if obs.size == 0:
            return 0.0
        K = self._noisy(self._blocks(p, obs, obs), p, obs)
        y = np.concatenate([np.asarray(Y, dtype=float)[obs, c] for c in range(self.k)])
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            raise np.linalg.LinAlgError("covariance not positive definite")
        return float(-0.5 * (y @ np.linalg.solve(K, y) + logdet + y.size * _LOG2PI))

    def posterior(self, Y, mask, x):
        p = self.unpack(x)
        mask = np.asarray(mask, dtype=bool)
        obs = np.where(mask)[0]
        allidx = np.arange(self.N)
        if obs.size == 0:
            var = np.tile(np.diag(p["B"]), (self.N, 1))
            return np.zeros((self.N, self.k)), np.sqrt(var)
        K_oo = self._noisy(self._blocks(p, obs, obs), p, obs)
        K_ao = self._blocks(p, allidx, obs)
        y = np.concatenate([np.asarray(Y, dtype=float)[obs, c] for c in range(self.k)])
        A = np.linalg.solve(K_oo, K_ao.T)
        m = K_ao @ np.linalg.solve(K_oo, y)
        from .gp import SHAPE_KERNELS, _G_MIN, _G_MAX
        Kg_diag = np.einsum("ij,j,ij->i", self.U, np.clip(
            SHAPE_KERNELS[self.kernel](self.nu, p["s"]), _G_MIN, _G_MAX), self.U)
        pv = np.empty(self.k * self.N)
        for c in range(self.k):
            v = p["B"][c, c] * Kg_diag
            for X, cf in zip(self.features, p["feature_scales"]):
                v = v + cf[c] * np.einsum("ij,ij->i", X, X)
            pv[c * self.N:(c + 1) * self.N] = v
        var = pv - np.einsum("ij,ji->i", K_ao, A)
        return (m.reshape(self.k, self.N).T,
                np.sqrt(np.clip(var, 0.0, None)).reshape(self.k, self.N).T)


def fit_robust_tau(nu_lap: np.ndarray, U: np.ndarray, Y: np.ndarray,
                   mask: np.ndarray, features=None, kernel: str = "additive",
                   nu_t: float = 4.0, em_iters: int = 3,
                   noise_floor: Optional[np.ndarray] = None, maxiter: int = 200,
                   tau_channel: int = 0
                   ) -> Tuple[WeightedNoiseGP, np.ndarray, np.ndarray]:
    """EM fit with Student-t noise on the tau channel only. Returns (gp, x_hat, w).

    ``w`` are the final mixture weights at observed notes (1 = Gaussian-like,
    small = downweighted outlier).
    """
    Y = np.asarray(Y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    obs = np.where(mask)[0]
    gp = WeightedNoiseGP(nu_lap, U, kernel=kernel, features=features,
                         n_channels=Y.shape[1])
    w = np.ones(obs.size)
    x_hat, _ = gp.fit(Y, mask, noise_floor=noise_floor, maxiter=maxiter)
    for _ in range(em_iters):
        # E-step: standardized observed tau residuals under the current fit
        M, S = gp.posterior(Y, mask, x_hat)
        nv = gp.unpack(x_hat)["noise"]
        r = Y[obs, tau_channel] - M[obs, tau_channel]
        s2 = S[obs, tau_channel] ** 2 + nv[tau_channel] * gp.noise_scale[obs, tau_channel]
        z2 = r ** 2 / np.maximum(s2, 1e-12)
        w = (nu_t + 1.0) / (nu_t + z2)
        # M-step: refit with inflated per-note tau noise
        scale = np.ones((gp.N, gp.k))
        scale[obs, tau_channel] = 1.0 / np.clip(w, 1e-3, None)
        gp.noise_scale = scale
        x_hat, _ = gp.fit(Y, mask, x0=x_hat, noise_floor=noise_floor,
                          maxiter=maxiter)
    return gp, x_hat, w


def t_predictive_nll(y: np.ndarray, m: np.ndarray, s: np.ndarray,
                     nu_t: float = 4.0) -> np.ndarray:
    """Per-point NLL under a Student-t predictive with matched location/scale.

    ``s`` is the Gaussian predictive std (latent + noise); the t density with the
    same scale has heavier tails, bounding the outlier penalty. Used ONLY for the
    robust variant's own scoring — Gaussian rows keep the Gaussian NLL.
    """
    from math import lgamma
    z = (np.asarray(y) - np.asarray(m)) / np.asarray(s)
    c = (lgamma((nu_t + 1) / 2) - lgamma(nu_t / 2)
         - 0.5 * np.log(nu_t * np.pi))
    return -(c - np.log(np.asarray(s)) - (nu_t + 1) / 2 * np.log1p(z ** 2 / nu_t))
