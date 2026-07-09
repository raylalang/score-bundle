"""One orthodox graph Gaussian process over (note, channel) — the GP-first model.

This module reformulates the Phase-1 model as a single multi-output graph GP in the
sense of Borovitskiy et al. (2021) and Venkitaraman et al. (2018), closing the three
orthodoxy gaps of the two-stage pipeline (docs/graphgp_first_design.md):

    1. channels are coupled by an intrinsic-coregionalization (ICM) matrix B instead
       of being three independent scalar GPs:      K = B (x) K_G  (Kronecker);
    2. the prior mean is folded INTO the kernel — a linear kernel on score features
       (and optionally LM embeddings) is exactly the marginalized Bayesian linear
       mean, so there is one model and one marginal likelihood, no plug-in head;
    3. the graph's own edge parameters (length scales, chord/voice-leading weights)
       can enter the evidence as kernel hyperparameters.

Covariance over the 3N-dimensional stacked field (channel-major blocks):

    K_total = B (x) K_G(theta_shape)  +  sum_f  diag(c_f) (x) X_f X_f^T  +  diag(nv) (x) I

where K_G = U g(nu) U^T is a *shape-normalized* spectral graph kernel (g(0) = 1, one
shape parameter; all scale lives in B), X_f are per-note feature matrices, c_f
per-channel feature-kernel scales, and nv per-channel observation noise.  Everything
is learned jointly by exact log marginal likelihood; inference is exact conjugate GP
regression.  NumPy-only; `scipy.optimize` is used when available (import-guarded).

The current published model is the special case: B diagonal, no feature kernel,
fixed graph — which is the validation gate (tests/test_graphgp.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .optimize import nelder_mead

_LOG2PI = np.log(2.0 * np.pi)

# shape-normalized spectral kernels: g(nu; s) with g(0) = 1 and ONE shape parameter s
# (overall scale deliberately lives in the coregionalization matrix B)
SHAPE_KERNELS = {
    "additive": lambda nu, s: 1.0 / (1.0 + s * nu),                # regularized Laplacian
    "matern1": lambda nu, s: (s / (s + nu)),                       # (kappa^2=s)
    "matern2": lambda nu, s: (s / (s + nu)) ** 2,
    "matern3": lambda nu, s: (s / (s + nu)) ** 3,
    "diffusion": lambda nu, s: np.exp(-s * nu),                    # heat kernel
    # ablation: NO graph coupling (K_G = I); the shape parameter is inert.  Isolates
    # the graph's marginal value inside the GP-first model.
    "none": lambda nu, s: np.ones_like(nu),
}
_G_MIN, _G_MAX = 1e-12, 1e12


def shape_cov(nu: np.ndarray, U: np.ndarray, kernel: str, s: float) -> np.ndarray:
    """K_G = U diag(g(nu; s)) U^T with g(0)=1 (unit prior variance at nu=0)."""
    g = np.clip(SHAPE_KERNELS[kernel](np.asarray(nu, dtype=float), float(s)),
                _G_MIN, _G_MAX)
    return (U * g) @ U.T


def chol_to_B(theta: np.ndarray, k: int = 3) -> np.ndarray:
    """Log-Cholesky parameterization of a k x k PSD coregionalization matrix.

    ``theta`` has k*(k+1)/2 entries: the first k are log-diagonals of the Cholesky
    factor, the rest fill the strict lower triangle row by row.
    """
    Lc = np.zeros((k, k))
    Lc[np.diag_indices(k)] = np.exp(theta[:k])
    Lc[np.tril_indices(k, -1)] = theta[k:]
    return Lc @ Lc.T


class MultiOutputGraphGP:
    """Exact multi-output graph GP with ICM coupling and optional feature kernels.

    Parameters
    ----------
    nu, U:      eigendecomposition of the graph Laplacian (np.linalg.eigh(L)).
    kernel:     shape-kernel name in :data:`SHAPE_KERNELS`.
    features:   optional list of per-note feature matrices X_f (N, d_f); each adds a
                linear kernel diag(c_f) (x) X_f X_f^T — the marginalized Bayesian
                linear mean (a bias column makes the offset Bayesian too).
    n_channels: number of output channels k (3 for [tau, log r, v]).
    """

    def __init__(self, nu: np.ndarray, U: np.ndarray, kernel: str = "additive",
                 features: Optional[Sequence[np.ndarray]] = None, n_channels: int = 3):
        self.nu = np.asarray(nu, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.kernel = kernel
        self.features = [np.asarray(X, dtype=float) for X in (features or [])]
        self.k = int(n_channels)
        self.N = self.nu.size
        self._ntri = self.k * (self.k + 1) // 2

    # --- parameter vector layout (all unconstrained reals) --------------------
    # [log shape s | B log-cholesky (ntri) | per-feature log c (k each) | log nv (k)]
    def n_params(self) -> int:
        return 1 + self._ntri + self.k * len(self.features) + self.k

    def x0(self) -> np.ndarray:
        x = np.zeros(self.n_params())
        x[-self.k:] = -2.0  # noise start, matching the established EB fits
        return x

    def unpack(self, x: np.ndarray) -> dict:
        i = 0
        s = float(np.exp(x[i])); i += 1
        B = chol_to_B(x[i:i + self._ntri], self.k); i += self._ntri
        cs = []
        for _ in self.features:
            cs.append(np.exp(x[i:i + self.k])); i += self.k
        nv = np.exp(x[i:i + self.k])
        return {"s": s, "B": B, "feature_scales": cs, "noise": nv}

    # --- covariance assembly ---------------------------------------------------
    def _blocks(self, p: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Dense covariance between (channel-major) stacked points restricted to
        note-index sets ``rows`` and ``cols`` — WITHOUT observation noise."""
        Kg = shape_cov(self.nu, self.U, self.kernel, p["s"])[np.ix_(rows, cols)]
        lin = []
        for X, c in zip(self.features, p["feature_scales"]):
            lin.append((X[rows] @ X[cols].T, c))
        nr, nc = rows.size, cols.size
        K = np.zeros((self.k * nr, self.k * nc))
        for a in range(self.k):
            for b in range(self.k):
                blk = p["B"][a, b] * Kg
                for XXt, c in lin:
                    if a == b:
                        blk = blk + c[a] * XXt
                K[a * nr:(a + 1) * nr, b * nc:(b + 1) * nc] = blk
        return K

    def log_marginal_likelihood(self, Y: np.ndarray, mask: np.ndarray,
                                x: np.ndarray) -> float:
        """log N(vec(Y_obs); 0, K_oo + noise) for parameter vector ``x``.

        ``Y`` is (N, k); ``mask`` boolean over notes (a masked note hides all its
        channels, which keeps the observed block in Kronecker-compatible form).
        """
        p = self.unpack(x)
        obs = np.where(np.asarray(mask, dtype=bool))[0]
        if obs.size == 0:
            return 0.0
        K = self._blocks(p, obs, obs)
        n_o = obs.size
        for c in range(self.k):
            K[c * n_o:(c + 1) * n_o, c * n_o:(c + 1) * n_o] += p["noise"][c] * np.eye(n_o)
        y = np.concatenate([Y[obs, c] for c in range(self.k)])
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            raise np.linalg.LinAlgError("covariance not positive definite")
        alpha = np.linalg.solve(K, y)
        return float(-0.5 * (y @ alpha + logdet + y.size * _LOG2PI))

    def posterior(self, Y: np.ndarray, mask: np.ndarray, x: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact conjugate posterior at ALL notes: mean and latent std, (N, k) each.

        Predictive std for a held-out observation is sqrt(std**2 + noise_c) — the
        caller adds the channel noise, mirroring the established pipeline.
        """
        p = self.unpack(x)
        mask = np.asarray(mask, dtype=bool)
        obs = np.where(mask)[0]
        allidx = np.arange(self.N)
        if obs.size == 0:
            var = np.tile(np.diag(p["B"]), (self.N, 1))
            return np.zeros((self.N, self.k)), np.sqrt(var)
        K_oo = self._blocks(p, obs, obs)
        n_o = obs.size
        for c in range(self.k):
            K_oo[c * n_o:(c + 1) * n_o, c * n_o:(c + 1) * n_o] += p["noise"][c] * np.eye(n_o)
        K_ao = self._blocks(p, allidx, obs)
        y = np.concatenate([Y[obs, c] for c in range(self.k)])
        A = np.linalg.solve(K_oo, K_ao.T)          # (k n_o, k N)
        m = K_ao @ np.linalg.solve(K_oo, y)
        # prior variance diag per (channel, note): B_cc * Kg_ii + sum_f c_f[c] * ||x_i||^2
        Kg_diag = np.einsum("ij,j,ij->i", self.U, np.clip(
            SHAPE_KERNELS[self.kernel](self.nu, p["s"]), _G_MIN, _G_MAX), self.U)
        pv = np.empty(self.k * self.N)
        for c in range(self.k):
            v = p["B"][c, c] * Kg_diag
            for X, cf in zip(self.features, p["feature_scales"]):
                v = v + cf[c] * np.einsum("ij,ij->i", X, X)
            pv[c * self.N:(c + 1) * self.N] = v
        var = pv - np.einsum("ij,ji->i", K_ao, A)
        m = m.reshape(self.k, self.N).T
        std = np.sqrt(np.clip(var, 0.0, None)).reshape(self.k, self.N).T
        return m, std

    def loo_predictive(self, Y: np.ndarray, x: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Leave-one-out predictive N(mean, var) of each fully-observed observation.

        Joint-GP version of :func:`downstream.loo_predictive`: with the full
        observation covariance C (all notes, all channels, plus noise) and P = C^-1,
        ``mean_i = y_i - (P r)_i / P_ii`` and ``var_i = 1 / P_ii`` (variance of the
        noisy observation — no extra floor needed).  Returns (N, k) arrays.
        """
        p = self.unpack(x)
        allidx = np.arange(self.N)
        C = self._blocks(p, allidx, allidx)
        for c in range(self.k):
            C[c * self.N:(c + 1) * self.N, c * self.N:(c + 1) * self.N] += \
                p["noise"][c] * np.eye(self.N)
        P = np.linalg.inv(C)
        y = np.concatenate([np.asarray(Y, dtype=float)[:, c] for c in range(self.k)])
        Pr = P @ y  # zero prior mean
        dii = np.clip(np.diag(P), 1e-12, None)
        loo_mean = y - Pr / dii
        loo_var = 1.0 / dii
        return (loo_mean.reshape(self.k, self.N).T,
                loo_var.reshape(self.k, self.N).T)

    # --- fitting -----------------------------------------------------------------
    def fit(self, Y: np.ndarray, mask: np.ndarray, x0: Optional[np.ndarray] = None,
            noise_floor: Optional[np.ndarray] = None, maxiter: int = 300,
            noise_fixed: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, dict]:
        """Maximize the exact marginal likelihood over ALL parameters jointly.

        ``noise_floor`` (length k, variances) clamps the per-channel noise inside
        the objective and in the returned parameters — same principle as the
        established EB noise floor.  Uses scipy L-BFGS-B when available, else the
        dependency-free Nelder–Mead.  Returns (x_hat, info dict).
        """
        Y = np.asarray(Y, dtype=float)
        x0 = self.x0() if x0 is None else np.asarray(x0, dtype=float)
        floor_log = None
        if noise_floor is not None:
            floor_log = np.log(np.maximum(np.asarray(noise_floor, dtype=float), 1e-12))
        fixed_log = None
        if noise_fixed is not None:
            # oracle-noise setting (e.g. denoising with a known level): the noise
            # coordinates are pinned, everything else is still learned by evidence
            fixed_log = np.log(np.maximum(np.asarray(noise_fixed, dtype=float), 1e-12))

        def clamp(x: np.ndarray) -> np.ndarray:
            z = x.copy()
            if fixed_log is not None:
                z[-self.k:] = fixed_log
            elif floor_log is not None:
                z[-self.k:] = np.maximum(z[-self.k:], floor_log)
            return z

        def neg(x: np.ndarray) -> float:
            try:
                v = -self.log_marginal_likelihood(Y, mask, clamp(x))
            except (np.linalg.LinAlgError, ValueError):
                return 1e12
            # a NaN objective is not an exception but poisons L-BFGS line search
            return v if np.isfinite(v) else 1e12

        used = "nelder_mead"
        best = None
        try:
            from scipy.optimize import minimize  # optional dependency (CLAUDE.md)
            res = minimize(neg, x0, method="L-BFGS-B",
                           options={"maxiter": maxiter, "eps": 1e-5})
            best, used = res.x, "lbfgs"
        except ImportError:
            pass
        if best is None:
            best = nelder_mead(neg, x0, max_iter=1200)
        # one Nelder-Mead polish from the L-BFGS point costs little and guards
        # against finite-difference stalls on the log-Cholesky coordinates
        polished = nelder_mead(neg, best, max_iter=300)
        if neg(polished) < neg(best):
            best = polished
        best = clamp(best)
        return best, {"optimizer": used, "nll": float(neg(best)),
                      **{k2: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k2, v in self.unpack(best).items()
                         if k2 in ("s", "noise")}}

    def fit_guarded(self, Y: np.ndarray, mask: np.ndarray,
                    noise_floor: Optional[np.ndarray] = None,
                    calib_frac: float = 0.3, guard_factor: float = 2.0,
                    nll_margin: float = 0.5,
                    rng: Optional[np.random.Generator] = None,
                    maxiter: int = 300) -> Tuple[np.ndarray, dict]:
        """Guarded evidence fit — the GP-first version of the published EB guard.

        The demonstrated failure mode (one confirmation piece, NLL +2.2) is
        *overconfidence*: the per-piece evidence picks a noise/scale regime whose
        held-out intervals are far too tight, while RMSE stays unremarkable.  So the
        screen checks BOTH: on a held-back calibration split of the observed notes,
        the fitted GP must (i) not exceed ``guard_factor`` x the mean-only RMSE and
        (ii) not exceed the mean-only homoscedastic NLL by more than ``nll_margin``
        nats.  Ladder, recorded in ``info["guard"]``:

            "marglik"       screen passed;
            "floored"       refit with the noise floor raised x5 passed;
            "conservative"  no-coupling, no-feature diagonal at the observed
                            per-channel variance (predict ~the mean with honest
                            scale — the same bottom rung as the published guard).
        """
        Y = np.asarray(Y, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if rng is None:
            rng = np.random.default_rng(0)
        obs = np.where(mask)[0]
        floor = (np.asarray(noise_floor, dtype=float) if noise_floor is not None
                 else np.full(self.k, 1e-12))

        x_ml, info = self.fit(Y, mask, noise_floor=floor, maxiter=maxiter)
        n_calib = int(round(calib_frac * obs.size))
        if obs.size < 8 or n_calib < 2 or obs.size - n_calib < 2:
            return x_ml, {**info, "guard": "marglik"}

        perm = rng.permutation(obs)
        calib = perm[:n_calib]
        fit_mask = np.zeros(self.N, dtype=bool)
        fit_mask[perm[n_calib:]] = True

        # mean-only reference on the calibration subset: per-channel mean/std of the
        # FIT subset (never the calibration targets themselves)
        base_mu = Y[fit_mask].mean(axis=0)
        base_sd = np.maximum(Y[fit_mask].std(axis=0), 1e-9)
        r0 = Y[calib] - base_mu
        base_rmse = float(np.sqrt(np.mean(r0 ** 2)))
        base_nll = float(np.mean(0.5 * (_LOG2PI + 2 * np.log(base_sd)
                                        + (r0 / base_sd) ** 2)))

        def screen(x) -> bool:
            try:
                M, S = self.posterior(Y, fit_mask, x)
            except (np.linalg.LinAlgError, ValueError):
                return False
            nv = self.unpack(x)["noise"]
            r = Y[calib] - M[calib]
            pv = S[calib] ** 2 + nv[None, :]
            rmse = float(np.sqrt(np.mean(r ** 2)))
            nll = float(np.mean(0.5 * (_LOG2PI + np.log(pv) + r ** 2 / pv)))
            return (rmse <= guard_factor * max(base_rmse, 1e-12)
                    and nll <= base_nll + nll_margin)

        if screen(x_ml):
            return x_ml, {**info, "guard": "marglik"}

        x_fl, info_fl = self.fit(Y, mask, noise_floor=5.0 * floor, maxiter=maxiter)
        if screen(x_fl):
            return x_fl, {**info_fl, "guard": "floored"}

        resid_var = np.maximum(Y[obs].var(axis=0), 1e-12)
        x_c = self.x0()
        # decouple the graph: g(nu) -> 1 for every nu.  The limit direction depends
        # on the shape family: additive/diffusion decouple as s -> 0, Matern as
        # s -> infinity ("none" is already decoupled).
        x_c[0] = np.log(1e-8) if self.kernel in ("additive", "diffusion", "none") \
            else np.log(1e8)
        x_c[1:1 + self.k] = 0.5 * np.log(resid_var)     # B = diag(observed variance)
        x_c[1 + self.k:1 + self._ntri] = 0.0
        for i in range(len(self.features)):             # feature kernels off
            base_i = 1 + self._ntri + i * self.k
            x_c[base_i:base_i + self.k] = np.log(1e-8)
        x_c[-self.k:] = np.log(np.maximum(0.05 * resid_var, floor))
        return x_c, {"optimizer": "conservative", "guard": "conservative"}
