"""Spectral-mixture GP estimator for the within-note cents curve.

The "estimator v2" candidate of docs/kernel_papers_review.md (Slot A) and
docs/sm_estimator_note.tex: replace the parametric sine fit
(:func:`intonation.fit_vibrato_note`) by exact GP regression under a
two-component spectral-mixture (SM) kernel (Wilson & Adams, ICML 2013),

    k(tau) = w1 * exp(-2 pi^2 v1 tau^2) * cos(2 pi mu1 tau)   [vibrato]
           + w2 * exp(-2 pi^2 v2 tau^2)                        [drift]

with iid tracker noise s2 on the diagonal and a flat prior on the constant
centre c, marginalized in closed form.  Read-outs (sm_estimator_note.tex §4):
c = GLS posterior mean (exact variance given the fitted kernel);
f = mu1, gamma = sqrt(2 w1), with Laplace variances from the curvature of
the log evidence.  The sine fit is the degenerate sub-model v1 -> 0, w2 = 0
with phase treated as known; here phase never enters (the kernel reads the
oscillation from correlations).

Development-only study code path: nothing here touches the registered
Phase-2 pipeline or its frozen confirmation.  numpy-only, deterministic
(grid + Nelder-Mead from :mod:`score_bundle.optimize`; no RNG).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..optimize import nelder_mead

_SENTINEL = 1e12  # non-finite / out-of-band objective value (cf. gp.py)


def sm_kernel(tau: np.ndarray, w1: float, mu1: float, v1: float,
              w2: float, v2: float) -> np.ndarray:
    """The two-component SM kernel k(tau) (latent part, no noise)."""
    tau = np.asarray(tau, dtype=float)
    return (w1 * np.exp(-2.0 * np.pi ** 2 * v1 * tau ** 2)
            * np.cos(2.0 * np.pi * mu1 * tau)
            + w2 * np.exp(-2.0 * np.pi ** 2 * v2 * tau ** 2))


def _unpack(p: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Parameter vector [log w1, mu1(Hz), log v1, log w2, log v2, log s2]."""
    return (float(np.exp(p[0])), float(p[1]), float(np.exp(p[2])),
            float(np.exp(p[3])), float(np.exp(p[4])), float(np.exp(p[5])))


def _chol_terms(t: np.ndarray, y: np.ndarray, p: np.ndarray):
    """Cholesky pieces of K = k(dt) + s2 I; None on failure."""
    w1, mu1, v1, w2, v2, s2 = _unpack(p)
    if not np.all(np.isfinite([w1, mu1, v1, w2, v2, s2])):
        return None
    tau = t[:, None] - t[None, :]
    K = sm_kernel(tau, w1, mu1, v1, w2, v2)
    K[np.diag_indices_from(K)] = w1 + w2 + s2 + 1e-10 * (w1 + w2 + s2)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return None
    one = np.ones(t.size)
    a1 = np.linalg.solve(L, one)
    ay = np.linalg.solve(L, y)
    a = float(a1 @ a1)                       # 1' K^-1 1
    b = float(a1 @ ay)                       # 1' K^-1 y
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return L, a, b, ay, logdet


def sm_log_evidence(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Flat-prior-c marginal log likelihood (sm_estimator_note.tex eq. 6).

    log int N(y; c 1, K) dc = -(n-1)/2 log 2pi - 1/2 log|K|
                              - 1/2 r' K^-1 r - 1/2 log(1' K^-1 1),
    with r = y - c_hat 1 and c_hat the GLS mean.  Returns -_SENTINEL on a
    failed Cholesky or out-of-range parameters.
    """
    terms = _chol_terms(np.asarray(t, float), np.asarray(y, float), p)
    if terms is None:
        return -_SENTINEL
    _, a, b, ay, logdet = terms
    n = np.asarray(y).size
    quad = float(ay @ ay) - b * b / a       # r' K^-1 r
    val = (-0.5 * (n - 1) * np.log(2.0 * np.pi) - 0.5 * logdet
           - 0.5 * quad - 0.5 * np.log(a))
    return val if np.isfinite(val) else -_SENTINEL


def _decimate(t: np.ndarray, y: np.ndarray, n_max: int):
    if t.size <= n_max:
        return t, y
    idx = np.linspace(0, t.size - 1, n_max).round().astype(int)
    return t[idx], y[idx]


def fit_sm_note(t: np.ndarray, cents: np.ndarray,
                f_lo: float = 2.5, f_hi: float = 9.0,
                n_grid: int = 33, n_max: int = 400,
                nm_iter: int = 400) -> Dict[str, float]:
    """Fit the SM-GP to one note's voiced cents frames.

    Procedure (sm_estimator_note.tex §3): a coarse grid over the vibrato
    frequency mu1 in [f_lo, f_hi] with moment-based inner initializations
    (the evidence is multimodal in mu1), then Nelder-Mead over all six
    parameters in log space (mu1 natural, soft-barriered to the band), then
    a central-difference Hessian of the log evidence for Laplace variances.
    Frames are uniformly thinned to ``n_max`` for the fit.

    Returns a dict: ``c, gamma, f, var_c, var_gamma, var_f, var_log_gamma,
    var_log_f, evidence, n, wide`` -- ``wide`` flags a non-PD Hessian
    (variances set to inf); the estimator never refuses a note with
    >= 6 frames.  Mirrors :func:`intonation.fit_vibrato_note`'s conventions
    so the two are comparable cell for cell.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(cents, dtype=float)
    n_full = int(t.size)
    if n_full < 6:
        c = float(y.mean()) if n_full else 0.0
        v = float(y.var(ddof=1) / n_full) if n_full > 1 else np.inf
        return {"c": c, "gamma": 0.0, "f": 0.0, "var_c": v,
                "var_gamma": np.inf, "var_f": np.inf,
                "var_log_gamma": np.inf, "var_log_f": np.inf,
                "evidence": -np.inf, "n": n_full, "wide": True,
                "params": None}
    tf, yf = _decimate(t, y, n_max)

    # --- identifiability bounds (they break a real degeneracy) --------
    # A drift component with v2 -> 0 is a pure constant, exactly redundant
    # with the flat-prior centre c: the evidence is flat in that direction
    # and w2 can run away, destroying var_c and the Hessian.  A "drift"
    # must vary within the note -- the constant belongs to c -- so v2 is
    # bounded below such that the component decorrelates by at least
    # exp(-1/2) over the note span T (2 pi^2 v2 T^2 >= 1/2).  Generous
    # upper caps on the bandwidths and powers guard the other flat
    # directions (a component faster than the frame rate is noise; power
    # far beyond the data variance explains nothing).
    vy = max(float(y.var(ddof=1)), 1e-6)
    span = max(float(t.max() - t.min()), 1e-3)
    v2_min = 0.025 / span ** 2
    v_max = 25.0
    w_max = 1e3 * vy

    def neg(p: np.ndarray) -> float:
        w1, mu1, v1, w2, v2, s2 = _unpack(p)
        if not (f_lo - 0.5 <= mu1 <= f_hi + 0.5):
            return _SENTINEL
        if v2 < v2_min or v2 > v_max or v1 > v_max:
            return _SENTINEL
        if w1 > w_max or w2 > w_max or s2 > 10.0 * vy or s2 < 1e-4 * vy:
            return _SENTINEL
        return -sm_log_evidence(tf, yf, p)

    # --- moment-based initial guesses ---------------------------------
    s2_0 = float(np.clip(np.median(np.diff(y) ** 2) / 2.0,
                         1e-3 * vy, vy)) if n_full > 2 else 0.5 * vy

    ts, ys = _decimate(t, y, 150)            # cheap scan frames

    def amp2_at(f: float) -> float:
        th = 2.0 * np.pi * f * ts
        A = np.stack([np.ones(ts.size), np.sin(th), np.cos(th)], axis=1)
        beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
        return float(beta[1] ** 2 + beta[2] ** 2)

    v2_0 = max(0.1, 2.0 * v2_min)
    grid = np.linspace(f_lo, f_hi, n_grid)
    best = (np.inf, None)
    for f in grid:
        w1_0 = float(np.clip(amp2_at(f) / 2.0, 1e-3 * vy, 0.5 * w_max))
        w2_0 = float(np.clip(vy - w1_0 - s2_0, 0.05 * vy, 0.5 * w_max))
        p0 = np.array([np.log(w1_0), f, np.log(0.25),
                       np.log(w2_0), np.log(v2_0), np.log(s2_0)])
        val = neg(p0)
        if val < best[0]:
            best = (val, p0)
    if best[1] is None:                       # pragma: no cover - defensive
        best = (np.inf, np.array([np.log(vy / 2), 0.5 * (f_lo + f_hi),
                                  np.log(0.25), np.log(vy / 4),
                                  np.log(v2_0), np.log(s2_0)]))

    p_hat = nelder_mead(neg, best[1], step=0.4, max_iter=nm_iter)
    p_hat = nelder_mead(neg, p_hat, step=0.1, max_iter=nm_iter // 2)
    ev = -neg(p_hat)

    # --- Laplace: central-difference Hessian of -log evidence ---------
    h = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.05])
    m = p_hat.size
    H = np.zeros((m, m))
    f0 = neg(p_hat)
    at_boundary = [f0 >= 0.5 * _SENTINEL]

    def ev_at(dp):
        val = neg(p_hat + dp)
        if val >= 0.5 * _SENTINEL:            # FD stencil crossed a bound:
            at_boundary[0] = True             # curvature is meaningless there
        return val

    for i in range(m):
        ei = np.zeros(m)
        ei[i] = h[i]
        H[i, i] = (ev_at(ei) - 2 * f0 + ev_at(-ei)) / h[i] ** 2
        for j in range(i + 1, m):
            ej = np.zeros(m)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                ev_at(ei + ej) - ev_at(ei - ej)
                - ev_at(-ei + ej) + ev_at(-ei - ej)
            ) / (4 * h[i] * h[j])

    wide = bool(at_boundary[0])
    if not wide:
        try:
            Sig = np.linalg.inv(H)
            d = np.diag(Sig)
            if not np.all(np.isfinite(d)) or np.any(d[[0, 1]] <= 0):
                wide = True
        except np.linalg.LinAlgError:
            wide = True
    if wide:
        var_log_w1 = np.inf
        var_mu1 = np.inf
    else:
        var_log_w1 = float(Sig[0, 0])
        var_mu1 = float(Sig[1, 1])

    w1, mu1, v1, w2, v2, s2 = _unpack(p_hat)

    # exact GLS centre on the FULL frames at the fitted kernel, and the
    # realized-amplitude extent read-out.  The channel's estimand is THIS
    # note's realized vibrato amplitude (what the sine fit reports), not
    # the ensemble scale sqrt(2 w1): for a near-coherent component w1 is
    # informed by only ~2 effective degrees of freedom (the realized
    # sin/cos amplitudes), so sqrt(2 w1) is chi^2_2-spread around the
    # realized value (median ~0.83 of it).  The posterior mean of the
    # vibrato component m1 = K1 K^-1 (y - c 1) recovers the realized
    # oscillation directly; its empirical std over the frames times
    # sqrt(2) is the amplitude (shrunk toward 0 only when the evidence
    # for the component is genuinely weak).
    terms = _chol_terms(t, y, p_hat)
    if terms is None:                         # pragma: no cover - defensive
        c_hat, var_c = float(y.mean()), float(y.var(ddof=1) / n_full)
        gamma = float(np.sqrt(2.0 * w1))
    else:
        L, a, b, _, _ = terms
        c_hat, var_c = b / a, 1.0 / a
        tau_full = t[:, None] - t[None, :]
        K1 = (w1 * np.exp(-2.0 * np.pi ** 2 * v1 * tau_full ** 2)
              * np.cos(2.0 * np.pi * mu1 * tau_full))
        r1 = np.linalg.solve(L, y - c_hat)
        r2 = np.linalg.solve(L.T, r1)         # K^-1 (y - c 1)
        m1 = K1 @ r2
        gamma = float(np.sqrt(2.0) * m1.std())

    var_log_gamma = 0.25 * var_log_w1        # scale uncertainty from w1
    var_gamma = gamma ** 2 * var_log_gamma
    var_f = var_mu1
    var_log_f = var_mu1 / max(mu1 ** 2, 1e-12)

    return {"c": float(c_hat), "gamma": gamma, "f": mu1,
            "var_c": float(var_c), "var_gamma": float(var_gamma),
            "var_f": float(var_f), "var_log_gamma": float(var_log_gamma),
            "var_log_f": float(var_log_f), "evidence": float(ev),
            "n": n_full, "wide": bool(wide),
            "params": p_hat.copy()}


def sm_predict(t_fit: np.ndarray, y_fit: np.ndarray, params: np.ndarray,
               t_new: np.ndarray, include_noise: bool = True
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean and variance of the cents curve at ``t_new``.

    Universal-kriging form under the flat-prior constant mean
    (sm_estimator_note.tex §4): mean = c_hat + k*' K^-1 (y - c_hat 1),
    var = k(0) - k*' K^-1 k* + (1 - k*' K^-1 1)^2 / (1' K^-1 1),
    plus the fitted noise s2 when ``include_noise`` (predicting an
    observable frame value rather than the latent curve).
    """
    t_fit = np.asarray(t_fit, float)
    y_fit = np.asarray(y_fit, float)
    t_new = np.asarray(t_new, float)
    w1, mu1, v1, w2, v2, s2 = _unpack(params)
    terms = _chol_terms(t_fit, y_fit, params)
    if terms is None:
        m = np.full(t_new.size, float(y_fit.mean()))
        v = np.full(t_new.size, float(y_fit.var(ddof=1)) + s2)
        return m, v
    L, a, b, _, _ = terms
    c_hat = b / a
    Ks = sm_kernel(t_new[:, None] - t_fit[None, :], w1, mu1, v1, w2, v2)
    A = np.linalg.solve(L, Ks.T)             # L^-1 k*
    r = np.linalg.solve(L, y_fit - c_hat)
    one = np.linalg.solve(L, np.ones(t_fit.size))
    mean = c_hat + A.T @ r
    var = (w1 + w2) - np.einsum("ij,ij->j", A, A)
    var = var + (1.0 - A.T @ one) ** 2 / a
    var = np.maximum(var, 1e-12)
    if include_noise:
        var = var + s2
    return mean, var
