"""Intonation / vibrato feature extraction (Phase 2).

`cents_from_f0` and `vibrato_from_f0` are implementable helpers; `extract_f0` is a
stub to be backed by a pitch tracker (e.g. CREPE/pYIN) on monophonic audio.

NB `vibrato_from_f0` is a crude starting point and is NOT the estimator the thesis
specifies (draft eq:vibrato): it mean-removes the cents curve (the thesis notes the
vibrato-free centre is *not* the mean) and reads rate/extent from FFT/RMS with no
onset delay; the specified estimator is a joint per-note nonlinear least-squares
fit of (c_i, f_i^vib, gamma_i, delta_i^vib) on voiced samples.

Once a per-note intonation field ``c`` (cents) is available, it plugs into the
thesis model as an extra channel of :class:`score_bundle.gp.MultiOutputGraphGP`
(the two-stage :class:`score_bundle.model.GraphGaussianField` route remains the
per-channel special case).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def cents_from_f0(f0: np.ndarray, f_ref: float, semitone: float) -> np.ndarray:
    """Deviation in cents of f0 from the equal-tempered target pitch.

    target = f_ref * 2**(semitone/12);  cents = 1200 * log2(f0 / target).
    """
    target = f_ref * 2.0 ** (semitone / 12.0)
    f0 = np.clip(np.asarray(f0, dtype=float), 1e-6, None)
    return 1200.0 * np.log2(f0 / target)


def vibrato_from_f0(cents_curve: np.ndarray, sr: float) -> Dict[str, float]:
    """Crude vibrato descriptors (rate Hz, extent cents) from a cents curve.

    Rate is the dominant frequency of the mean-removed curve; extent is its RMS
    amplitude (scaled to a peak estimate).  Replace with a robust estimator for
    real use; this is a starting point, not ground truth.
    """
    x = np.asarray(cents_curve, dtype=float)
    x = x - x.mean()
    if x.size < 4:
        return {"rate_hz": 0.0, "extent_cents": float(np.sqrt(np.mean(x ** 2)))}
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    spec[0] = 0.0
    rate = float(freqs[int(np.argmax(spec))])
    extent = float(np.sqrt(2.0) * np.sqrt(np.mean(x ** 2)))
    return {"rate_hz": rate, "extent_cents": extent}




def fit_vibrato_note(t: np.ndarray, cents: np.ndarray,
                     f_grid: np.ndarray | None = None,
                     min_cycles: float = 1.5,
                     min_samples: int = 8) -> Dict[str, float]:
    """The estimator the thesis specifies (draft eq:vibrato): a joint per-note
    NLLS fit of ``cents(t) = c + gamma * sin(2*pi*f*(t - delta))`` on the voiced
    samples of one note, with parameter variances and an identifiability flag.

    ``t`` is time in seconds relative to the note onset.  Implementation:
    a grid over the rate ``f`` (default 2.5–9 Hz, 66 points) with a closed-form
    linear solve for ``(c, a, b)`` in ``c + a sin(theta) + b cos(theta)`` at
    each grid point, a parabolic refinement of ``f`` around the grid optimum,
    then the Gauss–Newton covariance ``sigma^2 (J^T J)^{-1}`` at the optimum
    (``sigma^2 = SSE/(n-4)``); ``gamma = sqrt(a^2+b^2)`` and its variance by the
    delta method, ``delta`` from the phase (reported modulo one vibrato period).
    ``c`` here is the *vibrato-free centre*, NOT the curve mean — the mean
    coincides with the centre only over whole cycles, which a note rarely
    supplies (the thesis's explicit point).

    Identifiability rule: the vibrato channels ``(gamma, f, delta)`` are flagged
    unidentifiable when the note supplies fewer than ``min_samples`` voiced
    samples or fewer than ``min_cycles`` cycles at the fitted rate; ``c`` is
    then the plain mean with its standard error.  numpy-only, deterministic.

    Returns a dict: ``c, gamma, f, delta, var_c, var_gamma, var_f,
    vibrato_identifiable, sse, n``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(cents, dtype=float)
    n = x.size
    if n < 4:
        c = float(x.mean()) if n else 0.0
        v = float(x.var() / max(n, 1)) if n > 1 else np.inf
        return {"c": c, "gamma": 0.0, "f": 0.0, "delta": 0.0,
                "var_c": v, "var_gamma": np.inf, "var_f": np.inf,
                "vibrato_identifiable": False, "sse": 0.0, "n": int(n)}
    if f_grid is None:
        f_grid = np.linspace(2.5, 9.0, 66)

    def lin_solve(f):
        th = 2.0 * np.pi * f * t
        A = np.stack([np.ones(n), np.sin(th), np.cos(th)], axis=1)
        beta, *_ = np.linalg.lstsq(A, x, rcond=None)
        r = x - A @ beta
        return beta, float(r @ r)

    sses = np.empty(f_grid.size)
    betas = []
    for i, f in enumerate(f_grid):
        beta, sse = lin_solve(f)
        betas.append(beta)
        sses[i] = sse
    i0 = int(np.argmin(sses))
    f_hat = float(f_grid[i0])
    if 0 < i0 < f_grid.size - 1:  # parabolic refinement on the grid SSE
        y0, y1, y2 = sses[i0 - 1], sses[i0], sses[i0 + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom > 0:
            f_hat += 0.5 * (y0 - y2) / denom * (f_grid[1] - f_grid[0])
    (c_hat, a_hat, b_hat), sse = lin_solve(f_hat)
    gamma = float(np.hypot(a_hat, b_hat))
    phi = float(np.arctan2(-b_hat, a_hat))          # model = gamma sin(th - phi')
    delta = float((phi / (2.0 * np.pi * f_hat)) % (1.0 / f_hat)) if f_hat > 0 else 0.0

    # Gauss-Newton covariance at the optimum, params (c, a, b, f)
    th = 2.0 * np.pi * f_hat * t
    J = np.stack([np.ones(n), np.sin(th), np.cos(th),
                  (a_hat * np.cos(th) - b_hat * np.sin(th)) * 2.0 * np.pi * t],
                 axis=1)
    dof = max(n - 4, 1)
    sigma2 = sse / dof
    JtJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JtJ + 1e-10 * np.eye(4))
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.inf)
    var_c = float(cov[0, 0])
    g2 = max(gamma ** 2, 1e-12)
    var_gamma = float((a_hat ** 2 * cov[1, 1] + b_hat ** 2 * cov[2, 2]
                       + 2 * a_hat * b_hat * cov[1, 2]) / g2)
    var_f = float(cov[3, 3])

    span = float(t.max() - t.min()) if n else 0.0
    identifiable = bool(n >= min_samples and span * f_hat >= min_cycles
                        and np.isfinite(var_gamma))
    if not identifiable:
        c_hat, var_c = float(x.mean()), float(x.var(ddof=1) / n)
    return {"c": float(c_hat), "gamma": gamma, "f": f_hat, "delta": delta,
            "var_c": var_c, "var_gamma": var_gamma, "var_f": var_f,
            "vibrato_identifiable": identifiable, "sse": float(sse), "n": int(n)}


def extract_f0(audio: np.ndarray, sr: float):  # pragma: no cover - stub
    raise NotImplementedError(
        "f0 extraction stub. Wire in a monophonic pitch tracker (e.g. pYIN or CREPE) "
        "and return an f0 curve (Hz) on a fixed hop grid."
    )
