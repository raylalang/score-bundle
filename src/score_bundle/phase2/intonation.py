"""Intonation / vibrato feature extraction (Phase 2).

`extract_f0` tracks monophonic audio with probabilistic YIN (librosa,
import-guarded); `cents_from_f0` converts to cents; `fit_vibrato_note` is the
per-note NLLS estimator of the evaluated bundle (ungated), and
`fit_vibrato_note_gated` the eq:vibrato-exact gated form that supplies the
vibrato onset delay.

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


def fit_vibrato_note_gated(t: np.ndarray, cents: np.ndarray,
                           f_grid: np.ndarray | None = None,
                           n_delta: int = 13,
                           min_cycles: float = 1.5,
                           min_samples: int = 8) -> Dict[str, float]:
    """The GATED estimator matching draft eq:vibrato exactly:

        cents(t) = c + [t >= delta] * gamma * sin(2*pi*f*(t - delta)),

    i.e. the pitch curve is flat at the centre until the vibrato onset delay
    ``delta`` and oscillates after it, phase pinned to zero at the gate (the
    oscillation grows out of the centre).  This is the model the thesis
    equation specifies; :func:`fit_vibrato_note` (the evaluated-bundle
    estimator for ``[c, gamma, f]``) fits the UNGATED form, whose ``delta``
    is a phase reported modulo one period, not an onset delay.  This gated
    variant exists to answer the delta_vib channel-candidate question on
    development data (docs/phase2_prereg_design.md).

    Implementation: a joint grid over rate ``f`` (default 2.5-9 Hz, 66 pts)
    and delay ``delta`` (``n_delta`` points spanning [0, 0.6 * span]), with a
    closed-form 2-parameter solve for ``(c, gamma)`` at each grid point
    (basis 1 and the gated sinusoid); Gauss-Newton covariance at the optimum
    over ``(c, gamma, f, delta)`` (gate treated as fixed, the standard smooth
    approximation); a negative fitted ``gamma`` (vibrato starting downward)
    is folded into ``delta + T/2`` when that stays inside the note, else the
    fit is flagged unidentifiable.

    Identifiability of the DELAY (``delta_identifiable``): the vibrato rule
    (>= ``min_samples`` post-gate samples and >= ``min_cycles`` cycles after
    the gate) plus a finite delta variance and a gate that is not pinned to
    the upper grid edge.  ``delta = 0`` (no delay) is a legitimate
    identifiable value.  numpy-only, deterministic.

    Returns ``c, gamma, f, delta, var_c, var_gamma, var_f, var_delta,
    vibrato_identifiable, delta_identifiable, sse, n``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(cents, dtype=float)
    n = x.size
    bad = {"c": float(x.mean()) if n else 0.0, "gamma": 0.0, "f": 0.0,
           "delta": 0.0, "var_c": float(x.var() / max(n, 1)) if n > 1 else np.inf,
           "var_gamma": np.inf, "var_f": np.inf, "var_delta": np.inf,
           "vibrato_identifiable": False, "delta_identifiable": False,
           "sse": 0.0, "n": int(n)}
    if n < 4:
        return bad
    if f_grid is None:
        f_grid = np.linspace(2.5, 9.0, 66)
    span = float(t.max() - t.min())
    d_grid = np.linspace(0.0, 0.6 * span, n_delta) + float(t.min())

    best = (np.inf, None)                     # (sse, (c, g, f, d))
    for f in f_grid:
        for d in d_grid:
            gate = t >= d
            if gate.sum() < min_samples or (t.max() - d) * f < min_cycles:
                continue
            s = np.where(gate, np.sin(2.0 * np.pi * f * (t - d)), 0.0)
            # closed-form OLS for x ~ c + g*s
            sm, xm = s.mean(), x.mean()
            sv = float(s @ s) / n - sm * sm
            if sv < 1e-12:
                continue
            g = (float(s @ x) / n - sm * xm) / sv
            c = xm - g * sm
            r = x - c - g * s
            sse = float(r @ r)
            if sse < best[0]:
                best = (sse, (c, g, float(f), float(d)))
    if best[1] is None:
        return bad
    sse, (c_hat, g_hat, f_hat, d_hat) = best
    if g_hat < 0:                              # downward start = delta + T/2
        d_flip = d_hat + 0.5 / f_hat
        if (t.max() - d_flip) * f_hat >= min_cycles \
                and (t >= d_flip).sum() >= min_samples:
            g_hat, d_hat = -g_hat, d_flip
            gate = t >= d_hat
            s = np.where(gate, np.sin(2.0 * np.pi * f_hat * (t - d_hat)), 0.0)
            c_hat = float(x.mean() - g_hat * s.mean())
            r = x - c_hat - g_hat * s
            sse = float(r @ r)
        else:
            return bad

    gate = t >= d_hat
    th = 2.0 * np.pi * f_hat * (t - d_hat)
    s = np.where(gate, np.sin(th), 0.0)
    cth = np.where(gate, np.cos(th), 0.0)
    J = np.stack([np.ones(n), s,
                  g_hat * cth * 2.0 * np.pi * (t - d_hat),
                  -g_hat * cth * 2.0 * np.pi * f_hat], axis=1)
    dof = max(n - 4, 1)
    sigma2 = sse / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J + 1e-10 * np.eye(4))
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.inf)
    var_c, var_gamma, var_f, var_delta = (float(cov[i, i]) for i in range(4))

    post = int(gate.sum())
    vib_ok = bool(post >= min_samples
                  and (t.max() - d_hat) * f_hat >= min_cycles
                  and np.isfinite(var_gamma))
    at_edge = d_hat >= d_grid[-1] - 1e-12
    delta_ok = bool(vib_ok and np.isfinite(var_delta) and not at_edge)
    if not vib_ok:
        return bad
    return {"c": float(c_hat), "gamma": float(g_hat), "f": f_hat,
            "delta": float(d_hat - t.min()), "var_c": var_c,
            "var_gamma": var_gamma, "var_f": var_f, "var_delta": var_delta,
            "vibrato_identifiable": vib_ok, "delta_identifiable": delta_ok,
            "sse": float(sse), "n": int(n)}


def extract_f0(audio: np.ndarray, sr: float, hop_s: float = 0.010,
               fmin: float = 65.0, fmax: float = 2093.0) -> Dict[str, np.ndarray]:
    """Monophonic f0 tracking via probabilistic YIN (librosa ``pyin``).

    Returns a dict of aligned per-frame arrays on a fixed hop grid:

    - ``t``      frame times in seconds (hop ``hop_s``);
    - ``f0``     fundamental in Hz (NaN on unvoiced frames — callers must
      restrict to voiced frames, per the thesis: targets are fit on the
      samples the tracker marks as voiced);
    - ``voiced`` boolean voicing decision;
    - ``prob``   the tracker's per-frame voicing probability, the confidence
      that feeds the heteroscedastic observation noise of the Phase-2
      likelihood (draft eq:phase2-noise).

    ``librosa`` is an optional dependency (import-guarded, like scipy/torch
    elsewhere): the numpy core never imports it, and this function raises a
    clear ImportError when it is missing.  Deterministic for fixed inputs
    (pyin is a Viterbi decode, no sampling).  Default range covers C2–C7;
    widen ``fmin``/``fmax`` per instrument, and keep ``hop_s`` at 10 ms so
    vibrato at up to ~9 Hz (the ``fit_vibrato_note`` grid) is sampled with
    >10 frames per cycle.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "extract_f0 needs librosa (pip install librosa); the numpy core "
            "does not depend on it") from exc
    audio = np.asarray(audio, dtype=float)
    hop_length = max(int(round(hop_s * sr)), 1)
    # frame_length must cover ~2 periods of fmin for YIN's difference function
    frame_length = int(2 ** np.ceil(np.log2(2.0 * sr / fmin)))
    f0, voiced, prob = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length, center=True)
    t = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return {"t": np.asarray(t, dtype=float),
            "f0": np.asarray(f0, dtype=float),
            "voiced": np.asarray(voiced, dtype=bool),
            "prob": np.asarray(prob, dtype=float)}
