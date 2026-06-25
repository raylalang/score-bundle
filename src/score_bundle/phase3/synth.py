"""Minimal additive harmonic synthesizer (the deterministic basis Phi(z)).

Phase coherence (pitch) lives in this deterministic basis; the harmonic amplitudes
stay Gaussian and enter linearly.  This NumPy version is a single-note,
constant-/curve-f0 demonstrator of the forward map and the design matrix Phi(z);
a production version would be implemented in an autodiff framework and place many
notes on a shared time axis.
"""
from __future__ import annotations

import numpy as np


def instantaneous_f0(semitone: float, f_ref: float, cents: float = 0.0) -> float:
    """f0 = f_ref * 2**((semitone + cents/100)/12)."""
    return float(f_ref * 2.0 ** ((semitone + cents / 100.0) / 12.0))


def cumulative_phase(f0_curve: np.ndarray, t: np.ndarray, harmonic: int) -> np.ndarray:
    """phi_k(t) = 2 pi k * integral_0^t f0(tau) dtau (trapezoidal)."""
    f0_curve = np.asarray(f0_curve, dtype=float)
    t = np.asarray(t, dtype=float)
    dt = np.diff(t, prepend=t[0])
    integral = np.cumsum(f0_curve * dt)
    return 2.0 * np.pi * harmonic * integral


def harmonic_design_matrix(f0_curve: np.ndarray, t: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Quadrature basis Phi(z): columns [cos phi_k, sin phi_k] for k=1..K.

    Shape (len(t), 2 * n_harmonics).  Amplitudes a multiply these columns:
    x = Phi a (+ noise).
    """
    cols = []
    for k in range(1, n_harmonics + 1):
        phi = cumulative_phase(f0_curve, t, k)
        cols.append(np.cos(phi))
        cols.append(np.sin(phi))
    return np.stack(cols, axis=1)


def synthesize(f0_curve: np.ndarray, amps: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Forward synthesis x = Phi(z) a for given harmonic amplitudes ``amps``."""
    Phi = harmonic_design_matrix(f0_curve, t, n_harmonics=len(amps) // 2)
    return Phi @ np.asarray(amps, dtype=float)
