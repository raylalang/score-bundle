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


def extract_f0(audio: np.ndarray, sr: float):  # pragma: no cover - stub
    raise NotImplementedError(
        "f0 extraction stub. Wire in a monophonic pitch tracker (e.g. pYIN or CREPE) "
        "and return an f0 curve (Hz) on a fixed hop grid."
    )
