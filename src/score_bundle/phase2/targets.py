"""Per-note Phase-2 targets from a tracked f0 curve (real-audio pipeline).

Bridges the tracker (:func:`intonation.extract_f0` output, or URMP's
ground-truth F0 annotations in the same shape) and the thesis estimator
(:func:`intonation.fit_vibrato_note`): assign voiced frames to notes, convert
to cents against each note's equal-tempered target, run the per-note NLLS,
and return the channel arrays the cell-mask GP consumes —
``[c, log gamma, log f]`` with delta-method variances and the
identifiability mask, exactly the synthetic pilot's convention
(scripts/eval_phase2_synthetic.py).

Frame filter: per the measured tracker calibration
(results/tracker_calibration_dev.md), frames in the lowest confidence
quantile are discarded before fitting (``conf_quantile``); ground-truth
curves carry no confidence, so pass ``prob=None`` to skip the filter.
numpy-only.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .intonation import cents_from_f0, fit_vibrato_note

F_REF = 440.0


def hz_to_semitone(f_hz: np.ndarray) -> np.ndarray:
    """Nearest equal-tempered MIDI number for a (positive) frequency."""
    return np.round(69.0 + 12.0 * np.log2(np.asarray(f_hz, dtype=float)
                                          / F_REF)).astype(int)


def note_targets(t: np.ndarray, f0: np.ndarray, voiced: np.ndarray,
                 prob: Optional[np.ndarray], onset: np.ndarray,
                 duration: np.ndarray, pitch_hz: np.ndarray,
                 conf_quantile: float = 0.2,
                 min_frames: int = 4) -> Dict[str, np.ndarray]:
    """Per-note targets ``[c, log gamma, log f]`` from one track's f0 curve.

    Returns dict of arrays over the N notes: ``est`` (N, 3) with NaN where a
    channel is unavailable, ``var`` (N, 3) delta-method variances (log-space
    for the vibrato channels), ``ident`` (N,) the estimator's vibrato
    identifiability, ``n_frames`` (N,), ``midi`` (N,).  The observation rule
    for the GP: intonation ``c`` is observed where ``n_frames >= min_frames``;
    the vibrato channels where additionally ``ident``.
    """
    t = np.asarray(t, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (f0 > 0)
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        floor = np.quantile(prob[voiced], conf_quantile) if voiced.any() else 0.0
        voiced = voiced & (prob >= floor)
    onset = np.asarray(onset, dtype=float)
    duration = np.asarray(duration, dtype=float)
    midi = hz_to_semitone(pitch_hz)

    n = onset.size
    est = np.full((n, 3), np.nan)
    var = np.full((n, 3), np.nan)
    ident = np.zeros(n, dtype=bool)
    n_frames = np.zeros(n, dtype=int)
    for i in range(n):
        sel = voiced & (t >= onset[i]) & (t < onset[i] + duration[i])
        n_frames[i] = int(sel.sum())
        if n_frames[i] < min_frames:
            continue
        cents = cents_from_f0(f0[sel], F_REF, float(midi[i] - 69))
        out = fit_vibrato_note(t[sel] - onset[i], cents)
        est[i, 0], var[i, 0] = out["c"], out["var_c"]
        ident[i] = out["vibrato_identifiable"]
        if ident[i]:
            g = max(out["gamma"], 1e-6)
            f = max(out["f"], 1e-6)
            est[i, 1], var[i, 1] = np.log(g), out["var_gamma"] / g ** 2
            est[i, 2], var[i, 2] = np.log(f), out["var_f"] / f ** 2
    return {"est": est, "var": var, "ident": ident,
            "n_frames": n_frames, "midi": midi}
