"""Phase-1 target extraction from aligned score-performance data.

These functions turn an alignment (score beats <-> performed seconds) into the
Phase-1 performance variables.  The tempo curve is estimated *here* (feature
extraction), so that, conditional on it, the onset residual is well defined and
the note-level field is jointly Gaussian (concept note, section 8.2).

Dataset loaders (ASAP / MAESTRO / ATEPP) are intentionally left as stubs: drop in
your local parsing and return arrays in these units.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .score import Score


def estimate_tempo_curve(beats: np.ndarray, perf_onset: np.ndarray, smooth: float = 0.0) -> np.ndarray:
    """Local seconds-per-beat at each note from an alignment.

    Estimated by finite differences of performed time vs score beats, then
    (optionally) smoothed.  Returns one value per note (length N).
    """
    beats = np.asarray(beats, dtype=float)
    perf_onset = np.asarray(perf_onset, dtype=float)
    order = np.argsort(beats)
    b, t = beats[order], perf_onset[order]
    spb = np.gradient(t, b)  # d(seconds)/d(beat)
    spb = np.clip(spb, 1e-3, None)
    if smooth > 0:
        spb = _moving_average(spb, smooth)
    out = np.empty_like(spb)
    out[order] = spb
    return out


def tempo_implied_onset(beats: np.ndarray, sec_per_beat: np.ndarray) -> np.ndarray:
    """Cumulative time warp T(b) = integral of seconds-per-beat over score time."""
    beats = np.asarray(beats, dtype=float)
    order = np.argsort(beats)
    b = beats[order]
    spb = np.asarray(sec_per_beat, dtype=float)[order]
    db = np.diff(b, prepend=b[0])
    t = np.cumsum(spb * db)
    t = t - t[0]
    out = np.empty_like(t)
    out[order] = t
    return out


def onset_residual(beats, perf_onset, sec_per_beat) -> np.ndarray:
    """tau_i = performed onset - tempo-implied onset (seconds)."""
    implied = tempo_implied_onset(beats, sec_per_beat)
    implied = implied - implied.mean() + np.asarray(perf_onset, float).mean()
    return np.asarray(perf_onset, dtype=float) - implied


def articulation_ratio(score: Score, perf_duration, sec_per_beat) -> np.ndarray:
    """log r_i = log(performed duration / nominal duration)."""
    nominal = score.duration * np.asarray(sec_per_beat, dtype=float)
    nominal = np.clip(nominal, 1e-6, None)
    return np.log(np.clip(np.asarray(perf_duration, float), 1e-6, None) / nominal)


def normalize_velocity(midi_velocity, lo: float = 0.0, hi: float = 127.0) -> np.ndarray:
    """Map MIDI velocity to a centered, roughly unit-scale dynamics variable."""
    v = (np.asarray(midi_velocity, dtype=float) - lo) / (hi - lo)
    return (v - v.mean())


def _moving_average(x: np.ndarray, window: float) -> np.ndarray:
    k = max(1, int(window))
    if k == 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


# --- dataset loaders (stubs) ---------------------------------------------
def load_asap(path: str) -> Tuple[Score, dict]:  # pragma: no cover - stub
    raise NotImplementedError(
        "ASAP loader stub. Parse MusicXML/MIDI scores + aligned performance MIDI and "
        "return (Score, {'perf_onset', 'perf_duration', 'velocity', 'beats'})."
    )


def load_maestro(path: str) -> Tuple[Score, dict]:  # pragma: no cover - stub
    raise NotImplementedError(
        "MAESTRO loader stub. MAESTRO has aligned audio+MIDI but no symbolic score; "
        "pair with ASAP scores or a separate score source."
    )
