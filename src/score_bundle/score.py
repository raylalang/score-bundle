"""Score representation: discrete note events that anchor the continuous fibers.

A score is the *given* support  S = {(p_i, b_i, d_i)}  (pitch degree, beat onset,
beat duration).  Everything continuous (timing, dynamics, intonation, ...) is a
performance variable attached to these nodes -- see :mod:`score_bundle.variables`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Note:
    """A single score event.

    Attributes
    ----------
    pitch:    pitch degree p_i (e.g. MIDI number, semitone index from a reference).
    onset:    beat onset b_i in *score time* (beats), >= 0.
    duration: beat duration d_i in score time (beats), > 0.
    voice:    optional voice / staff id used for graph construction.
    """

    pitch: int
    onset: float
    duration: float
    voice: int = 0


class Score:
    """An ordered collection of :class:`Note` objects with vectorized accessors."""

    def __init__(self, notes: Sequence[Note]):
        self.notes: List[Note] = list(notes)

    def __len__(self) -> int:
        return len(self.notes)

    def __iter__(self):
        return iter(self.notes)

    # --- vectorized views -------------------------------------------------
    @property
    def pitch(self) -> np.ndarray:
        return np.array([n.pitch for n in self.notes], dtype=float)

    @property
    def onset(self) -> np.ndarray:
        return np.array([n.onset for n in self.notes], dtype=float)

    @property
    def duration(self) -> np.ndarray:
        return np.array([n.duration for n in self.notes], dtype=float)

    @property
    def voice(self) -> np.ndarray:
        return np.array([n.voice for n in self.notes], dtype=int)

    @classmethod
    def from_arrays(
        cls,
        pitch: Sequence[float],
        onset: Sequence[float],
        duration: Sequence[float],
        voice: Optional[Sequence[int]] = None,
    ) -> "Score":
        pitch = np.asarray(pitch)
        onset = np.asarray(onset)
        duration = np.asarray(duration)
        n = len(pitch)
        if voice is None:
            voice = np.zeros(n, dtype=int)
        notes = [
            Note(int(pitch[i]), float(onset[i]), float(duration[i]), int(voice[i]))
            for i in range(n)
        ]
        return cls(notes)
